from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from types import SimpleNamespace

import pytest

from livekit.agents import APIConnectOptions
from livekit.plugins.azure import tts as azure_tts

pytestmark = pytest.mark.plugin("azure")


class _Signal:
    def __init__(self) -> None:
        self.callbacks: list[Callable] = []
        self.on_disconnect: Callable | None = None

    def connect(self, callback: Callable) -> None:
        self.callbacks.append(callback)

    def disconnect_all(self) -> None:
        if self.on_disconnect:
            self.on_disconnect()
        self.callbacks.clear()

    def emit(self, result: SimpleNamespace) -> None:
        for callback in list(self.callbacks):
            callback(SimpleNamespace(result=result))


class _Request:
    def __init__(self, *, input_type: object) -> None:
        self.text = ""
        self.closed = False
        self.input_stream = self

    def write(self, text: str) -> None:
        self.text += text

    def close(self) -> None:
        self.closed = True


class _Synthesizer:
    def __init__(self, barrier: threading.Barrier, fail: bool = False) -> None:
        self.synthesizing = _Signal()
        self.synthesis_completed = _Signal()
        self.synthesis_canceled = _Signal()
        self.barrier = barrier
        self.fail = fail
        self.requests: list[_Request] = []

    def speak_async(self, request: _Request) -> SimpleNamespace:
        self.requests.append(request)

        def get() -> SimpleNamespace:
            assert request.closed
            if self.fail:
                raise RuntimeError("synthesis failed")
            result = SimpleNamespace(
                reason=azure_tts.speechsdk.ResultReason.SynthesizingAudioCompleted
            )
            if request.text == "Warm up.":
                return result
            self.barrier.wait(timeout=5)
            sample = b"\x01\x00" if request.text == "first" else b"\x02\x00"
            self.synthesizing.emit(SimpleNamespace(audio_data=sample * 4800))
            self.synthesis_completed.emit(result)
            self.synthesizing.emit(SimpleNamespace(audio_data=sample * 73))
            result.audio_data = sample * (4800 + 73 + 91)
            late_callbacks = list(self.synthesizing.callbacks)

            def late_audio() -> None:
                for callback in late_callbacks:
                    callback(SimpleNamespace(result=SimpleNamespace(audio_data=sample * 91)))

            self.synthesizing.on_disconnect = late_audio
            return result

        return SimpleNamespace(get=get)


async def test_overlapping_streams_keep_their_own_audio_and_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    barrier = threading.Barrier(2)
    synthesizers: list[_Synthesizer] = []

    def create(stream: azure_tts.SynthesizeStream) -> _Synthesizer:
        synth = _Synthesizer(barrier)
        synthesizers.append(synth)
        return synth

    monkeypatch.setattr(azure_tts.SynthesizeStream, "_create_synthesizer", create)
    monkeypatch.setattr(azure_tts.speechsdk, "SpeechSynthesisRequest", _Request)
    provider = azure_tts.TTS(speech_key="test", speech_region="eastus")

    async def collect(text: str) -> tuple[bytes, int]:
        async with provider.stream(conn_options=APIConnectOptions(max_retry=0)) as stream:
            stream.push_text(text)
            stream.end_input()
            frames = [event async for event in stream]
            return b"".join(bytes(event.frame.data) for event in frames), sum(
                event.is_final for event in frames
            )

    try:
        first, second = await asyncio.wait_for(
            asyncio.gather(collect("first"), collect("second")), timeout=10
        )
        assert first == (b"\x01\x00" * 4964, 1)
        assert second == (b"\x02\x00" * 4964, 1)
        assert len(synthesizers) == 2
        for synth in synthesizers:
            assert not synth.synthesizing.callbacks
            assert not synth.synthesis_completed.callbacks
            assert not synth.synthesis_canceled.callbacks
    finally:
        await provider.aclose()


async def test_sdk_failure_disconnects_callbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    synth = _Synthesizer(threading.Barrier(1), fail=True)
    monkeypatch.setattr(azure_tts.SynthesizeStream, "_create_synthesizer", lambda stream: synth)
    monkeypatch.setattr(azure_tts.speechsdk, "SpeechSynthesisRequest", _Request)
    provider = azure_tts.TTS(speech_key="test", speech_region="eastus")
    try:
        async with provider.stream(conn_options=APIConnectOptions(max_retry=0)) as stream:
            stream.push_text("first")
            stream.end_input()
            with pytest.raises(azure_tts.APIConnectionError, match="synthesis failed"):
                async for _ in stream:
                    pass
        assert not synth.synthesizing.callbacks
        assert not synth.synthesis_canceled.callbacks
    finally:
        await provider.aclose()


@pytest.mark.parametrize("result_kind", ["complete", "mismatch", "missing"])
async def test_final_result_reconciliation(
    monkeypatch: pytest.MonkeyPatch, result_kind: str
) -> None:
    class _ResultSynthesizer(_Synthesizer):
        def speak_async(self, request: _Request) -> SimpleNamespace:
            future = super().speak_async(request)

            def get() -> SimpleNamespace:
                result = future.get()
                if result_kind == "complete":
                    self.synthesizing.emit(SimpleNamespace(audio_data=result.audio_data[-182:]))
                elif result_kind == "mismatch":
                    result.audio_data = b"\x03\x00" + result.audio_data[2:]
                else:
                    result.audio_data = b""
                return result

            return SimpleNamespace(get=get)

    synth = _ResultSynthesizer(threading.Barrier(1))
    monkeypatch.setattr(azure_tts.SynthesizeStream, "_create_synthesizer", lambda stream: synth)
    monkeypatch.setattr(azure_tts.speechsdk, "SpeechSynthesisRequest", _Request)
    provider = azure_tts.TTS(speech_key="test", speech_region="eastus")
    try:
        async with provider.stream(conn_options=APIConnectOptions(max_retry=0)) as stream:
            stream.push_text("first")
            stream.end_input()
            if result_kind == "complete":
                pcm = b"".join([bytes(event.frame.data) async for event in stream])
                assert pcm == b"\x01\x00" * 4964
            else:
                with pytest.raises(azure_tts.APIConnectionError, match="callback audio differs"):
                    async for _ in stream:
                        pass
        assert not synth.synthesizing.callbacks
        assert not synth.synthesis_canceled.callbacks
    finally:
        await provider.aclose()


async def test_cancelling_one_stream_does_not_stop_another(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    stopped = threading.Event()

    class _BlockedSynthesizer(_Synthesizer):
        def speak_async(self, request: _Request) -> SimpleNamespace:
            def get() -> SimpleNamespace:
                started.set()
                assert stopped.wait(timeout=5)
                return SimpleNamespace(reason=azure_tts.speechsdk.ResultReason.Canceled)

            return SimpleNamespace(get=get)

        def stop_speaking_async(self) -> SimpleNamespace:
            stopped.set()
            return SimpleNamespace(get=lambda: None)

    blocked = _BlockedSynthesizer(threading.Barrier(1))
    healthy = _Synthesizer(threading.Barrier(1))
    available = iter([blocked, healthy])
    monkeypatch.setattr(
        azure_tts.SynthesizeStream, "_create_synthesizer", lambda stream: next(available)
    )
    monkeypatch.setattr(azure_tts.speechsdk, "SpeechSynthesisRequest", _Request)
    provider = azure_tts.TTS(speech_key="test", speech_region="eastus")
    try:
        first = provider.stream(conn_options=APIConnectOptions(max_retry=0))
        first.push_text("first")
        first.end_input()
        assert await asyncio.to_thread(started.wait, 5)
        async with provider.stream(conn_options=APIConnectOptions(max_retry=0)) as second:
            second.push_text("second")
            second.end_input()
            await first.aclose()
            pcm = b"".join([bytes(event.frame.data) async for event in second])
            assert pcm == b"\x02\x00" * 4964
        assert stopped.is_set()
    finally:
        stopped.set()
        await provider.aclose()


class _Connection:
    def __init__(self, synthesizer: _Synthesizer) -> None:
        self.synthesizer = synthesizer
        self.open_calls: list[bool] = []
        self.close_calls = 0

    def open(self, continuous: bool) -> None:
        self.open_calls.append(continuous)

    def close(self) -> None:
        self.close_calls += 1


@pytest.fixture
def connections(monkeypatch: pytest.MonkeyPatch) -> list[_Connection]:
    created: list[_Connection] = []

    def connect(synthesizer: _Synthesizer) -> _Connection:
        connection = _Connection(synthesizer)
        created.append(connection)
        return connection

    monkeypatch.setattr(
        azure_tts.TTS,
        "_create_synthesizer",
        lambda provider, opts: _Synthesizer(threading.Barrier(1)),
    )
    monkeypatch.setattr(azure_tts.speechsdk.Connection, "from_speech_synthesizer", connect)
    monkeypatch.setattr(azure_tts.speechsdk, "SpeechSynthesisRequest", _Request)
    return created


async def test_prewarm_is_exclusive_and_replenished(connections: list[_Connection]) -> None:
    provider = azure_tts.TTS(speech_key="test", speech_region="eastus")
    try:
        provider.prewarm()
        provider.prewarm()
        assert len(connections) == 1
        assert connections[0].open_calls == [True]
        assert not connections[0].synthesizer.requests

        first = provider.stream()
        second = provider.stream()
        assert first._synthesizer is connections[0].synthesizer
        assert second._synthesizer is connections[1].synthesizer
        assert first._synthesizer is not second._synthesizer

        async def collect(stream: azure_tts.SynthesizeStream, text: str) -> bytes:
            async with stream:
                stream.push_text(text)
                stream.end_input()
                return b"".join([bytes(event.frame.data) async for event in stream])

        pcm = await asyncio.gather(collect(first, "first"), collect(second, "second"))
        assert pcm == [b"\x01\x00" * 4964, b"\x02\x00" * 4964]
        assert len(connections) == 3
        assert connections[0].close_calls == connections[1].close_calls == 1
        assert not connections[2].synthesizer.requests
        assert connections[2].close_calls == 0

        third = provider.stream()
        assert third._synthesizer is connections[2].synthesizer
        assert await collect(third, "first") == b"\x01\x00" * 4964
        assert len(connections) == 4
        await third.aclose()
        assert connections[2].close_calls == 1
        assert len(connections) == 4
    finally:
        await provider.aclose()
    assert all(connection.close_calls == 1 for connection in connections)
    provider.prewarm()
    assert len(connections) == 4


@pytest.mark.parametrize("prewarm_again", [True, False])
async def test_voice_change_discards_idle_connection(
    connections: list[_Connection], prewarm_again: bool
) -> None:
    provider = azure_tts.TTS(speech_key="test", speech_region="eastus")
    try:
        provider.prewarm()
        provider.update_options(voice="en-US-Tessa:DragonHDLatestNeural")
        if prewarm_again:
            provider.prewarm()
        async with provider.stream() as stream:
            assert connections[0].close_calls == 1
            assert stream._synthesizer is connections[1].synthesizer
            assert stream._opts.voice == "en-US-Tessa:DragonHDLatestNeural"
            stream.push_text("first")
            stream.end_input()
            assert (
                b"".join([bytes(event.frame.data) async for event in stream]) == b"\x01\x00" * 4964
            )
    finally:
        await provider.aclose()
    assert all(connection.close_calls == 1 for connection in connections)
