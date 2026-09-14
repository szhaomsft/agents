from types import SimpleNamespace

import pytest

from livekit.agents import utils
from livekit.plugins.azure import tts as azure_tts

pytestmark = pytest.mark.plugin("azure")


class _EventSignal:
    def __init__(self) -> None:
        self.callbacks = []

    def connect(self, callback) -> None:
        self.callbacks.append(callback)

    def emit(self, event) -> None:
        for callback in self.callbacks:
            callback(event)


class _InputStream:
    def write(self, text: str) -> None:
        pass

    def close(self) -> None:
        pass


class _SynthesisRequest:
    def __init__(self, *, input_type) -> None:
        self.input_stream = _InputStream()


class _OutputEmitter:
    def __init__(self) -> None:
        self.audio = []
        self.segment_ended = False

    def start_segment(self, *, segment_id: str) -> None:
        pass

    def push(self, audio: bytes) -> None:
        self.audio.append(audio)

    def end_segment(self) -> None:
        self.segment_ended = True


async def test_stream_forwards_audio_queued_after_completion_callback(monkeypatch) -> None:
    synthesizing = _EventSignal()
    synthesis_completed = _EventSignal()
    synthesis_canceled = _EventSignal()

    class _ResultFuture:
        def get(self):
            synthesizing.emit(SimpleNamespace(result=SimpleNamespace(audio_data=b"first")))
            synthesis_completed.emit(SimpleNamespace(result=SimpleNamespace()))
            synthesizing.emit(SimpleNamespace(result=SimpleNamespace(audio_data=b"last")))
            return SimpleNamespace(audio_data=b"firstlasttail")

    synthesizer = SimpleNamespace(
        synthesizing=synthesizing,
        synthesis_completed=synthesis_completed,
        synthesis_canceled=synthesis_canceled,
        speak_async=lambda request: _ResultFuture(),
    )

    monkeypatch.setattr(azure_tts.speechsdk, "SpeechSynthesisRequest", _SynthesisRequest)

    stream = azure_tts.SynthesizeStream.__new__(azure_tts.SynthesizeStream)
    stream._tts = SimpleNamespace(_synthesizer=synthesizer)
    stream._text_ch = utils.aio.Chan()
    stream._text_ch.send_nowait("hello")
    stream._text_ch.send_nowait(None)
    stream._text_ch.close()
    stream._mark_started = lambda: None

    emitter = _OutputEmitter()
    await stream._synthesize_segment(emitter)

    assert emitter.audio == [b"first", b"last", b"tail"]
    assert emitter.segment_ended
