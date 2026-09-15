"""Exercise Python-console playback without a sound device or a TTS provider."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import numpy as np
import pytest

from livekit import rtc
from livekit.agents.cli import _legacy
from livekit.agents.utils import aio
from livekit.agents.voice.io import PlaybackFinishedEvent

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time]

RATE = 24000
BLOCK_SAMPLES = 2400


class _ConsoleHarness:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.loop = asyncio.get_running_loop()
        # Scope the clock replacement to this module, not Python's time module.
        # Virtual-time tests can then drive the device deadline deterministically.
        monkeypatch.setattr(_legacy, "time", SimpleNamespace(monotonic=self.loop.time))
        # Keep device-deadline cases independent of the initial prebuffer.
        monkeypatch.setattr(_legacy, "PREBUFFER_DURATION", 0)
        self.output = _legacy.ConsoleAudioOutput(self.loop)
        self.console = _legacy.AgentsConsole.__new__(_legacy.AgentsConsole)
        self.console._lock = threading.Lock()
        self.console._io_acquired = True
        self.console._io_loop = self.loop
        self.console._io_audio_output = self.output
        self.console._apm = MagicMock()
        self.tasks: list[asyncio.Task[PlaybackFinishedEvent]] = []

    async def push(self, samples: int, *, value: int = 1) -> None:
        data = np.full(samples, value, dtype=np.int16)
        await self.output.capture_frame(
            rtc.AudioFrame(
                data=data.tobytes(),
                sample_rate=RATE,
                num_channels=1,
                samples_per_channel=samples,
            )
        )

    def render(self, *, delay: float = 0.125) -> np.ndarray:
        data = np.full((BLOCK_SAMPLES, 1), -1, dtype=np.int16)
        # PortAudio uses its own clock; only the delta is transferable to the
        # console's monotonic clock. Make an accidental clock mix-up visible.
        device_now = 1000.0 + self.loop.time()
        timestamp = SimpleNamespace(currentTime=device_now, outputBufferDacTime=device_now + delay)
        self.console._sd_output_callback(data, BLOCK_SAMPLES, timestamp)
        return data[:, 0]

    def wait_for_playout(self) -> asyncio.Task[PlaybackFinishedEvent]:
        task = asyncio.create_task(self.output.wait_for_playout())
        self.tasks.append(task)
        return task

    async def close(self) -> None:
        await aio.cancel_and_wait(*self.tasks)
        if self.output._flush_task is not None:
            await aio.cancel_and_wait(self.output._flush_task)


@pytest.fixture
async def console(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[_ConsoleHarness]:
    harness = _ConsoleHarness(monkeypatch)
    try:
        yield harness
    finally:
        await harness.close()


@pytest.mark.parametrize("samples", [777, BLOCK_SAMPLES])
@pytest.mark.parametrize("flush_before_render", [False, True])
async def test_completion_waits_for_final_sample_on_device(
    console: _ConsoleHarness, samples: int, flush_before_render: bool
) -> None:
    await console.push(samples)
    if flush_before_render:
        console.output.flush()

    started = console.loop.time()
    rendered = console.render()
    if not flush_before_render:
        console.output.flush()
    playout = console.wait_for_playout()

    assert np.all(rendered[:samples] == 1)
    assert np.all(rendered[samples:] == 0)
    assert not console.output._output_buf
    duration = 0.125 + samples / RATE
    await asyncio.sleep(duration - 0.001)
    assert not playout.done(), "empty Python buffer was mistaken for finished device playback"

    event = await asyncio.wait_for(playout, timeout=1)
    assert console.loop.time() - started == pytest.approx(duration, abs=1e-5)
    assert event.playback_position == pytest.approx(samples / RATE)
    assert not event.interrupted


async def test_idle_callbacks_do_not_extend_the_audio_deadline(console: _ConsoleHarness) -> None:
    await console.push(480)
    started = console.loop.time()
    console.render()
    console.output.flush()
    playout = console.wait_for_playout()

    await asyncio.sleep(0.01)
    assert np.all(console.render(delay=10) == 0)
    event = await asyncio.wait_for(playout, timeout=1)
    assert console.loop.time() - started == pytest.approx(0.145, abs=1e-5)
    assert not event.interrupted


@pytest.mark.parametrize("delay", [0, -0.2])
async def test_nonpositive_device_latency_still_waits_for_the_final_block(
    console: _ConsoleHarness, delay: float
) -> None:
    await console.push(BLOCK_SAMPLES)
    started = console.loop.time()
    console.render(delay=delay)
    console.output.flush()
    await asyncio.wait_for(console.wait_for_playout(), timeout=1)
    assert console.loop.time() - started == pytest.approx(0.1, abs=1e-5)


async def test_late_empty_notification_does_not_finish_newly_buffered_audio(
    console: _ConsoleHarness,
) -> None:
    await console.push(480)
    console.render(delay=0)
    # The callback's empty notification is queued on the loop, but more text's
    # audio arrives before it runs. Flush must still wait for this new audio.
    await console.push(777, value=2)
    console.output.flush()
    playout = console.wait_for_playout()
    await asyncio.sleep(0.05)
    assert not playout.done()

    started = console.loop.time()
    rendered = console.render()
    assert np.all(rendered[:777] == 2)
    await asyncio.wait_for(playout, timeout=1)
    assert console.loop.time() - started == pytest.approx(0.125 + 777 / RATE, abs=1e-5)


async def test_pause_preserves_pending_tail_until_resume(console: _ConsoleHarness) -> None:
    await console.push(BLOCK_SAMPLES + 777)
    console.output.flush()
    assert np.all(console.render(delay=0) == 1)
    console.output.pause()
    playout = console.wait_for_playout()

    await asyncio.sleep(0.2)
    assert np.all(console.render() == 0)
    assert len(console.output._output_buf) == 777 * 2
    assert not playout.done()

    console.output.resume()
    started = console.loop.time()
    rendered = console.render()
    assert np.all(rendered[:777] == 1)
    event = await asyncio.wait_for(playout, timeout=1)
    assert console.loop.time() - started == pytest.approx(0.125 + 777 / RATE, abs=1e-5)
    assert event.playback_position == pytest.approx((BLOCK_SAMPLES + 777) / RATE)
    assert not event.interrupted


@pytest.mark.parametrize("samples", [BLOCK_SAMPLES, BLOCK_SAMPLES + 480])
async def test_interruption_does_not_wait_for_device_deadline(
    console: _ConsoleHarness, samples: int
) -> None:
    await console.push(samples)
    console.render(delay=10)
    console.output.flush()
    playout = console.wait_for_playout()
    await asyncio.sleep(0.01)
    started = console.loop.time()
    console.output.clear_buffer()
    event = await asyncio.wait_for(playout, timeout=1)
    assert console.loop.time() - started < 0.001
    assert event.interrupted
    assert not console.output._output_buf
    assert console.output._playback_end_at == 0

    # A retired segment's device deadline must not delay a subsequent segment.
    await console.push(480, value=2)
    started = console.loop.time()
    console.render(delay=0)
    console.output.flush()
    next_event = await asyncio.wait_for(console.wait_for_playout(), timeout=1)
    assert console.loop.time() - started == pytest.approx(0.02, abs=1e-5)
    assert not next_event.interrupted


async def test_successive_segments_each_wait_for_their_own_tail(console: _ConsoleHarness) -> None:
    for samples, delay in [(777, 0.125), (480, 0)]:
        await console.push(samples)
        console.output.flush()
        started = console.loop.time()
        console.render(delay=delay)
        event = await asyncio.wait_for(console.wait_for_playout(), timeout=1)
        assert console.loop.time() - started == pytest.approx(delay + samples / RATE, abs=1e-5)
        assert event.playback_position == pytest.approx(samples / RATE)
        assert not event.interrupted


async def test_cancelling_flush_cleans_up_device_waiter(console: _ConsoleHarness) -> None:
    await console.push(480)
    console.render(delay=10)
    console.output.flush()
    await asyncio.sleep(0.01)
    assert console.output._flush_task is not None
    await aio.cancel_and_wait(console.output._flush_task)
    assert console.output._flush_task.cancelled()
    # The suite's leaked-task fixture also checks the child wait/sleep tasks.


async def test_speaker_shutdown_drains_device_before_close(console: _ConsoleHarness) -> None:
    device = MagicMock()
    console.console._output_stream = device
    console.console._output_name = "test speaker"
    console.console.set_speaker_enabled(False)
    assert device.method_calls == [call.stop(), call.close()]
    assert console.console._output_stream is None
    assert console.console._output_name is None


async def test_speaker_shutdown_still_closes_after_stop_failure(console: _ConsoleHarness) -> None:
    device = MagicMock()
    device.stop.side_effect = RuntimeError("device unavailable")
    console.console._output_stream = device
    console.console._output_name = "test speaker"
    with pytest.raises(RuntimeError, match="device unavailable"):
        console.console.set_speaker_enabled(False)
    assert device.method_calls == [call.stop(), call.close()]
    assert console.console._output_stream is None


async def test_empty_flush_does_not_start_playout_waiter(console: _ConsoleHarness) -> None:
    console.output.flush()
    assert console.output._flush_task is None


async def test_prebuffered_tail_waits_for_device(
    console: _ConsoleHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_legacy, "PREBUFFER_DURATION", 0.3)
    await console.push(777)
    assert np.all(console.render(delay=10) == 0)
    assert console.output._playback_end_at == 0
    assert len(console.output._output_buf) == 777 * 2

    console.output.flush()
    started = console.loop.time()
    rendered = console.render()
    assert np.all(rendered[:777] == 1)
    assert np.all(rendered[777:] == 0)
    event = await asyncio.wait_for(console.wait_for_playout(), timeout=1)
    assert console.loop.time() - started == pytest.approx(0.125 + 777 / RATE, abs=1e-5)
    assert not event.interrupted
