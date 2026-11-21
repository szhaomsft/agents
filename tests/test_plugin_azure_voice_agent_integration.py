"""Opt-in, billable speech test against an existing Foundry realtime voice agent."""

from __future__ import annotations

import asyncio
import json
import os
import wave
from pathlib import Path
from typing import Any

import pytest

from livekit import rtc
from livekit.agents import APIConnectOptions, llm
from livekit.plugins.azure.realtime import VoiceAgentModel

pytestmark = pytest.mark.plugin("azure")


@pytest.mark.skipif(
    os.environ.get("FOUNDRY_VOICE_AGENT_SMOKE") != "1",
    reason="Set FOUNDRY_VOICE_AGENT_SMOKE=1 to enable a billable deployment test",
)
async def test_foundry_voice_agent_speech_turn() -> None:
    for name in ("FOUNDRY_PROJECT_ENDPOINT", "AGENT_NAME"):
        assert os.environ.get(name), f"{name} is required"
    inputs = json.loads(os.environ.get("FOUNDRY_STRUCTURED_INPUTS", "null"))
    assert inputs is None or isinstance(inputs, dict)
    with wave.open(
        str(Path(__file__).parent / "test_realtime" / "weather_question.wav"), "rb"
    ) as f:
        assert (f.getframerate(), f.getnchannels(), f.getsampwidth()) == (24000, 1, 2)
        audio = f.readframes(f.getnframes())

    model = VoiceAgentModel(
        structured_inputs=inputs, conn_options=APIConnectOptions(max_retry=0, timeout=30)
    )
    session = model.session()
    errors: list[Any] = []
    transcripts: list[str] = []
    output: list[tuple[str, int]] = []
    tasks: list[asyncio.Task[None]] = []
    speech_stopped = False
    done = asyncio.Event()

    def on_stopped(_: llm.InputSpeechStoppedEvent) -> None:
        nonlocal speech_stopped
        speech_stopped = True

    async def read_generation(event: llm.GenerationCreatedEvent) -> None:
        async for message in event.message_stream:

            async def read_text(message: llm.MessageGeneration) -> str:
                return "".join([part async for part in message.text_stream])

            async def read_audio(message: llm.MessageGeneration) -> int:
                return sum(
                    frame.data.nbytes for frame in [frame async for frame in message.audio_stream]
                )

            text, size = await asyncio.gather(read_text(message), read_audio(message))
            output.append((text, size))
            done.set()

    def on_generation(event: llm.GenerationCreatedEvent) -> None:
        if speech_stopped:
            tasks.append(asyncio.create_task(read_generation(event)))

    session.on("input_speech_stopped", on_stopped)
    session.on("input_audio_transcription_completed", lambda ev: transcripts.append(ev.transcript))
    session.on("generation_created", on_generation)
    session.on("error", errors.append)

    async def exercise() -> None:
        await session.wait_until_ready()
        for offset in range(0, len(audio), 4800):
            chunk = audio[offset : offset + 4800]
            chunk += bytes(4800 - len(chunk))
            session.push_audio(rtc.AudioFrame(chunk, 24000, 1, 2400))
            await asyncio.sleep(0.1)
        for _ in range(18):
            session.push_audio(rtc.AudioFrame(bytes(4800), 24000, 1, 2400))
            await asyncio.sleep(0.1)
        await done.wait()
        while not transcripts:
            await asyncio.sleep(0.05)

    try:
        await asyncio.wait_for(exercise(), 55)
        assert session.conversation_id and session.session_id
        assert any(transcripts)
        assert any(text and size > 0 for text, size in output)
        assert not errors
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await model.aclose()
