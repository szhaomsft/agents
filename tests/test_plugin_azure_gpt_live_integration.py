"""Opt-in smoke test against a real, preconfigured Foundry GPT Live agent."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, cast

import pytest

from livekit import rtc
from livekit.agents import APIConnectOptions, utils
from livekit.plugins.azure.realtime import GPTLiveModel
from livekit.plugins.azure.realtime.gpt_live_model import FoundryDelegation

pytestmark = pytest.mark.plugin("azure")


@pytest.mark.skipif(
    os.environ.get("FOUNDRY_GPT_LIVE_SMOKE") != "1",
    reason="Set FOUNDRY_GPT_LIVE_SMOKE=1 to enable a billable Foundry deployment test",
)
async def test_foundry_agent_speaks_and_transcribes() -> None:
    for name in ("FOUNDRY_PROJECT_ENDPOINT", "AGENT_NAME"):
        assert os.environ.get(name), f"{name} is required for the live smoke test"
    inputs = json.loads(os.environ.get("FOUNDRY_GPT_LIVE_STRUCTURED_INPUTS", "null"))
    assert inputs is None or isinstance(inputs, dict), "Structured inputs must be a JSON object"
    model = GPTLiveModel(
        structured_inputs=inputs,
        delegation=cast(
            FoundryDelegation, os.environ.get("FOUNDRY_GPT_LIVE_DELEGATION", "responses")
        ),
        conn_options=APIConnectOptions(max_retry=0, timeout=30),
    )
    session = model.session()
    transcripts: list[Any] = []
    errors: list[Any] = []
    session.on("transcript_delta", transcripts.append)
    session.on("error", errors.append)

    async def send_silence() -> None:
        while True:
            session.push_audio(
                rtc.AudioFrame(
                    b"\x00\x00" * 2400, sample_rate=24000, num_channels=1, samples_per_channel=2400
                )
            )
            await asyncio.sleep(0.1)

    async def receive_speech() -> bool:
        async for output in session.audio_stream:
            if any(output.frame.data):
                return True
        return False

    tasks: list[asyncio.Task[Any]] = []
    try:
        await session._update_session()
        audio = asyncio.create_task(receive_speech())
        tasks.extend([audio, asyncio.create_task(send_silence())])
        async with asyncio.timeout(45):
            await session.wait_until_ready()
            assert session.session_id
            session.append_commentary("Say hello to confirm that the voice connection works.")
            assert await audio, "Foundry closed without producing speech"
            while not any(event.text.strip() for event in transcripts):
                assert not errors, "Foundry session failed before producing a transcript"
                await asyncio.sleep(0.05)
        assert session.conversation_id
        assert not errors
    finally:
        await utils.aio.cancel_and_wait(*tasks)
        try:
            await session.aclose()
        finally:
            await model.aclose()
