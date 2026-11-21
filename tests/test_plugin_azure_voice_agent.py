"""Foundry Voice Agent SDK integration without a cloud connection."""

from __future__ import annotations

import asyncio
import base64
import contextlib
from collections.abc import AsyncIterator
from typing import Any

import pytest
from aiohttp import ClientResponseError
from azure.core.credentials import AccessToken

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    function_tool,
    llm,
)
from livekit.plugins.azure.realtime import VoiceAgentModel

from .fake_io import FakeAudioOutput
from .test_plugin_azure_realtime import _wait_until

pytestmark = pytest.mark.unit


class _Credential:
    def __init__(self) -> None:
        self.closed = False

    async def get_token(self, *scopes: str, **kwargs: Any) -> AccessToken:
        return AccessToken("test-token", 9999999999)

    async def close(self) -> None:
        self.closed = True


class _Connection:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.incoming: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self.count = 0
        self.tool_reply = False

    def emit(self, event_type: str, **kwargs: Any) -> None:
        self.incoming.put_nowait({"type": event_type, "event_id": "server_event", **kwargs})

    def configured(
        self,
        *,
        rate: int = 24000,
        vad: bool = True,
        instructions: str | None = "Saved instructions",
        transcription: bool = True,
    ) -> None:
        self.emit("session.created", session={"id": "session"}, conversation_id="conversation")
        self.emit(
            "session.updated",
            session={
                "id": "session",
                "output_modalities": ["audio"],
                "instructions": instructions,
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": rate},
                        "transcription": {"model": "azure-speech"} if transcription else None,
                        "turn_detection": {"type": "server_vad"} if vad else None,
                    },
                    "output": {"format": {"type": "audio/pcm", "rate": 24000}},
                },
            },
        )

    def __aiter__(self) -> _Connection:
        return self

    async def __anext__(self) -> dict[str, Any]:
        event = await self.incoming.get()
        if event is None:
            raise StopAsyncIteration
        return event

    async def send(self, event: dict[str, Any]) -> None:
        self.events.append(event)
        if event["type"] == "conversation.item.create":
            self.emit(
                "conversation.item.created",
                previous_item_id=event.get("previous_item_id"),
                item=event["item"],
            )
        elif event["type"] == "response.create":
            self.count += 1
            response_id = f"response_{self.count}"
            self.emit(
                "response.created",
                response={
                    "id": response_id,
                    "status": "in_progress",
                    "metadata": event["response"]["metadata"],
                    "output": [],
                },
            )
            if self.tool_reply and self.count == 1:
                item = {
                    "id": "item_call",
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": "",
                    "status": "in_progress",
                }
                self.emit("response.output_item.added", response_id=response_id, item=item)
                self.emit("conversation.item.created", previous_item_id=None, item=item)
                self.emit(
                    "response.function_call_arguments.done",
                    response_id=response_id,
                    item_id="item_call",
                    call_id="call_1",
                    name="get_weather",
                    arguments='{"location":"Tokyo"}',
                )
            else:
                item = {
                    "id": f"message_{self.count}",
                    "type": "message",
                    "role": "assistant",
                    "status": "in_progress",
                    "content": [],
                }
                self.emit("response.output_item.added", response_id=response_id, item=item)
                self.emit("conversation.item.created", previous_item_id=None, item=item)
                self.emit(
                    "response.content_part.added",
                    response_id=response_id,
                    item_id=item["id"],
                    part={"type": "audio", "transcript": ""},
                )
                self.emit(
                    "response.output_audio_transcript.delta",
                    response_id=response_id,
                    item_id=item["id"],
                    delta="Hello!",
                )
                self.emit(
                    "response.output_audio.delta",
                    response_id=response_id,
                    item_id=item["id"],
                    delta=base64.b64encode(b"\x01\x00" * 480).decode(),
                )
            self.emit(
                "response.done",
                response={"id": response_id, "status": "completed", "output": []},
            )


class _Project:
    def __init__(self) -> None:
        self.voice_agents = self
        self.realtime = self
        self.connections: list[_Connection] = []
        self.arguments: list[dict[str, Any]] = []
        self.next_config: tuple[int, bool] | None = (24000, True)
        self.failure: Exception | None = None
        self.closed = False

    @contextlib.asynccontextmanager
    async def connect(self, **kwargs: Any) -> AsyncIterator[_Connection]:
        self.arguments.append(kwargs)
        if self.failure:
            raise self.failure
        conn = _Connection()
        self.connections.append(conn)
        if self.next_config:
            rate, vad = self.next_config
            conn.configured(rate=rate, vad=vad)
        yield conn

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def project(monkeypatch: pytest.MonkeyPatch) -> _Project:
    sdk = pytest.importorskip("azure.ai.projects.aio")
    project = _Project()
    monkeypatch.setattr(sdk, "AIProjectClient", lambda **kwargs: project)
    return project


def _model(credential: _Credential, **kwargs: Any) -> VoiceAgentModel:
    return VoiceAgentModel(
        project_endpoint="https://example.test/api/projects/project",
        agent_name="saved-agent",
        credential=credential,
        conn_options=APIConnectOptions(max_retry=0, timeout=1),
        **kwargs,
    )


async def test_sdk_startup_and_audio_events_preserve_saved_settings(project: _Project) -> None:
    project.next_config = None
    credential = _Credential()
    model = _model(credential, agent_version="2", structured_inputs={"name": "Sam"})
    session = model.session()
    try:
        ready = asyncio.create_task(session.wait_until_ready())
        await _wait_until(lambda: bool(project.connections))
        conn = project.connections[0]
        frame = rtc.AudioFrame(b"\x01\x00" * 2400, 24000, 1, 2400)
        session.push_audio(frame)
        assert not ready.done()
        assert not conn.events
        conn.configured()
        await ready
        await _wait_until(lambda: bool(conn.events))
        assert session.conversation_id == "conversation"
        assert session.session_id == "session"
        assert project.arguments[0]["extra_query"] == {"agent_version_override": "2"}
        assert project.arguments[0]["structured_inputs"] == {"name": "Sam"}
        assert all(e["type"] != "session.update" for e in conn.events)
        reply = await session.generate_reply(instructions="Say hello")
        messages = [m async for m in reply.message_stream]
        assert len(messages) == 1
        assert "".join([t async for t in messages[0].text_stream]) == "Hello!"
        assert sum(f.data.nbytes for f in [f async for f in messages[0].audio_stream]) == 960
        request = next(e for e in conn.events if e["type"] == "response.create")
        assert request["response"]["instructions"] == "Saved instructions\n\nSay hello"
        assert "additional_instructions" not in request
    finally:
        await model.aclose()
    assert project.closed
    assert not credential.closed


async def test_response_instructions_require_saved_instructions(project: _Project) -> None:
    project.next_config = None
    model = _model(_Credential())
    session = model.session()
    try:
        await _wait_until(lambda: bool(project.connections))
        conn = project.connections[0]
        conn.configured(instructions=None)
        await session.wait_until_ready()
        with pytest.raises(llm.RealtimeError, match="did not return saved instructions"):
            await session.generate_reply(instructions="Say hello")
        assert not any(e["type"] == "response.create" for e in conn.events)
        reply = await session.generate_reply()
        assert [message async for message in reply.message_stream]
    finally:
        await model.aclose()


async def test_cancelled_waiter_does_not_cancel_startup(project: _Project) -> None:
    project.next_config = None
    model = _model(_Credential())
    session = model.session()
    try:
        waiter = asyncio.create_task(session.wait_until_ready())
        await _wait_until(lambda: bool(project.connections))
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        project.connections[0].configured()
        await session.wait_until_ready()
    finally:
        await model.aclose()


async def test_closing_before_readiness_fails_waiter(project: _Project) -> None:
    project.next_config = None
    model = _model(_Credential())
    session = model.session()
    waiter = asyncio.create_task(session.wait_until_ready())
    await _wait_until(lambda: bool(project.connections))
    await model.aclose()
    with pytest.raises(APIConnectionError, match="session closed"):
        await waiter


async def test_transcription_setting_matches_saved_agent(project: _Project) -> None:
    project.next_config = None
    model = _model(_Credential(), user_transcription=False)
    session = model.session()
    try:
        await _wait_until(lambda: bool(project.connections))
        project.connections[0].configured(transcription=False)
        await session.wait_until_ready()
        assert not model.capabilities.user_transcription
    finally:
        await model.aclose()


async def test_handshake_failure_is_not_retried(project: _Project) -> None:
    project.failure = ClientResponseError(None, (), status=400)
    model = _model(_Credential())
    session = model.session()
    try:
        with pytest.raises(APIStatusError) as error:
            await session.wait_until_ready()
        assert error.value.status_code == 400
        assert not error.value.retryable
        assert len(project.arguments) == 1
    finally:
        await model.aclose()


async def test_reconnection_replays_history_without_replacing_configuration(
    project: _Project,
) -> None:
    model = _model(_Credential())
    session = model.session()
    try:
        await session.wait_until_ready()
        context = session.chat_ctx.copy()
        context.add_message(role="user", content="Hello")
        await session.update_chat_ctx(context)
        project.connections[0].incoming.put_nowait(None)
        await _wait_until(lambda: len(project.connections) == 2)
        await _wait_until(
            lambda: any(
                e["type"] == "conversation.item.create" for e in project.connections[1].events
            )
        )
        assert not any(e["type"] == "session.update" for e in project.connections[1].events)
        reply = await session.generate_reply()
        assert [message async for message in reply.message_stream]
    finally:
        await model.aclose()


@pytest.mark.parametrize("rate,vad", [(16000, True), (24000, False)])
async def test_saved_configuration_is_validated(project: _Project, rate: int, vad: bool) -> None:
    project.next_config = (rate, vad)
    model = _model(_Credential())
    session = model.session()
    try:
        with pytest.raises(APIConnectionError, match="PCM16|turn detection"):
            await session.wait_until_ready()
    finally:
        await model.aclose()


async def test_agent_session_runs_locally_bound_tool(project: _Project) -> None:
    calls: list[str] = []

    class WeatherAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="")

        @function_tool
        async def get_weather(self, location: str) -> str:
            """Get the weather for a location."""
            calls.append(location)
            return "sunny"

    model = _model(_Credential())
    async with AgentSession(llm=model) as session:
        session.output.audio = FakeAudioOutput()
        await session.start(WeatherAgent())
        await _wait_until(lambda: bool(project.connections))
        project.connections[0].tool_reply = True
        session.generate_reply(user_input="What is the weather?")
        await _wait_until(lambda: calls == ["Tokyo"])
        await _wait_until(
            lambda: any(
                e.get("item", {}).get("type") == "function_call_output"
                for e in project.connections[0].events
            )
        )
        assert not any(e["type"] == "session.update" for e in project.connections[0].events)
    await model.aclose()
