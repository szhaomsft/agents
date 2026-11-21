"""Foundry GPT Live integration against a local, server-started WebSocket."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import time
from collections.abc import AsyncIterator, Callable
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qs, urlsplit

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from azure.core.credentials import AccessToken
from azure.core.credentials_async import AsyncTokenCredential
from azure.core.exceptions import ClientAuthenticationError

from livekit import rtc
from livekit.agents import Agent, AgentSession, APIConnectOptions, APIError, llm
from livekit.plugins.azure.realtime import GPTLiveModel, GPTLiveSession, gpt_live_model

from .fake_io import FakeAudioOutput

pytestmark = pytest.mark.unit

PCM = b"\x10\x20" * 2400


class _Credential(AsyncTokenCredential):
    def __init__(self) -> None:
        self.scopes: list[tuple[str, ...]] = []
        self.closed = False

    async def get_token(self, *scopes: str, **kwargs: Any) -> AccessToken:
        self.scopes.append(scopes)
        return AccessToken(f"token-{len(self.scopes)}", int(time.time()) + 3600)

    async def close(self) -> None:
        self.closed = True


class _Foundry:
    def __init__(self) -> None:
        self.endpoint = ""
        self.requests: list[web.Request] = []
        self.connections: list[web.WebSocketResponse] = []
        self.events: list[dict[str, Any]] = []
        self.auto_start = True
        self.rate = 24000
        self.status = 101
        self.delegation = "responses"

    async def start(self, ws: web.WebSocketResponse) -> None:
        await ws.send_json(
            {
                "type": "session.started",
                "conversation_id": f"conversation-{len(self.connections)}",
                "session": {
                    "id": f"live-{len(self.connections)}",
                    "audio": {"format": {"type": "audio/pcm", "rate": self.rate}},
                    "delegation": {"type": self.delegation},
                },
            }
        )

    async def handle(self, request: web.Request) -> web.StreamResponse:
        self.requests.append(request)
        if self.status != 101:
            return web.Response(status=self.status)
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.connections.append(ws)
        if self.auto_start:
            await self.start(ws)
        async for message in ws:
            if message.type == aiohttp.WSMsgType.TEXT:
                event = json.loads(message.data)
                self.events.append(event)
                if event["type"] == "session.close":
                    await ws.send_json(
                        {
                            "type": "session.closed",
                            "reason": "close_requested",
                            "usage": {"seconds": 1},
                        }
                    )
                    await ws.close()
        return ws


@pytest.fixture
async def foundry(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[_Foundry]:
    fake = _Foundry()
    app = web.Application()
    app.router.add_get("/{path:.*}", fake.handle)
    async with TestServer(app, host="127.0.0.1") as server:
        fake.endpoint = f"https://127.0.0.1:{server.port}/api/projects/test"
        original = GPTLiveSession._create_ws_url_and_headers

        def local_transport(self: GPTLiveSession) -> tuple[str, dict[str, str]]:
            url, headers = original(self)
            assert url.startswith(f"wss://127.0.0.1:{server.port}/")
            return url.replace("wss://", "ws://", 1), headers

        monkeypatch.setattr(GPTLiveSession, "_create_ws_url_and_headers", local_transport)
        try:
            yield fake
        finally:
            for ws in fake.connections:
                await ws.close()


async def _wait(predicate: Callable[[], object]) -> None:
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(0.01)


def _model(foundry: _Foundry, **kwargs: Any) -> GPTLiveModel:
    kwargs.setdefault("credential", _Credential())
    kwargs.setdefault("conn_options", APIConnectOptions(max_retry=0, timeout=2))
    return GPTLiveModel(project_endpoint=foundry.endpoint, agent_name="teacher", **kwargs)


@contextlib.asynccontextmanager
async def _session(
    foundry: _Foundry, **kwargs: Any
) -> AsyncIterator[tuple[GPTLiveModel, GPTLiveSession]]:
    model = _model(foundry, **kwargs)
    session = model.session()
    try:
        yield model, session
    finally:
        with contextlib.suppress(APIError):
            await session.aclose()
        await model.aclose()


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://resource/api/projects/demo",
        "https://user:password@resource/api/projects/demo",
        "https://resource/api/projects/demo?api-key=secret",
        "https://resource/api/projects/demo#fragment",
        "https://resource/api/projects/",
        "https://resource/",
    ],
)
def test_rejects_invalid_project_endpoints(endpoint: str) -> None:
    with pytest.raises(ValueError, match="HTTPS Foundry"):
        GPTLiveModel(project_endpoint=endpoint, agent_name="teacher", credential=_Credential())


def test_env_configuration_and_encoded_agent_options(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FOUNDRY_PROJECT_ENDPOINT", "https://resource/api/projects/demo/")
    monkeypatch.setenv("AGENT_NAME", "teacher /?#")
    values = {"learner_name": "Mei & Alex", "level": "intermediate"}
    model = GPTLiveModel(
        agent_version="7",
        structured_inputs=values,
        store=False,
        credential=_Credential(),
    )
    values["level"] = "advanced"
    url = urlsplit(model._connection_url)
    assert url.path.endswith("/agents/teacher%20%2F%3F%23/endpoint/protocols/voice/live/sessions")
    query = parse_qs(url.query)
    assert query["api-version"] == ["v1"]
    assert query["agent_version_override"] == ["7"]
    assert query["store"] == ["false"]
    assert json.loads(query["structured_inputs"][0]) == {
        "learner_name": "Mei & Alex",
        "level": "intermediate",
    }


async def test_server_starts_session_before_audio_and_preserves_saved_config(
    foundry: _Foundry,
) -> None:
    foundry.auto_start = False
    credential = _Credential()
    async with _session(foundry, credential=credential, agent_version="3") as (_, session):
        await session._update_session(instructions="Be concise.")
        session.push_audio(rtc.AudioFrame(PCM, 24000, 1, 2400))
        await _wait(lambda: foundry.connections)
        assert foundry.events == []
        await foundry.start(foundry.connections[0])
        await _wait(lambda: any(e["type"] == "session.input_audio.append" for e in foundry.events))

        assert [event["type"] for event in foundry.events] == [
            "session.instructions.append",
            "session.input_audio.append",
        ]
        assert base64.b64decode(foundry.events[-1]["audio"]) == PCM
        assert session.session_id == "live-1"
        assert session.conversation_id == "conversation-1"
        request = foundry.requests[0]
        assert (
            request.path
            == "/api/projects/test/agents/teacher/endpoint/protocols/voice/live/sessions"
        )
        assert request.query["agent_version_override"] == "3"
        assert request.headers["Authorization"] == "Bearer token-1"
        assert request.headers["Foundry-Features"] == "VoiceAgents=V1Preview"
        assert "api-key" not in request.headers
        assert credential.scopes == [("https://ai.azure.com/.default",)]
    assert not credential.closed
    assert foundry.events[-1]["type"] == "session.close"


async def test_ready_waits_for_server_start(foundry: _Foundry) -> None:
    foundry.auto_start = False
    async with _session(foundry) as (_, session):
        await session._update_session()
        ready = asyncio.create_task(session.wait_until_ready())
        await _wait(lambda: foundry.connections)
        assert not ready.done()
        await foundry.start(foundry.connections[0])
        await asyncio.wait_for(ready, 2)
        assert session.session_id == "live-1"
        await session.wait_until_ready()


async def test_cancelled_ready_wait_does_not_cancel_startup(foundry: _Foundry) -> None:
    foundry.auto_start = False
    async with _session(foundry) as (_, session):
        await session._update_session()
        ready = asyncio.create_task(session.wait_until_ready())
        await _wait(lambda: foundry.connections)
        ready.cancel()
        with pytest.raises(asyncio.CancelledError):
            await ready
        await foundry.start(foundry.connections[0])
        await asyncio.wait_for(session.wait_until_ready(), 2)


async def test_console_reports_ready_after_foundry_starts(
    foundry: _Foundry, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from examples.voice_agents import foundry_gpt_live as example

    foundry.auto_start = False
    model = _model(foundry)
    try:
        async with AgentSession(llm=model) as session:
            session.output.audio = FakeAudioOutput()

            class ConsoleSession:
                async def start(self, *, agent: Agent, room: Any) -> None:
                    await session.start(agent)

            monkeypatch.setattr(example.azure.realtime, "GPTLiveModel", lambda: model)
            monkeypatch.setattr(example, "AgentSession", lambda **kwargs: ConsoleSession())
            ctx = SimpleNamespace(room=None, add_shutdown_callback=lambda callback: None)
            entrypoint = asyncio.create_task(example.entrypoint(ctx))
            await _wait(lambda: foundry.connections)
            assert "Ready to speak." not in capsys.readouterr().out
            await foundry.start(foundry.connections[0])
            await asyncio.wait_for(entrypoint, 2)
            assert "Ready to speak." in capsys.readouterr().out
    finally:
        await model.aclose()


async def test_audio_and_transcripts_use_shared_duplex_engine(foundry: _Foundry) -> None:
    async with _session(foundry) as (_, session):
        transcripts: list[Any] = []
        session.on("transcript_delta", transcripts.append)
        await session._update_session()
        await _wait(lambda: session.session_id)
        ws = foundry.connections[0]
        await ws.send_json(
            {"type": "session.output_audio.delta", "delta": base64.b64encode(PCM).decode()}
        )
        await ws.send_json(
            {
                "type": "session.output_transcript.delta",
                "delta": "Hello Mei!",
                "start_ms": 0,
                "end_ms": 100,
            }
        )
        frame = await asyncio.wait_for(anext(aiter(session.audio_stream)), 2)
        assert frame.frame.data.tobytes() == PCM
        assert frame.frame.sample_rate == 24000
        await _wait(lambda: transcripts)
        assert transcripts[0].text == "Hello Mei!"


async def test_client_delegation_answers_without_responses_commands(foundry: _Foundry) -> None:
    foundry.delegation = "client"
    async with _session(foundry, delegation="client") as (_, session):
        delegations: list[Any] = []
        session.on("delegation_created", delegations.append)
        await session._update_session()
        await _wait(lambda: session.session_id)
        await foundry.connections[0].send_json(
            {"type": "session.delegation.created", "delegation": {"id": "d1", "target": "client"}}
        )
        await _wait(lambda: delegations)
        assert delegations[0].mode == "client"
        session.append_commentary("The answer is 42.", delegation_id=delegations[0].id)
        await _wait(lambda: foundry.events)
        assert foundry.events[0]["type"] == "session.commentary.append"
        assert foundry.events[0]["delegation_id"] == "d1"
        assert not any(event["type"].startswith("response.") for event in foundry.events)


async def test_rejects_tools_for_client_delegation_and_backend_configuration_changes(
    foundry: _Foundry,
) -> None:
    foundry.delegation = "client"
    async with _session(foundry, delegation="client") as (_, session):
        with pytest.raises(llm.RealtimeError, match="delegation_created handler"):
            await session._update_tools([lookup])
        with pytest.raises(llm.RealtimeError, match="saved agent"):
            session._update_options(tool_choice="none")
        assert session._opts.responses == {}
        assert not foundry.events


@llm.function_tool
async def lookup() -> str:
    """Look up the answer."""
    return "42"


@pytest.mark.parametrize("delegation", ["responses", "foundry_managed_agent"])
async def test_saved_backend_tools_return_results_and_continue(
    foundry: _Foundry, delegation: str
) -> None:
    foundry.delegation = delegation
    async with _session(foundry, delegation=delegation) as (_, session):
        calls: list[llm.FunctionCall] = []
        session.on("function_call", calls.append)
        await session._update_session(tools=[lookup])
        await _wait(lambda: session.session_id)
        for event in (
            {"type": "response.created"},
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "status": "completed",
                    "id": "item1",
                    "call_id": "call1",
                    "name": "lookup",
                    "arguments": "{}",
                },
            },
            {"type": "response.completed", "response": {"id": "response1"}},
        ):
            await foundry.connections[0].send_json(
                {"type": "response.event", "delegation_id": "d1", "event": event}
            )
        await _wait(lambda: calls and not session._backend_running_responses)
        await session._append_items(
            [
                llm.FunctionCallOutput(
                    call_id=calls[0].call_id, output="42", name="lookup", is_error=False
                )
            ]
        )
        await _wait(lambda: len(foundry.events) == 2)
        assert foundry.events[0]["type"] == "response.item.create"
        assert foundry.events[0]["item"]["output"] == "42"
        assert foundry.events[1]["type"] == "response.create"
        assert not any(event["type"] == "session.update" for event in foundry.events)


async def test_reconnect_refreshes_token_and_waits_for_new_server_start(foundry: _Foundry) -> None:
    credential = _Credential()
    async with _session(
        foundry, credential=credential, conn_options=APIConnectOptions(max_retry=1, timeout=2)
    ) as (_, session):
        await session._update_session()
        await _wait(lambda: session.session_id)
        await foundry.connections[0].close()
        await _wait(lambda: session.conversation_id == "conversation-2")
        assert credential.scopes == [("https://ai.azure.com/.default",)] * 2
        assert foundry.requests[1].headers["Authorization"] == "Bearer token-2"
        assert not any(event["type"] == "session.start" for event in foundry.events)


@pytest.mark.parametrize("failure", ["timeout", "audio_format", "delegation", "unauthorized"])
async def test_startup_failures_surface_without_sending_audio(
    foundry: _Foundry, failure: str
) -> None:
    if failure == "timeout":
        foundry.auto_start = False
    elif failure == "audio_format":
        foundry.rate = 48000
    elif failure == "delegation":
        foundry.delegation = "client"
    else:
        foundry.status = 401
    async with _session(foundry, conn_options=APIConnectOptions(max_retry=0, timeout=0.2)) as (
        _,
        session,
    ):
        errors: list[Any] = []
        session.on("error", errors.append)
        await session._update_session()
        session.push_audio(rtc.AudioFrame(PCM, 24000, 1, 2400))
        with pytest.raises(APIError):
            await asyncio.wait_for(session.wait_until_ready(), 3)
        assert errors and not errors[-1].recoverable
        assert not any(event["type"] == "session.input_audio.append" for event in foundry.events)


async def test_owned_default_credential_is_closed(
    foundry: _Foundry, monkeypatch: pytest.MonkeyPatch
) -> None:
    credential = _Credential()
    monkeypatch.setattr(gpt_live_model, "DefaultAzureCredential", lambda: credential)
    async with _session(foundry, credential=None) as (_, session):
        await session._update_session()
        await _wait(lambda: session.session_id)
    assert credential.closed


async def test_authentication_failure_does_not_expose_credentials(
    foundry: _Foundry, caplog: pytest.LogCaptureFixture
) -> None:
    class BrokenCredential(_Credential):
        async def get_token(self, *scopes: str, **kwargs: Any) -> AccessToken:
            raise ClientAuthenticationError("secret-token-value")

    async with _session(foundry, credential=BrokenCredential()) as (_, session):
        await session._update_session()
        with pytest.raises(APIError, match="authentication failed"):
            await session._main_atask
        assert not foundry.requests
    assert "secret-token-value" not in caplog.text


async def test_foundry_runs_inside_agent_session(foundry: _Foundry) -> None:
    model = _model(foundry)
    try:
        async with AgentSession(llm=model) as session:
            session.output.audio = FakeAudioOutput()
            await session.start(Agent(instructions="Be helpful."))
            await _wait(lambda: foundry.events)
            assert foundry.events[0]["type"] == "session.instructions.append"
            assert foundry.events[0]["content"] == "Be helpful."
            assert not any(event["type"] == "session.start" for event in foundry.events)
    finally:
        await model.aclose()


@pytest.mark.parametrize("delegation", ["responses", "foundry_managed_agent"])
async def test_agent_session_executes_saved_backend_function(
    foundry: _Foundry, delegation: str
) -> None:
    foundry.delegation = delegation
    model = _model(foundry, delegation=delegation)
    try:
        async with AgentSession(llm=model) as session:
            session.output.audio = FakeAudioOutput()
            await session.start(Agent(instructions="", tools=[lookup]))
            await _wait(lambda: foundry.connections)
            ws = foundry.connections[0]
            await ws.send_json(
                {
                    "type": "session.delegation.created",
                    "offset_ms": 100,
                    "delegation": {"id": "d1", "target": delegation},
                }
            )
            for event in (
                {"type": "response.created"},
                {
                    "type": "response.output_item.done",
                    "item": {
                        "type": "function_call",
                        "status": "completed",
                        "id": "item1",
                        "call_id": "call1",
                        "name": "lookup",
                        "arguments": "{}",
                    },
                },
                {"type": "response.completed", "response": {"id": "response1"}},
            ):
                await ws.send_json(
                    {"type": "response.event", "delegation_id": "d1", "event": event}
                )
            await _wait(lambda: any(e["type"] == "response.create" for e in foundry.events))
            results = [e["item"] for e in foundry.events if e["type"] == "response.item.create"]
            assert results == [{"type": "function_call_output", "call_id": "call1", "output": "42"}]
            assert not any(e["type"] == "session.update" for e in foundry.events)
    finally:
        await model.aclose()


async def test_shared_http_session_and_close_before_readiness(foundry: _Foundry) -> None:
    foundry.auto_start = False
    async with aiohttp.ClientSession() as http_session:
        async with _session(foundry, http_session=http_session) as (_, session):
            await session._update_session()
            await _wait(lambda: foundry.connections)
        assert not http_session.closed
        assert not any(e["type"] == "session.input_audio.append" for e in foundry.events)
