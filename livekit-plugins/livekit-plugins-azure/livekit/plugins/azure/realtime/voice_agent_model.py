"""Saved Foundry realtime and cascaded voice agents, using the Voice Agent SDK."""

from __future__ import annotations

import asyncio
import contextlib
import os
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlsplit

import aiohttp

from azure.ai.voicelive.aio import VoiceLiveConnection
from azure.ai.voicelive.models import (
    AudioInputTranscriptionOptions,
    ClientEvent,
    ClientEventResponseCreate,
    ServerEvent,
)
from azure.core.credentials_async import AsyncTokenCredential
from azure.core.exceptions import ClientAuthenticationError, HttpResponseError
from azure.identity.aio import DefaultAzureCredential
from livekit.agents import APIConnectionError, APIConnectOptions, APIStatusError, llm, utils
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .realtime_model import (
    RealtimeModel,
    RealtimeSession,
    _ConnectionRequests,
)

if TYPE_CHECKING:
    from azure.ai.projects.aio.operations import AsyncRealtimeConnection


class _SDKConnection:
    def __init__(self, connection: AsyncRealtimeConnection) -> None:
        self._connection = connection
        self._instructions: str | None = None

    @property
    def has_instructions(self) -> bool:
        return self._instructions is not None

    async def send(self, event: ClientEvent) -> None:
        payload = event.as_dict()
        if payload.get("type") == "response.create":
            extra = payload.pop("additional_instructions", None)
            if extra:
                if self._instructions is None:
                    raise llm.RealtimeError(
                        "Foundry did not return saved instructions; cannot append response instructions"
                    )
                payload["response"]["instructions"] = "\n\n".join(
                    part for part in (self._instructions, extra) if part
                )
        await self._connection.send(payload)

    async def __aiter__(self) -> AsyncIterator[ServerEvent]:
        async for event in self._connection:
            payload = event.as_dict() if hasattr(event, "as_dict") else dict(event)
            if payload.get("type") == "session.updated":
                instructions = payload.get("session", {}).get("instructions")
                self._instructions = instructions if isinstance(instructions, str) else None
            # Foundry uses the GA output event names; the shared Voice Live handlers
            # use their preview equivalents. Transport and authentication remain in the SDK.
            kind = payload.get("type", "")
            for output in ("audio_transcript", "audio", "text"):
                kind = kind.replace(f"response.output_{output}.", f"response.{output}.")
            payload["type"] = kind
            yield cast(ServerEvent, ServerEvent._deserialize(payload, []))  # type: ignore[no-untyped-call]


class VoiceAgentModel(RealtimeModel):
    """Connect to a saved Foundry GPT Realtime or STT/LLM/TTS cascade agent.

    Install ``livekit-plugins-azure[agents]``. The saved agent must use the
    ``realtime`` sub-protocol, server turn detection and 24 kHz PCM16 audio.
    Model, voice, instructions and tool schemas come from the saved definition.
    Use an empty LiveKit Agent instruction string to preserve those instructions.

    Args:
        project_endpoint: HTTPS project URL, or FOUNDRY_PROJECT_ENDPOINT.
        agent_name: Saved voice agent name, or AGENT_NAME.
        agent_version: Optional saved version override.
        structured_inputs: Values for inputs declared by the saved agent.
        user_transcription: Whether the saved agent enables input transcription.
            Set to False if the agent has no input transcription configured.
        credential: Caller-owned asynchronous Entra credential. If omitted,
            the model owns a DefaultAzureCredential.
        conn_options: Connection timeout and retry policy.
    """

    def __init__(
        self,
        *,
        project_endpoint: str | None = None,
        agent_name: str | None = None,
        agent_version: str | None = None,
        structured_inputs: dict[str, Any] | None = None,
        user_transcription: bool = True,
        credential: AsyncTokenCredential | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        endpoint = project_endpoint or os.environ.get("FOUNDRY_PROJECT_ENDPOINT")
        name = agent_name or os.environ.get("AGENT_NAME")
        if not endpoint:
            raise ValueError("project_endpoint or FOUNDRY_PROJECT_ENDPOINT is required")
        parsed = urlsplit(endpoint)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or not parsed.path.partition("/api/projects/")[2].strip("/")
        ):
            raise ValueError(
                "project_endpoint must be an HTTPS Foundry /api/projects/<project> URL "
                "without credentials, query parameters or a fragment"
            )
        if not name or not name.strip():
            raise ValueError("agent_name or AGENT_NAME is required")
        if agent_version is not None and not agent_version.strip():
            raise ValueError("agent_version must not be empty")
        try:
            from azure.ai.projects.aio import AIProjectClient
        except ImportError:
            raise ImportError(
                "Foundry Voice Agent SDK support requires "
                "`pip install 'livekit-plugins-azure[agents]'`"
            ) from None

        super().__init__(
            endpoint=endpoint,
            model=name,
            use_default_credential=True,
            conn_options=conn_options,
        )
        self.capabilities.mutable_instructions = False
        self.capabilities.mutable_tools = False
        self.capabilities.user_transcription = user_transcription
        self._agent_name = name
        self._agent_version = agent_version
        self._structured_inputs = dict(structured_inputs) if structured_inputs is not None else None
        self._foundry_credential = credential or DefaultAzureCredential()
        self._credential_owned = credential is None
        self._project = AIProjectClient(
            endpoint=endpoint.rstrip("/"),
            credential=self._foundry_credential,
            allow_preview=True,
        )
        self._closed = False

    @property
    def provider(self) -> str:
        return "azure-foundry-voice-agent"

    def session(self, *, turn_detection_disabled: bool = False) -> VoiceAgentSession:
        if self._closed:
            raise RuntimeError("Foundry Voice Agent model is closed")
        if turn_detection_disabled:
            raise ValueError("Configure turn detection in the saved Foundry voice agent")
        session = VoiceAgentSession(self)
        self._sessions.add(session)
        return session

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            for session in list(self._sessions):
                await session.aclose()
        finally:
            try:
                await self._project.close()
            finally:
                if self._credential_owned:
                    await self._foundry_credential.close()


class VoiceAgentSession(RealtimeSession):
    """A response-oriented session configured by a saved Foundry voice agent."""

    def __init__(self, realtime_model: VoiceAgentModel) -> None:
        self._foundry_model = realtime_model
        self._conversation_id: str | None = None
        self._session_id: str | None = None
        self._first_ready: asyncio.Future[None] = asyncio.Future()
        self._configured = asyncio.Event()
        self._configuration_error: Exception | None = None
        self._startup_error: Exception | None = None
        super().__init__(realtime_model)

    @property
    def conversation_id(self) -> str | None:
        """Current Foundry conversation ID; reconnects start a new conversation."""
        return self._conversation_id

    @property
    def session_id(self) -> str | None:
        return self._session_id

    async def wait_until_ready(self) -> None:
        """Wait for validated saved configuration, or raise if startup fails."""
        await asyncio.wait(
            (self._first_ready, self._main_atask), return_when=asyncio.FIRST_COMPLETED
        )
        if self._main_atask.done():
            if not self._main_atask.cancelled():
                await self._main_atask
            if self._startup_error is not None:
                raise self._startup_error
            raise APIConnectionError("Foundry voice agent session closed", retryable=False)
        await self._first_ready

    @contextlib.asynccontextmanager
    async def _connect(self) -> AsyncIterator[_SDKConnection]:
        self._configured.clear()
        self._configuration_error = None
        self._session_id = None
        self._conversation_id = None
        model = self._foundry_model
        query = (
            {"agent_version_override": model._agent_version}
            if model._agent_version is not None
            else None
        )
        try:
            async with model._project.voice_agents.realtime.connect(
                agent_name=model._agent_name,
                transport="websocket",
                structured_inputs=model._structured_inputs,
                extra_query=query,
            ) as connection:
                yield _SDKConnection(connection)
        except ClientAuthenticationError:
            raise APIConnectionError(
                "Foundry Entra authentication failed", retryable=False
            ) from None
        except (HttpResponseError, aiohttp.ClientResponseError) as error:
            status = (
                error.status
                if isinstance(error, aiohttp.ClientResponseError)
                else error.status_code
            )
            raise APIStatusError(
                "Foundry Voice Agent SDK connection failed",
                status_code=status or -1,
                retryable=status == 429 or (status is not None and status >= 500),
            ) from None

    async def _run_connection(self, *, reconnecting: bool) -> None:
        async with contextlib.AsyncExitStack() as stack:
            try:
                sdk_conn = await asyncio.wait_for(
                    stack.enter_async_context(self._connect()), self._opts.conn_options.timeout
                )
            except asyncio.TimeoutError:
                raise APIConnectionError("Foundry voice agent connection timed out") from None
            except (OSError, aiohttp.ClientError) as error:
                raise APIConnectionError("Foundry voice agent connection failed") from error

            # The inherited engine consumes only send() and async iteration, both
            # implemented by this bridge. Keep the nominal-type cast at this boundary
            # so the existing Voice Live implementation can remain unchanged.
            conn = cast(VoiceLiveConnection, sdk_conn)
            requests = _ConnectionRequests()
            self._requests = requests
            tasks = [asyncio.create_task(self._receive_sdk_events(sdk_conn))]
            try:
                await self._wait_for_configuration()
                if reconnecting:
                    await self._replay_conversation(conn, requests)
                    self.emit("session_reconnected", llm.RealtimeSessionReconnectedEvent())
                requests.replayed = True
                self._check_established()
                tasks.append(asyncio.create_task(self._send_task(conn, requests)))
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
            finally:
                await utils.aio.cancel_and_wait(*tasks)
                self._on_connection_closed(requests)

    async def _wait_for_configuration(self) -> None:
        try:
            await asyncio.wait_for(self._configured.wait(), self._opts.conn_options.timeout)
        except asyncio.TimeoutError:
            raise APIConnectionError("Foundry voice agent startup timed out") from None
        if self._configuration_error is not None:
            raise self._configuration_error

    async def _send_event(
        self, conn: VoiceLiveConnection, requests: _ConnectionRequests, event: ClientEvent
    ) -> None:
        sdk_conn = cast(_SDKConnection, conn)
        if (
            isinstance(event, ClientEventResponseCreate)
            and event.additional_instructions
            and not sdk_conn.has_instructions
        ):
            future = self._response_created_futures.pop(event.event_id or "", None)
            if future is not None and not future.done():
                future.set_exception(
                    llm.RealtimeError(
                        "Foundry did not return saved instructions; cannot append response instructions"
                    )
                )
            return
        await super()._send_event(conn, requests, event)

    async def _receive_sdk_events(self, conn: _SDKConnection) -> None:
        try:
            async for event in conn:
                self._handle_server_event(event)
        except (OSError, aiohttp.ClientError) as error:
            self._configuration_error = APIConnectionError("Foundry voice agent connection lost")
            raise self._configuration_error from error
        except Exception as error:
            self._configuration_error = error
            raise
        finally:
            if not self._configured.is_set():
                if self._configuration_error is None:
                    self._configuration_error = APIConnectionError(
                        "Foundry voice agent closed before applying saved configuration"
                    )
                self._configured.set()

    def _handle_server_event(self, event: ServerEvent) -> None:
        payload = event.as_dict()
        if event.type == "session.created":
            self._conversation_id = payload.get("conversation_id")
        elif event.type == "session.updated":
            session = payload.get("session", {})
            audio = session.get("audio", {})
            for direction in ("input", "output"):
                if audio.get(direction, {}).get("format") != {
                    "type": "audio/pcm",
                    "rate": 24000,
                }:
                    raise APIConnectionError(
                        "Foundry voice agents require PCM16 audio at 24000 Hz",
                        retryable=False,
                    )
            if "audio" not in session.get("output_modalities", []):
                raise APIConnectionError(
                    "Enable audio output in the saved Foundry voice agent", retryable=False
                )
            if not audio["input"].get("turn_detection"):
                raise APIConnectionError(
                    "Enable server turn detection in the saved Foundry voice agent",
                    retryable=False,
                )
            if audio["input"]["turn_detection"].get("create_response") is False:
                raise APIConnectionError(
                    "Enable automatic responses in the saved Foundry voice agent",
                    retryable=False,
                )
            transcription = audio["input"].get("transcription")
            if bool(transcription) != self._foundry_model.capabilities.user_transcription:
                raise APIConnectionError(
                    "user_transcription must match the saved Foundry agent configuration",
                    retryable=False,
                )
            self._opts.input_audio_transcription = (
                AudioInputTranscriptionOptions(transcription) if transcription else None
            )
            self._session_id = session.get("id")
            self._configured.set()
        elif event.type == "error" and not self._configured.is_set():
            raise APIConnectionError("Foundry rejected voice agent startup", retryable=False)
        super()._handle_server_event(event)

    def _check_established(self) -> None:
        super()._check_established()
        if self._connection_established and not self._first_ready.done():
            self._first_ready.set_result(None)

    def _emit_error(self, error: Exception, recoverable: bool) -> None:
        if not recoverable:
            self._startup_error = error
        super()._emit_error(error, recoverable)

    def update_options(self, *, tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN) -> None:
        if is_given(tool_choice):
            raise llm.RealtimeError("Configure tool_choice in Foundry or pass it to generate_reply")

    async def update_instructions(self, instructions: str) -> None:
        if instructions:
            raise llm.RealtimeError(
                "Configure instructions in the saved Foundry agent; use Agent(instructions='')"
            )

    async def update_tools(self, tools: list[llm.Tool]) -> None:
        """Bind local implementations of tool schemas already declared in Foundry."""
        async with self._update_fnc_ctx_lock:
            self._tools = llm.ToolContext(tools)
