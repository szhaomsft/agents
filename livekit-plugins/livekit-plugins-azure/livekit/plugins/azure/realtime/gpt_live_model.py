"""Foundry-managed GPT Live agents using the shared full-duplex protocol engine."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Literal
from urllib.parse import quote, urlencode, urlsplit, urlunsplit

import aiohttp

from azure.core.credentials_async import AsyncTokenCredential
from azure.core.exceptions import ClientAuthenticationError
from azure.identity.aio import DefaultAzureCredential
from livekit.agents import APIConnectionError, APIStatusError, llm
from livekit.agents.llm._realtime import gpt_live as _impl, gpt_live_types as types
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given

_TOKEN_SCOPE = "https://ai.azure.com/.default"
FoundryDelegation = Literal["responses", "client", "foundry_managed_agent"]


class GPTLiveModel(_impl.GPTLiveModel):
    """Connect an existing Foundry GPT Live voice agent to ``AgentSession(llm=...)``.

    The agent must have ``kind="voice"``, ``sub_protocol="live"`` and PCM16/24 kHz
    audio configured in Foundry. Foundry supplies its model (such as the managed
    ``gpt-live-1-datazone``), voice, instructions, greeting and delegation backend;
    this client never sends ``session.start``.

    Args:
        project_endpoint: HTTPS project endpoint ending in ``/api/projects/<project>``.
            Defaults to ``FOUNDRY_PROJECT_ENDPOINT``.
        agent_name: Saved voice agent name. Defaults to ``AGENT_NAME``.
        agent_version: Pin a saved version; otherwise Foundry selects the active version.
        structured_inputs: JSON values for the saved agent's template placeholders.
        delegation: Must match the saved agent's delegation type. Local function tools
            must already be declared on its Responses or managed-agent backend.
        credential: Async Entra credential. Defaults to an owned ``DefaultAzureCredential``.
            Caller-provided credentials remain owned by the caller.
        api_version: Foundry API version, ``v1`` by default.
        store: Override Foundry conversation storage; None preserves the agent's setting.
        http_session: Optional caller-owned HTTP session.
        conn_options: Connection, startup timeout and retry settings.
    """

    def __init__(
        self,
        *,
        project_endpoint: str | None = None,
        agent_name: str | None = None,
        agent_version: str | None = None,
        structured_inputs: dict[str, Any] | None = None,
        delegation: FoundryDelegation = "responses",
        credential: AsyncTokenCredential | None = None,
        api_version: str = "v1",
        store: bool | None = None,
        http_session: aiohttp.ClientSession | None = None,
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
        if not api_version.strip():
            raise ValueError("api_version must not be empty")
        if delegation not in ("responses", "client", "foundry_managed_agent"):
            raise ValueError("Unsupported Foundry delegation type")

        query = {"api-version": api_version}
        if agent_version is not None:
            query["agent_version_override"] = agent_version
        if structured_inputs is not None:
            query["structured_inputs"] = json.dumps(structured_inputs, allow_nan=False)
        if store is not None:
            query["store"] = str(store).lower()
        path = (
            f"{parsed.path.rstrip('/')}/agents/{quote(name, safe='')}"
            "/endpoint/protocols/voice/live/sessions"
        )
        self._connection_url = urlunsplit(("wss", parsed.netloc, path, urlencode(query), ""))
        self._foundry_delegation = delegation
        self._credential = credential
        self._credential_owned = credential is None
        # Initialize the shared engine without its direct-model API-key/deployment validation.
        # Foundry owns that configuration and obtains its credential asynchronously.
        llm.DuplexModel.__init__(
            self,
            capabilities=llm.DuplexCapabilities(
                user_transcription=True,
                auto_tool_reply_generation=True,
                mutable_chat_context=False,
                mutable_instructions=False,
                mutable_tools=False,
            ),
        )
        self._opts = _impl._LiveOptions(
            model=name,
            voice=_impl.DEFAULT_VOICE,
            base_url=endpoint,
            delegation="client" if delegation == "client" else "responses",
            responses={},
            decisions={},
            triggers=None,
            delegation_recovery=None,
            service_tier=None,
            api_key=None,
            entra_token=None,
            is_azure=True,
            max_session_duration=None,
            conn_options=conn_options,
        )
        self._http_session = http_session
        self._http_session_owned = False
        self._owned_detector = None
        self._provider_label = "Azure Foundry GPT Live"

    async def _get_token(self) -> str:
        if self._credential is None:
            self._credential = DefaultAzureCredential()
        token = await self._credential.get_token(_TOKEN_SCOPE)
        return token.token

    def session(self) -> GPTLiveSession:
        return GPTLiveSession(self)

    async def aclose(self) -> None:
        try:
            await super().aclose()
        finally:
            if self._credential_owned and self._credential is not None:
                await self._credential.close()
                self._credential = None


class GPTLiveSession(_impl.GPTLiveSession):
    """A full-duplex session started by Foundry from a saved voice agent."""

    def __init__(self, duplex_model: GPTLiveModel) -> None:
        self._foundry_model = duplex_model
        self._conversation_id: str | None = None
        self._first_started_fut: asyncio.Future[None] = asyncio.Future()
        super().__init__(duplex_model)

    @property
    def conversation_id(self) -> str | None:
        """Foundry's conversation ID, distinct from the GPT Live session ID."""
        return self._conversation_id

    async def wait_until_ready(self) -> None:
        """Wait for Foundry to start the first session, or raise if startup fails."""
        await asyncio.wait(
            (self._first_started_fut, self._main_atask),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if self._main_atask.done():
            await self._main_atask
            raise APIConnectionError(
                "Foundry GPT Live session closed before readiness", retryable=False
            )
        await self._first_started_fut

    async def _create_ws_conn(self) -> aiohttp.ClientWebSocketResponse:
        try:
            self._opts.entra_token = await asyncio.wait_for(
                self._foundry_model._get_token(), self._opts.conn_options.timeout
            )
        except ClientAuthenticationError:
            raise APIConnectionError(
                "Foundry Entra authentication failed", retryable=False
            ) from None
        except asyncio.TimeoutError:
            raise APIConnectionError("Foundry Entra authentication timed out") from None
        return await super()._create_ws_conn()

    def _create_ws_url_and_headers(self) -> tuple[str, dict[str, str]]:
        return self._foundry_model._connection_url, {
            "Authorization": f"Bearer {self._opts.entra_token}",
            "Foundry-Features": "VoiceAgents=V1Preview",
            "User-Agent": "LiveKit Agents",
        }

    def _create_ws_response_error(self, error: aiohttp.ClientResponseError) -> APIStatusError:
        return APIStatusError(
            "Foundry GPT Live WebSocket handshake failed",
            status_code=error.status,
            retryable=error.status == 429 or error.status >= 500,
        )

    async def _ws_send(
        self, ws_conn: aiohttp.ClientWebSocketResponse, event: types.ClientEvent | dict[str, Any]
    ) -> None:
        event_type = event.get("type") if isinstance(event, dict) else event.type
        if event_type != "session.start":
            await super()._ws_send(ws_conn, event)
            return
        # Intercept the engine's startup event: Foundry already supplied its saved definition.
        try:
            await asyncio.wait_for(
                asyncio.shield(self._session_started_fut), self._opts.conn_options.timeout
            )
        except asyncio.TimeoutError:
            raise APIConnectionError("Foundry GPT Live session startup timed out") from None

        if self._instructions:
            await self._ws_send(
                ws_conn,
                types.InstructionsAppendEvent(delegation_id=None, content=self._instructions),
            )
        # A reconnect opens a new Foundry conversation, not a resume of the old one.
        history = [
            f"{item.role}: {item.text_content}"
            for item in self._history.items
            if isinstance(item, llm.ChatMessage) and item.text_content
        ]
        if history:
            await self._ws_send(
                ws_conn,
                types.ThinkingAppendEvent(delegation_id=None, content="\n".join(history)),
            )

    def _handle_event(self, event: dict[str, Any]) -> None:
        if event.get("type") == "session.started":
            session = event.get("session") or {}
            audio_format = (session.get("audio") or {}).get("format") or {}
            if (
                audio_format.get("type") != "audio/pcm"
                or audio_format.get("rate") != _impl.SAMPLE_RATE
            ):
                raise APIConnectionError(
                    "Foundry agent must be configured for PCM16 audio at 24000 Hz",
                    retryable=False,
                )
            delegation = (session.get("delegation") or {}).get("type")
            if delegation is not None and delegation != self._foundry_model._foundry_delegation:
                raise APIConnectionError(
                    "Foundry agent delegation does not match the configured delegation",
                    retryable=False,
                )
            self._conversation_id = event.get("conversation_id")
        super()._handle_event(event)
        if event.get("type") == "session.started" and not self._first_started_fut.done():
            self._first_started_fut.set_result(None)

    def _reset_for_reconnect(self) -> None:
        super()._reset_for_reconnect()
        self._conversation_id = None

    async def _update_tools(self, tools: list[llm.Tool]) -> None:
        if self._opts.delegation == "client" and tools:
            raise llm.RealtimeError(
                "Foundry client delegation requires a delegation_created handler, not local tools"
            )
        # Bind implementations for tools declared in Foundry; never replace its saved tools.
        self._tools = llm.ToolContext(tools)

    def _update_options(
        self, *, tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN
    ) -> None:
        if is_given(tool_choice):
            raise llm.RealtimeError("Configure Foundry delegation settings on the saved agent")
