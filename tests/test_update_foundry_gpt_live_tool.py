from __future__ import annotations

import sys
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

from examples.voice_agents import update_foundry_gpt_live_tool as updater
from examples.voice_agents.update_foundry_gpt_live_tool import (
    CURRENT_TIME_TOOL,
    update_managed_model,
    update_tool_schema,
)

pytestmark = pytest.mark.unit


def _definition() -> dict[str, Any]:
    return {
        "kind": "voice",
        "sub_protocol": "live",
        "delegation": {
            "type": "responses",
            "responses": {
                "model": "backend",
                "instructions": "Use the available tools.",
                "tools": [
                    {
                        "type": "function",
                        "name": "lookup",
                        "description": "Look up a value.",
                        "parameters": {"type": "object"},
                    }
                ],
            },
        },
    }


def test_adds_schema_without_changing_other_agent_settings() -> None:
    definition = _definition()
    original_instructions = definition["delegation"]["responses"]["instructions"]
    original_tools = deepcopy(definition["delegation"]["responses"]["tools"])

    assert update_tool_schema(definition)
    responses = definition["delegation"]["responses"]
    assert responses["tools"] == [*original_tools, CURRENT_TIME_TOOL]
    assert responses["instructions"] == original_instructions


def test_updates_existing_schema_and_is_idempotent() -> None:
    definition = _definition()
    tools = definition["delegation"]["responses"]["tools"]
    tools.append({"type": "function", "name": "get_current_time", "description": "Old schema"})

    assert update_tool_schema(definition)
    assert tools[1] == {**{"description": "Old schema"}, **CURRENT_TIME_TOOL}
    assert not update_tool_schema(definition)
    assert len(tools) == 2


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sub_protocol", "realtime"),
        ("delegation", {"type": "client"}),
    ],
)
def test_rejects_non_gpt_live_responses_agents(field: str, value: object) -> None:
    definition = _definition()
    definition[field] = value
    with pytest.raises(ValueError):
        update_tool_schema(definition)


def test_updates_managed_model_without_changing_other_settings() -> None:
    definition = {**_definition(), "model_type": "managed", "model": "gpt-live-1"}
    original = deepcopy(definition)

    assert update_managed_model(definition, "gpt-live-1-datazone")
    assert definition == {**original, "model": "gpt-live-1-datazone"}
    assert not update_managed_model(definition, "gpt-live-1-datazone")


@pytest.mark.parametrize(
    ("field", "value", "model"),
    [
        ("kind", "prompt", "gpt-live-1-datazone"),
        ("sub_protocol", "realtime", "gpt-live-1-datazone"),
        ("model_type", "self_deployed", "gpt-live-1-datazone"),
        ("model_type", "managed", " "),
        ("model_type", "managed", " gpt-live-1-datazone"),
    ],
)
def test_rejects_invalid_managed_model_updates(field: str, value: str, model: str) -> None:
    definition = {**_definition(), "model_type": "managed", "model": "gpt-live-1"}
    definition[field] = value
    original = deepcopy(definition)

    with pytest.raises(ValueError):
        update_managed_model(definition, model)
    assert definition == original


@pytest.mark.parametrize("apply", [False, True])
def test_model_update_is_dry_run_by_default_and_preserves_agent_settings(
    apply: bool, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class Credential:
        def __enter__(self) -> Credential:
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def get_token(self, scope: str) -> SimpleNamespace:
            assert scope == "https://ai.azure.com/.default"
            return SimpleNamespace(token="test-token")

    definition = {**_definition(), "model_type": "managed", "model": "gpt-live-1"}
    latest = {
        "version": "3",
        "definition": definition,
        "description": "A voice agent",
        "metadata": {"owner": "example"},
        "draft": False,
    }
    calls: list[tuple[str, dict[str, Any] | None]] = []

    def request_json(url: str, token: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        assert token == "test-token"
        calls.append((url, deepcopy(body)))
        return {"versions": {"latest": deepcopy(latest)}} if body is None else {"version": "4"}

    monkeypatch.setattr("azure.identity.DefaultAzureCredential", Credential)
    monkeypatch.setattr("dotenv.load_dotenv", lambda: True)
    monkeypatch.setattr(updater, "request_json", request_json)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "update_foundry_gpt_live_tool.py",
            "--project-endpoint",
            "https://resource.services.ai.azure.com/api/projects/test",
            "--agent-name",
            "test-agent",
            "--model",
            "gpt-live-1-datazone",
            *(["--apply"] if apply else []),
        ],
    )

    updater.main()

    assert len(calls) == (2 if apply else 1)
    if apply:
        url, body = calls[1]
        assert url.endswith("/agents/test-agent/versions?api-version=v1")
        assert body == {
            "definition": {**definition, "model": "gpt-live-1-datazone"},
            "description": latest["description"],
            "metadata": latest["metadata"],
            "draft": False,
        }
    else:
        assert "gpt-live-1-datazone" in capsys.readouterr().out
