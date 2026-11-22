from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from examples.voice_agents.update_foundry_gpt_live_tool import (
    CURRENT_TIME_TOOL,
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
