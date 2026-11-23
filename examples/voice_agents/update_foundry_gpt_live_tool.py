# /// script
# dependencies = ["azure-identity>=1.25.1", "python-dotenv>=1.0.0"]
# ///
"""Update a local tool schema or managed model on a saved GPT Live agent.

Run with uv run --script. The default is a dry run; pass --apply to create a
new immutable version of the saved agent. Pass --model to update only the
managed voice model instead of the get_current_time tool schema.
"""

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from typing import Any
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

CURRENT_TIME_TOOL: dict[str, Any] = {
    "type": "function",
    "name": "get_current_time",
    "description": "Get the current UTC time.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    },
    "strict": True,
}


def _validate_gpt_live_agent(definition: dict[str, Any]) -> None:
    if definition.get("kind") != "voice" or definition.get("sub_protocol") != "live":
        raise ValueError("The selected agent version must use the GPT Live voice protocol")


def update_tool_schema(definition: dict[str, Any]) -> bool:
    """Update the local time tool declaration; return whether the definition changed."""
    _validate_gpt_live_agent(definition)
    delegation = definition.get("delegation")
    if not isinstance(delegation, dict) or delegation.get("type") != "responses":
        raise ValueError("The selected agent version must use Responses delegation")

    responses = delegation.get("responses")
    if not isinstance(responses, dict):
        raise ValueError("The selected agent version has no Responses configuration")
    tools = responses.setdefault("tools", [])
    if not isinstance(tools, list):
        raise ValueError("The Responses tools setting must be a list")

    for index, tool in enumerate(tools):
        if (
            isinstance(tool, dict)
            and tool.get("type") == "function"
            and tool.get("name") == "get_current_time"
        ):
            updated_tool = {**tool, **deepcopy(CURRENT_TIME_TOOL)}
            if updated_tool == tool:
                return False
            tools[index] = updated_tool
            return True

    tools.append(deepcopy(CURRENT_TIME_TOOL))
    return True


def update_managed_model(definition: dict[str, Any], model: str) -> bool:
    """Update the managed voice model; return whether the definition changed."""
    _validate_gpt_live_agent(definition)
    if definition.get("model_type") != "managed":
        raise ValueError("The selected agent version must use a managed voice model")
    if not model or model != model.strip():
        raise ValueError("The managed voice model name must not be empty or padded with whitespace")
    if definition.get("model") == model:
        return False
    definition["model"] = model
    return True


def request_json(url: str, token: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    request = Request(
        url,
        data=None if body is None else json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Foundry-Features": "VoiceAgents=V1Preview",
        },
    )
    with urlopen(request, timeout=30) as response:
        result: dict[str, Any] = json.load(response)
        return result


def main() -> None:
    from azure.identity import DefaultAzureCredential
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-endpoint", default=os.environ.get("FOUNDRY_PROJECT_ENDPOINT"))
    parser.add_argument("--agent-name", default=os.environ.get("AGENT_NAME"))
    parser.add_argument(
        "--model",
        help="Update only the managed voice model instead of the local tool schema",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Create a new immutable agent version (default: show changes without saving)",
    )
    args = parser.parse_args()
    if not args.project_endpoint:
        parser.error("Set FOUNDRY_PROJECT_ENDPOINT or pass --project-endpoint")
    if not args.agent_name:
        parser.error("Set AGENT_NAME or pass --agent-name")

    endpoint = urlsplit(args.project_endpoint)
    if (
        endpoint.scheme != "https"
        or not endpoint.netloc
        or not endpoint.path.rstrip("/").startswith("/api/projects/")
        or endpoint.query
        or endpoint.fragment
    ):
        parser.error("--project-endpoint must be an HTTPS Foundry project endpoint")

    project_url = args.project_endpoint.rstrip("/")
    agent_path = f"{project_url}/agents/{quote(args.agent_name, safe='')}"
    api_version = "api-version=v1"

    with DefaultAzureCredential() as credential:
        token = credential.get_token("https://ai.azure.com/.default").token
        agent = request_json(f"{agent_path}?{api_version}", token)
        latest = agent.get("versions", {}).get("latest")
        if not isinstance(latest, dict):
            raise ValueError("Foundry did not return the saved agent's latest version")
        definition = latest.get("definition")
        if not isinstance(definition, dict):
            raise ValueError("Foundry did not return the latest agent definition")

        if args.model is None:
            changed = update_tool_schema(definition)
            update_description = "add or update get_current_time"
            preview: dict[str, Any] = CURRENT_TIME_TOOL
        else:
            changed = update_managed_model(definition, args.model)
            update_description = "change the managed voice model"
            preview = {"model": args.model}
        current_version = latest.get("version", "unknown")
        if not changed:
            if args.model is None:
                print(
                    f"{args.agent_name} version {current_version} already has the requested schema."
                )
            else:
                print(f"{args.agent_name} version {current_version} already uses {args.model}.")
            return

        if not args.apply:
            print(
                f"Dry run: {args.agent_name} version {current_version} would {update_description}:"
            )
            print(json.dumps(preview, indent=2))
            print("Review the change, then rerun with --apply to create a new version.")
            return

        version_body: dict[str, Any] = {"definition": definition}
        for field in ("description", "metadata", "draft"):
            if field in latest:
                version_body[field] = latest[field]
        created = request_json(f"{agent_path}/versions?{api_version}", token, version_body)
        print(f"Created {args.agent_name} version {created.get('version', 'unknown')}.")


if __name__ == "__main__":
    main()
