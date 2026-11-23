# /// script
# dependencies = ["azure-ai-projects>=2.11.0", "azure-identity>=1.25.1"]
# ///
"""Create a Foundry GPT Live agent using managed or self-deployed Data Zone models.

Run with uv run --script; use --check-only to validate without creating an agent.
Requires management-plane read access and Foundry agent creation permission.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen
from uuid import uuid4


def validate_account(
    account: dict[str, Any],
    locations: dict[str, Any],
    project_endpoint: str,
    *,
    require_central_us: bool = False,
) -> None:
    location = account["location"].lower()
    if require_central_us and location != "centralus":
        raise ValueError(f"Managed GPT Live requires Central US, not {location!r}")
    us_locations = {
        entry["name"].lower()
        for entry in locations["value"]
        if entry.get("metadata", {}).get("geographyGroup") == "US"
    }
    if location not in us_locations:
        raise ValueError(f"Account region {location!r} is not a verified US Azure region")
    endpoint = urlsplit(project_endpoint)
    subdomain = account["properties"]["customSubDomainName"]
    if (
        endpoint.scheme != "https"
        or endpoint.netloc != f"{subdomain}.services.ai.azure.com"
        or not re.fullmatch(r"/api/projects/[^/]+/?", endpoint.path)
        or endpoint.query
        or endpoint.fragment
    ):
        raise ValueError("Project endpoint must belong to the verified US Foundry account")


def validate_deployment(deployment: dict[str, Any], *, voice: bool) -> None:
    name = deployment["name"]
    sku = deployment["sku"]["name"]
    if sku not in {"DataZoneStandard", "DataZoneProvisionedManaged"}:
        raise ValueError(f"Deployment {name!r} uses {sku!r}, not a realtime Data Zone SKU")
    properties = deployment["properties"]
    if properties["provisioningState"] != "Succeeded":
        raise ValueError(f"Deployment {name!r} is not successfully provisioned")
    if voice and properties["model"]["name"] != "gpt-live-1":
        raise ValueError(f"Voice deployment {name!r} must serve gpt-live-1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-endpoint", default=os.environ.get("FOUNDRY_PROJECT_ENDPOINT"))
    parser.add_argument("--account-resource-id", required=True)
    parser.add_argument(
        "--model-type", choices=("managed", "self_deployed"), default="self_deployed"
    )
    parser.add_argument("--voice-deployment")
    parser.add_argument("--backend-deployment")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not args.project_endpoint:
        parser.error("Set FOUNDRY_PROJECT_ENDPOINT or pass --project-endpoint")
    args.account_resource_id = args.account_resource_id.rstrip("/")
    match = re.fullmatch(
        r"/subscriptions/([0-9a-fA-F-]+)/resourceGroups/[^/?#]+"
        r"/providers/Microsoft.CognitiveServices/accounts/[^/?#]+",
        args.account_resource_id,
    )
    if match is None:
        parser.error("--account-resource-id must be an Azure Cognitive Services account ARM ID")
    args.subscription_id = match.group(1)
    if args.model_type == "managed":
        if args.voice_deployment is not None or args.backend_deployment is not None:
            parser.error("Managed models do not use --voice-deployment or --backend-deployment")
    else:
        if not args.voice_deployment or not args.backend_deployment:
            parser.error("Self-deployed models require --voice-deployment and --backend-deployment")
        for name in (args.voice_deployment, args.backend_deployment):
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
                parser.error("Deployment names must contain only letters, digits, '.', '_' or '-'")
    return args


def main() -> None:
    from azure.ai.projects import AIProjectClient
    from azure.ai.projects.models import (
        LiveInitialSessionAudioOutputParam,
        LiveInitialSessionAudioParam,
        LiveResponsesDelegationSettingsInputParam,
        LiveSessionAudioFormatPCMParam,
        VoiceAgentLiveDefinition,
        VoiceAgentLiveResponsesDelegation,
    )
    from azure.identity import DefaultAzureCredential

    args = parse_args()
    resource_id = args.account_resource_id

    with DefaultAzureCredential() as credential:
        token = credential.get_token("https://management.azure.com/.default").token

        def read_arm(path: str, api_version: str = "2025-06-01") -> dict[str, Any]:
            request = Request(
                f"https://management.azure.com{quote(path, safe='/')}?api-version={api_version}",
                headers={"Authorization": f"Bearer {token}"},
            )
            with urlopen(request, timeout=30) as response:
                result: dict[str, Any] = json.load(response)
                return result

        account = read_arm(resource_id)
        locations = read_arm(
            f"/subscriptions/{args.subscription_id}/locations", api_version="2022-12-01"
        )
        validate_account(
            account,
            locations,
            args.project_endpoint,
            require_central_us=args.model_type == "managed",
        )
        if args.model_type == "managed":
            voice_model = "gpt-live-1-datazone"
            backend_model = "gpt-6-luna"
            print(f"Verified managed Data Zone setup in {account['location']}")
        else:
            for name, voice in ((args.voice_deployment, True), (args.backend_deployment, False)):
                deployment = read_arm(f"{resource_id}/deployments/{name}")
                validate_deployment(deployment, voice=voice)
                print(f"Verified {name}: {deployment['sku']['name']} in {account['location']}")
            voice_model = args.voice_deployment
            backend_model = args.backend_deployment

        definition = VoiceAgentLiveDefinition(
            model=voice_model,
            model_type=args.model_type,
            instructions="Be a helpful, concise voice assistant. Delegate reasoning when needed.",
            audio=LiveInitialSessionAudioParam(
                format=LiveSessionAudioFormatPCMParam(rate=24000),
                output=LiveInitialSessionAudioOutputParam(voice="marin"),
            ),
            delegation=VoiceAgentLiveResponsesDelegation(
                responses=LiveResponsesDelegationSettingsInputParam(
                    model=backend_model,
                    instructions="Give accurate, concise guidance to the voice assistant.",
                )
            ),
            store=False,
        )
        if args.check_only:
            print("Validation complete; no agent or deployment was created.")
            return

        # A unique name avoids adding versions to an existing customer or demo agent.
        name = f"livekit-us-dz-{uuid4().hex}"
        with AIProjectClient(
            endpoint=args.project_endpoint, credential=credential, allow_preview=True
        ) as project:
            version = project.agents.create_version(name, definition=definition)
        print(f"Created agent {name}, version {version.version}. This agent is retained.")
        print(f"Set AGENT_NAME={name}, then run examples/voice_agents/foundry_gpt_live.py console")


if __name__ == "__main__":
    main()
