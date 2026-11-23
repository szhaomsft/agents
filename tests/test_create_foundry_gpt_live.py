from __future__ import annotations

import io
import json
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from examples.voice_agents import create_foundry_gpt_live as creator
from examples.voice_agents.create_foundry_gpt_live import (
    parse_args,
    validate_account,
    validate_deployment,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("region", "host", "valid"),
    [
        ("eastus2", "example", True),
        ("francecentral", "example", False),
        ("unknown", "example", False),
        ("eastus2", "another-account", False),
    ],
)
def test_creation_requires_the_verified_us_account(region: str, host: str, valid: bool) -> None:
    account = {"location": region, "properties": {"customSubDomainName": "example"}}
    locations = {
        "value": [
            {"name": "eastus2", "metadata": {"geographyGroup": "US"}},
            {"name": "francecentral", "metadata": {"geographyGroup": "Europe"}},
        ]
    }
    endpoint = f"https://{host}.services.ai.azure.com/api/projects/demo"
    if valid:
        validate_account(account, locations, endpoint)
    else:
        with pytest.raises(ValueError):
            validate_account(account, locations, endpoint)


@pytest.mark.parametrize(("region", "valid"), [("centralus", True), ("eastus2", False)])
def test_managed_creation_requires_central_us(region: str, valid: bool) -> None:
    account = {"location": region, "properties": {"customSubDomainName": "example"}}
    locations = {
        "value": [
            {"name": region, "metadata": {"geographyGroup": "US"}},
        ]
    }
    endpoint = "https://example.services.ai.azure.com/api/projects/demo"
    if valid:
        validate_account(account, locations, endpoint, require_central_us=True)
    else:
        with pytest.raises(ValueError, match="Central US"):
            validate_account(account, locations, endpoint, require_central_us=True)


@pytest.mark.parametrize(
    ("extra", "valid"),
    [
        (["--model-type", "managed"], True),
        (["--model-type", "managed", "--voice-deployment", "voice"], False),
        (["--model-type", "managed", "--backend-deployment", "backend"], False),
        ([], False),
        (["--voice-deployment", "voice", "--backend-deployment", "backend"], True),
    ],
)
def test_model_type_requires_the_right_deployment_options(
    extra: list[str], valid: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "create_foundry_gpt_live.py",
            "--account-resource-id",
            "/subscriptions/1234/resourceGroups/group/providers/"
            "Microsoft.CognitiveServices/accounts/account",
            "--project-endpoint",
            "https://example.services.ai.azure.com/api/projects/demo",
            *extra,
        ],
    )
    if valid:
        args = parse_args()
        assert args.model_type == ("managed" if "managed" in extra else "self_deployed")
    else:
        with pytest.raises(SystemExit):
            parse_args()


@pytest.mark.parametrize("check_only", [False, True])
def test_managed_creation_uses_data_zone_models_without_deployments(
    check_only: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Model:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    class Credential:
        def __enter__(self) -> Credential:
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def get_token(self, scope: str) -> SimpleNamespace:
            return SimpleNamespace(token="test-token")

    created: list[Model] = []

    class Project:
        def __init__(self, **kwargs: Any) -> None:
            self.agents = self

        def __enter__(self) -> Project:
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def create_version(self, name: str, *, definition: Model) -> SimpleNamespace:
            created.append(definition)
            return SimpleNamespace(version="1")

    def read_arm(request: Any, *, timeout: int) -> io.BytesIO:
        if "/locations?" in request.full_url:
            data = {"value": [{"name": "centralus", "metadata": {"geographyGroup": "US"}}]}
        else:
            assert "/deployments/" not in request.full_url
            data = {
                "location": "centralus",
                "properties": {"customSubDomainName": "example"},
            }
        return io.BytesIO(json.dumps(data).encode())

    projects = ModuleType("azure.ai.projects")
    projects.__dict__["AIProjectClient"] = Project
    models = ModuleType("azure.ai.projects.models")
    for name in (
        "LiveInitialSessionAudioOutputParam",
        "LiveInitialSessionAudioParam",
        "LiveResponsesDelegationSettingsInputParam",
        "LiveSessionAudioFormatPCMParam",
        "VoiceAgentLiveDefinition",
        "VoiceAgentLiveResponsesDelegation",
    ):
        setattr(models, name, Model)
    identity = ModuleType("azure.identity")
    identity.__dict__["DefaultAzureCredential"] = Credential
    monkeypatch.setitem(sys.modules, "azure.ai.projects", projects)
    monkeypatch.setitem(sys.modules, "azure.ai.projects.models", models)
    monkeypatch.setitem(sys.modules, "azure.identity", identity)
    monkeypatch.setattr(creator, "urlopen", read_arm)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "create_foundry_gpt_live.py",
            "--account-resource-id",
            "/subscriptions/1234/resourceGroups/group/providers/"
            "Microsoft.CognitiveServices/accounts/account",
            "--project-endpoint",
            "https://example.services.ai.azure.com/api/projects/demo",
            "--model-type",
            "managed",
            *(["--check-only"] if check_only else []),
        ],
    )

    creator.main()

    assert len(created) == (0 if check_only else 1)
    if created:
        definition = created[0]
        assert definition.model_type == "managed"
        assert definition.model == "gpt-live-1-datazone"
        assert definition.delegation.responses.model == "gpt-6-luna"
        assert definition.audio.format.rate == 24000
        assert definition.store is False


@pytest.mark.parametrize("voice", [True, False])
@pytest.mark.parametrize(
    ("sku", "state", "model", "valid"),
    [
        ("DataZoneStandard", "Succeeded", "gpt-live-1", True),
        ("DataZoneProvisionedManaged", "Succeeded", "gpt-live-1", True),
        ("GlobalStandard", "Succeeded", "gpt-live-1", False),
        ("DataZoneBatch", "Succeeded", "gpt-live-1", False),
        ("DataZoneStandard", "Creating", "gpt-live-1", False),
        ("DataZoneStandard", "Succeeded", "gpt-5.4", True),
    ],
)
def test_creation_requires_data_zone_for_both_models(
    voice: bool, sku: str, state: str, model: str, valid: bool
) -> None:
    deployment = {
        "name": "deployment",
        "sku": {"name": sku},
        "properties": {"provisioningState": state, "model": {"name": model}},
    }
    valid = valid and (not voice or model == "gpt-live-1")
    if valid:
        validate_deployment(deployment, voice=voice)
    else:
        with pytest.raises(ValueError):
            validate_deployment(deployment, voice=voice)
