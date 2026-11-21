from __future__ import annotations

import pytest

from examples.voice_agents.create_foundry_gpt_live import validate_account, validate_deployment

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
