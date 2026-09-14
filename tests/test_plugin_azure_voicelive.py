import inspect

import pytest

from livekit.plugins.azure.realtime import RealtimeModel, RealtimeSession

pytestmark = pytest.mark.plugin("azure")


def test_realtime_interface_compatibility() -> None:
    session_parameters = inspect.signature(RealtimeModel.session).parameters
    assert "turn_detection_disabled" in session_parameters

    generate_reply_parameters = inspect.signature(RealtimeSession.generate_reply).parameters
    assert "tool_choice" in generate_reply_parameters
    assert "tools" in generate_reply_parameters
