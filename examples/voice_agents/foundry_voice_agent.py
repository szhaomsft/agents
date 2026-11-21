"""Chat with a saved Foundry GPT Realtime or cascade agent using the Voice Agent SDK.

Install livekit-plugins-azure[agents], authenticate with az login, and configure
FOUNDRY_PROJECT_ENDPOINT and AGENT_NAME. Optional FOUNDRY_STRUCTURED_INPUTS is a
JSON object containing only inputs declared by the saved agent.
Run: python examples/voice_agents/foundry_voice_agent.py console
"""

import json
import os
from typing import cast

from dotenv import load_dotenv

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli
from livekit.plugins import azure

load_dotenv()
server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    inputs = json.loads(os.environ.get("FOUNDRY_STRUCTURED_INPUTS", "null"))
    if inputs is not None and not isinstance(inputs, dict):
        raise ValueError("FOUNDRY_STRUCTURED_INPUTS must be a JSON object")
    model = azure.realtime.VoiceAgentModel(structured_inputs=inputs)
    ctx.add_shutdown_callback(model.aclose)
    session = AgentSession(llm=model)
    agent = Agent(instructions="")
    await session.start(agent=agent, room=ctx.room)
    realtime = cast(azure.realtime.VoiceAgentSession, agent.realtime_llm_session)
    await realtime.wait_until_ready()
    print("\nReady to speak.", flush=True)


if __name__ == "__main__":
    cli.run_app(server)
