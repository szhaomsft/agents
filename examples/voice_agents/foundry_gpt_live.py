"""Use an existing Foundry GPT Live voice agent with LiveKit.

Configure FOUNDRY_PROJECT_ENDPOINT and AGENT_NAME, then authenticate with `az login`.
Copy examples/voice_agents/.env.example to examples/voice_agents/.env for settings.
The saved agent must use the live sub-protocol, Responses delegation and PCM16
audio at 24000 Hz. Declare get_current_time in delegation.responses.tools using
the schema in the Azure plugin README, then ask "What is the current UTC time?"
Run: python foundry_gpt_live.py console
"""

from datetime import datetime, timezone
from typing import cast

from dotenv import load_dotenv

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli, function_tool
from livekit.plugins import azure

load_dotenv()
server = AgentServer()


@function_tool
async def get_current_time() -> str:
    """Get the current UTC time."""
    return datetime.now(timezone.utc).isoformat()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    model = azure.realtime.GPTLiveModel()
    ctx.add_shutdown_callback(model.aclose)
    session: AgentSession[None] = AgentSession(llm=model)
    # Foundry supplies the instructions, voice and greeting from the saved definition.
    agent = Agent(instructions="", tools=[get_current_time])
    await session.start(agent=agent, room=ctx.room)
    duplex = cast(azure.realtime.GPTLiveSession, agent.duplex_session)
    await duplex.wait_until_ready()
    print("\nReady to speak.", flush=True)


if __name__ == "__main__":
    cli.run_app(server)
