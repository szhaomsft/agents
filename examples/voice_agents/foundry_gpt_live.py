"""Use an existing Foundry GPT Live voice agent with LiveKit.

Configure FOUNDRY_PROJECT_ENDPOINT and AGENT_NAME, then authenticate with `az login`.
The saved agent must use the live sub-protocol and PCM16 audio at 24000 Hz.
Run: python foundry_gpt_live.py console
"""

from typing import cast

from dotenv import load_dotenv

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli
from livekit.plugins import azure

load_dotenv()
server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    model = azure.realtime.GPTLiveModel()
    ctx.add_shutdown_callback(model.aclose)
    session = AgentSession(llm=model)
    # Foundry supplies the instructions, voice and greeting from the saved definition.
    agent = Agent(instructions="")
    await session.start(agent=agent, room=ctx.room)
    duplex = cast(azure.realtime.GPTLiveSession, agent.duplex_session)
    await duplex.wait_until_ready()
    print("\nReady to speak.", flush=True)


if __name__ == "__main__":
    cli.run_app(server)
