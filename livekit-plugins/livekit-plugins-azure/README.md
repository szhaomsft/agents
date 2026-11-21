# Azure Plugin for LiveKit Agents

Complete Azure AI integration for LiveKit Agents, including Azure Speech Services and Azure Voice Live Realtime API.

**What's included:**
- **RealtimeModel** - Azure Voice Live for speech-to-speech
- **STT** - Powered by Azure Speech Services
- **TTS** - Powered by Azure Speech Services with neural voices

For Azure OpenAI LLM (non-realtime), see the [OpenAI plugin](https://github.com/livekit/agents/tree/main/livekit-plugins/livekit-plugins-openai) which supports `LLM.with_azure()`.

See [https://docs.livekit.io/agents/integrations/azure/](https://docs.livekit.io/agents/integrations/azure/) for more information.

## Installation

```bash
pip install livekit-plugins-azure

# For the Azure Voice Live realtime model
pip install "livekit-plugins-azure[realtime]"
```

## Realtime Mode (Azure Voice Live)

### Foundry GPT Live voice agents (full duplex)

Use `azure.realtime.GPTLiveModel` for an **existing Foundry voice agent** with
`kind="voice"` and `sub_protocol="live"`. This is distinct from `RealtimeModel`
below: GPT Live continuously listens and speaks, while Foundry owns its model,
voice, greeting, instructions and delegation backend.

Create/configure the agent using the
[Microsoft GPT Live samples](https://github.com/microsoft/voice-agent/tree/main/samples/gpt_live).
Configure PCM16 audio at **24000 Hz** in the saved agent definition. Authenticate
with `az login` or another identity supported by `DefaultAzureCredential`, and set:

```bash
export FOUNDRY_PROJECT_ENDPOINT=https://<resource>.services.ai.azure.com/api/projects/<project>
export AGENT_NAME=<saved-voice-agent-name>
```

```python
from livekit.agents import Agent, AgentSession
from livekit.plugins import azure

model = azure.realtime.GPTLiveModel()
session = AgentSession(llm=model)
await session.start(agent=Agent(instructions=""), room=ctx.room)
```

Optionally pass `agent_version` to pin an existing saved version, or
`structured_inputs` for template placeholders declared by your agent.
Omit `structured_inputs` when the agent declares none: undeclared inputs cause an
HTTP 400 WebSocket handshake failure. Foundry starts the session
and fills its templates; the client waits for `session.started` and does **not**
send `session.start`. Nonempty LiveKit `Agent.instructions` are appended as
additional instructions, not substituted for the saved definition. Initial chat
history is appended as context. Configure frontend model, voice, and backend
settings in Foundry rather than passing direct OpenAI deployment options.

Set `delegation` to match the saved agent: `"responses"` (default), `"client"`,
or `"foundry_managed_agent"`. Responses and managed-agent backends support local
`@function_tool` implementations for functions **already declared in Foundry**;
the plugin does not replace their saved tool schemas. Client delegation instead
requires a `delegation_created` handler on `agent.duplex_session` that calls
`append_commentary(text, delegation_id=event.id)` with the answer.

Authentication uses the `https://ai.azure.com/.default` scope and the
`Foundry-Features: VoiceAgents=V1Preview` header. Tokens are reacquired for every
connection. You can pass an async Azure `credential` (for example
`AzureCliCredential`) and a shared `http_session`; these remain caller-owned.
Close the model with `await model.aclose()` after its sessions to release any
internally owned resources. No API key is required.

`store=None` preserves the saved agent's storage policy; explicitly pass `True`
or `False` to override it. `GPTLiveSession.conversation_id` exposes the Foundry
conversation ID separately from `session_id`. A reconnect opens a **new**
conversation and supplies previous transcripts as context; it does not resume
the old backend conversation or pending tool execution.

See `examples/voice_agents/foundry_gpt_live.py` for a runnable LiveKit agent.

#### Create an agent for the US Data Zone

Data Zone routing is selected by the **model deployment SKU**, not by an agent
flag, its name, or a US project endpoint alone. The previously shown connection
example does not create an agent or change deployment routing.

Create a GPT Live deployment and a Responses-compatible backend deployment in the
same US Foundry account using `DataZoneStandard` (or supported
`DataZoneProvisionedManaged`). Both must use Data Zone SKUs: a global reasoning
backend would still process delegated content outside the intended boundary.
Availability and quota depend on the model and region.

The creation example verifies the account's US geography and the project endpoint,
checks both existing deployments, and creates a uniquely named saved agent. It
never provisions models or modifies an existing agent. It uses the Azure Projects
SDK only for authoring; the runtime plugin does not require that SDK.

```powershell
az login
$env:FOUNDRY_PROJECT_ENDPOINT = 'https://<account>.services.ai.azure.com/api/projects/<project>'
$account = '/subscriptions/<subscription>/resourceGroups/<group>/providers/Microsoft.CognitiveServices/accounts/<account>'

uv run --script examples/voice_agents/create_foundry_gpt_live.py `
  --account-resource-id $account `
  --voice-deployment '<gpt-live-us-datazone-deployment>' `
  --backend-deployment '<responses-us-datazone-deployment>' `
  --check-only
```

Remove `--check-only` to create and retain a new agent. The command requires Azure
management-plane read access and Foundry agent creation permission. Set
`AGENT_NAME` to the printed name, then run
`python examples/voice_agents/foundry_gpt_live.py console` using your LiveKit environment.
The agent uses PCM16/24 kHz, `marin`, Responses delegation and `store=False`.

These checks cover the two model deployments, not every part of a customer's
data-residency policy. Review Foundry storage, tools, telemetry and your LiveKit
hosting separately. See Microsoft's
[deployment-type documentation](https://learn.microsoft.com/en-us/azure/foundry/foundry-models/concepts/deployment-types).

Offline protocol tests run with:

```bash
uv run pytest tests/test_plugin_azure_gpt_live.py --unit
```

An opt-in, billable deployment smoke test checks server startup, generated audio,
and its transcript. Set `FOUNDRY_GPT_LIVE_SMOKE=1` alongside the project endpoint
and agent name, then run:

```bash
uv run pytest tests/test_plugin_azure_gpt_live_integration.py --plugin azure
```

If needed, set `FOUNDRY_GPT_LIVE_STRUCTURED_INPUTS` to a JSON object matching the
saved agent's placeholders and `FOUNDRY_GPT_LIVE_DELEGATION` to its delegation
type. The test uses your existing agent and does not provision or delete resources.

### Azure Credentials

For the realtime speech-to-speech model:

```bash
export AZURE_VOICE_LIVE_ENDPOINT=https://<region>.api.cognitive.microsoft.com/
export AZURE_VOICE_LIVE_API_KEY=<your-speech-key>
export AZURE_VOICE_LIVE_MODEL=<model-name>  # e.g., gpt-realtime, gpt-4o, gpt-4o-mini, etc.
```

The legacy `AZURE_VOICELIVE_ENDPOINT` and `AZURE_VOICELIVE_API_KEY` environment
variables are also supported.

To authenticate with Microsoft Entra ID instead of an API key, pass `use_default_credential=True`, which uses [`DefaultAzureCredential`](https://learn.microsoft.com/en-us/python/api/azure-identity/azure.identity.aio.defaultazurecredential).

To power the intelligence of your voice agent, you have flexibility and choice in the generative AI model between GPT-Realtime, GPT-5, GPT-4.1, Phi, and more options. For supported models and regions, see [Supported models and regions](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/voice-live#supported-models-and-regions).



### Quick Start
Azure Voice Live provides end-to-end speech-to-speech:

```python
from livekit import agents
from livekit.agents import Agent, AgentServer, AgentSession
from livekit.plugins import azure

class Assistant(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a helpful voice assistant powered by Azure."
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions="Greet the user and offer assistance"
        )

server = AgentServer()

@server.rtc_session()
async def entrypoint(ctx: agents.JobContext):
    session = AgentSession(
        llm=azure.realtime.RealtimeModel(
            voice="en-US-AvaMultilingualNeural",
        )
    )

    await session.start(room=ctx.room, agent=Assistant())

if __name__ == "__main__":
    agents.cli.run_app(server)
```

### Advanced Configuration

For full control over Azure Voice Live settings:

```python
import os
from azure.ai.voicelive.models import AzureSemanticVadEn, AudioInputTranscriptionOptions
from livekit.agents import AgentSession
from livekit.plugins import azure

# Configure Semantic VAD for intelligent turn detection
# Options:
# - AzureSemanticVad: Default semantic VAD (multilingual)
# - AzureSemanticVadEn: English-only, optimized for English
# - AzureSemanticVadMultilingual: Explicit multilingual support
turn_detection = AzureSemanticVadEn(
    threshold=0.5,              # Voice activity detection threshold (0.0-1.0)
    silence_duration_ms=500,    # Silence duration before turn ends
    prefix_padding_ms=300,      # Audio padding before speech
    speech_duration_ms=200,     # Minimum speech duration to trigger detection
    remove_filler_words=True,   # Remove filler words like "um", "uh"
)

# Configure input audio transcription with language constraint
# This helps prevent language misidentification
# Use azure-speech with text models such as gpt-4o, whisper-1 with gpt-realtime or gpt-realtime-mini
input_audio_transcription = AudioInputTranscriptionOptions(
    model="azure-speech",
    language="en-US",  # Constrain to English for reliable detection
)

session = AgentSession(
    llm=azure.realtime.RealtimeModel(
        endpoint=os.getenv("AZURE_VOICE_LIVE_ENDPOINT"),
        api_key=os.getenv("AZURE_VOICE_LIVE_API_KEY"),
        model=os.getenv("AZURE_VOICE_LIVE_MODEL", "gpt-4o"),
        voice=os.getenv("AZURE_VOICE_LIVE_VOICE", "en-US-AvaMultilingualNeural"),
        input_audio_transcription=input_audio_transcription,
        turn_detection=turn_detection,
        tool_choice="auto",  # Enable function calling
    )
)
```

### RealtimeModel Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `endpoint` | Azure Voice Live endpoint URL | `AZURE_VOICE_LIVE_ENDPOINT` env var |
| `api_key` | Azure API key | `AZURE_VOICE_LIVE_API_KEY` env var |
| `use_default_credential` | Authenticate with `DefaultAzureCredential` instead of an API key | `False` |
| `model` | Model name | `AZURE_VOICE_LIVE_MODEL` env var, or `gpt-4o-realtime-preview` |
| `voice` | Azure neural voice name | `en-US-AvaNeural` |
| `modalities` | Output modalities, `["text"]` for text-only responses | `["text", "audio"]` |
| `input_audio_transcription` | Audio transcription config (model, language), `None` to disable | `azure-speech` for non-multimodal models (e.g. `gpt-4.1`) and `phi4-mm-realtime`, `whisper-1` for the other models, with auto-detect |
| `turn_detection` | VAD configuration object | `ServerVad(threshold=0.5)` |
| `tool_choice` | Function calling mode ("auto", "none", etc.) | "auto" |
| `temperature` | Sampling temperature | `0.8` |
| `max_output_tokens` | Maximum output tokens per response | `4096` |
| `save_audio_per_turn` | Save each completed response's audio as a WAV file in `audio_debug` | `False` |

### Turn Detection Options

| VAD Type | Description |
|----------|-------------|
| `AzureSemanticVad` | Default semantic VAD (multilingual) |
| `AzureSemanticVadEn` | English-only, optimized for English |
| `AzureSemanticVadMultilingual` | Explicit multilingual support |

### VAD Parameters

| Parameter | Description | Range |
|-----------|-------------|-------|
| `threshold` | Voice activity detection sensitivity | 0.0-1.0 |
| `silence_duration_ms` | Silence duration before turn ends | milliseconds |
| `prefix_padding_ms` | Audio padding before detected speech | milliseconds |
| `speech_duration_ms` | Minimum speech duration to trigger | milliseconds |
| `remove_filler_words` | Remove "um", "uh", etc. | boolean |

### Azure Voice Live Capabilities

- **Low-latency conversations** - Realtime bidirectional streaming
- **Multilingual support** - Works with Azure's multilingual neural voices
- **Function calling** - Built-in tool use for agentic workflows
- **Interruption handling** - Graceful handling of user interruptions

## Pipeline Mode (STT + LLM + TTS)

### Streaming TTS warm-up

`AgentSession` automatically calls `TTS.prewarm()` at startup. This starts opening
an Azure SDK connection without blocking on dummy speech synthesis. When using
TTS directly, call `tts.prewarm()` before you need the first response.

Each active stream owns its synthesizer. Closing a stream prepares one idle
connection for the next response; `await tts.aclose()` closes idle connections
and active streams. Changing voice options invalidates an unused warmed connection.


### Azure Credentials

**Azure Speech Services (for STT/TTS)**

```bash
export AZURE_SPEECH_KEY=<your-speech-key>
export AZURE_SPEECH_REGION=<your-region>  # e.g., eastus, westus2
```

**Azure OpenAI (for LLM)**

```bash
export AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com
export AZURE_OPENAI_API_KEY=<your-openai-key>
export AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
export AZURE_OPENAI_API_VERSION=2024-10-01-preview
```

### Quick Start

For more control over individual components:

```python
from livekit.agents import Agent, AgentSession
from livekit.plugins import azure, openai, silero

session = AgentSession(
    stt=azure.STT(),
    llm=openai.LLM.with_azure(
        model="gpt-4o",
        azure_endpoint="https://your-resource.openai.azure.com",
        api_key="your-api-key",
        api_version="2024-10-01-preview",
    ),
    tts=azure.TTS(voice="en-US-JennyNeural"),
    vad=silero.VAD.load(),
)

await session.start(room=ctx.room, agent=Agent(instructions="You are helpful."))
```

## Voice Selection

Azure provides a wide variety of neural voices. Use multilingual voices for multi-language support:

```python
# Multilingual voices (recommended)
model = azure.realtime.RealtimeModel(
    voice="en-US-AvaMultilingualNeural"
)

# Or use TTS with specific voices
tts = azure.TTS(voice="en-US-JennyNeural")
```

**Popular voices:**
- `en-US-AvaMultilingualNeural` - Multilingual, feminine
- `en-US-AndrewMultilingualNeural` - Multilingual, masculine
- `en-US-JennyNeural` - English, feminine
- `en-US-GuyNeural` - English, masculine

See [Azure Voice Gallery](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=tts) for full list.

### HD Voices

For higher quality audio, use DragonHD voices:

```python
tts = azure.TTS(voice="en-US-Ava:DragonHDLatestNeural")
```

## Resources

- [LiveKit Agents Documentation](https://docs.livekit.io/agents/)
- [LiveKit Azure Integration Guide](https://docs.livekit.io/agents/integrations/azure/)
- [Azure Voice Live Documentation](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/voice-live)
- [Azure Voice Live Supported Models and Regions](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/voice-live#supported-models-and-regions)
- [Azure Speech Service Documentation](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/)
- [Azure OpenAI Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- [Azure Voice Gallery](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=tts)
- [Azure Portal](https://portal.azure.com)
- [OpenAI Plugin (for Azure OpenAI LLM)](https://github.com/livekit/agents/tree/main/livekit-plugins/livekit-plugins-openai)
