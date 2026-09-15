# Azure plugin for LiveKit Agents

Support for Azure AI including Azure Speech. For Azure OpenAI, see the [OpenAI plugin](https://github.com/livekit/agents/tree/main/livekit-plugins/livekit-plugins-openai).

See [https://docs.livekit.io/agents/integrations/azure/](https://docs.livekit.io/agents/integrations/azure/) for more information.

## Installation

```bash
pip install livekit-plugins-azure
```

## Pre-requisites

You'll need to specify an Azure Speech Key and a Deployment Region. They can be set as environment variables: `AZURE_SPEECH_KEY` and `AZURE_SPEECH_REGION`, respectively.

## Streaming TTS warm-up

`AgentSession` automatically calls `TTS.prewarm()` at startup. This starts opening
an Azure SDK connection without blocking on dummy speech synthesis. When using
TTS directly, call `tts.prewarm()` before you need the first response.

Each active stream owns its synthesizer. Closing a stream prepares one idle
connection for the next response; `await tts.aclose()` closes idle connections
and active streams. Changing voice options invalidates an unused warmed connection.
