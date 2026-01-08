"""
Azure TTS 流式文本输入示例

此示例演示如何使用 Azure TTS 的流式 API 边接收文本边合成语音。
这对于实时对话场景非常有用,可以在 LLM 生成文本的同时开始语音合成,
从而显著降低延迟。
"""

import asyncio
import logging

from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import AgentServer, AutoSubscribe, JobContext, cli
from livekit.plugins import azure

load_dotenv()

logger = logging.getLogger("azure-tts-demo")
logger.setLevel(logging.INFO)

server = AgentServer()


@server.rtc_session()
async def entrypoint(job: JobContext):
    logger.info("starting Azure TTS streaming example agent")

    # 初始化 Azure TTS
    # 必须使用 TextStream API 才能支持流式文本输入
    tts = azure.TTS(
        voice="en-US-JennyNeural",
        sample_rate=24000,
    )

    source = rtc.AudioSource(tts.sample_rate, tts.num_channels)
    track = rtc.LocalAudioTrack.create_audio_track("agent-mic", source)
    options = rtc.TrackPublishOptions()
    options.source = rtc.TrackSource.SOURCE_MICROPHONE

    await job.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_NONE)
    publication = await job.room.local_participant.publish_track(track, options)
    await publication.wait_for_subscription()

    stream = tts.stream()

    async def _playback_task():
        async for audio in stream:
            await source.capture_frame(audio.frame)

    task = asyncio.create_task(_playback_task())

    # 模拟 LLM 逐步生成文本的场景
    text = (
        "Hello, this is a demonstration of Azure TTS "
        "with streaming text input. The audio synthesis "
        "begins as soon as the first text chunk is received, "
        "which significantly reduces latency in real-time conversations."
    )

    # split into chunks to simulate LLM streaming
    text_chunks = [
        "Hello, ",
        "this is a demonstration ",
        "of Azure TTS ",
        "with streaming text input. ",
        "The audio synthesis ",
        "begins as soon as ",
        "the first text chunk ",
        "is received, ",
        "which significantly ",
        "reduces latency ",
        "in real-time conversations.",
    ]

    for i, chunk in enumerate(text_chunks, 1):
        logger.info(f'pushing chunk {i}/{len(text_chunks)}: "{chunk}"')
        stream.push_text(chunk)
        # 模拟 LLM 生成延迟
        await asyncio.sleep(0.2)

    # Mark end of input segment
    stream.flush()
    stream.end_input()
    await asyncio.gather(task)


if __name__ == "__main__":
    cli.run_app(server)