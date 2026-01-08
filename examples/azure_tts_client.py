"""
Azure TTS 客户端示例

这个脚本展示如何连接到 LiveKit 房间并收听 Azure TTS 服务生成的音频。

使用方法:
1. 确保 azure_tts.py 服务器正在运行
2. 设置环境变量:
   - LIVEKIT_URL: LiveKit 服务器地址
   - LIVEKIT_API_KEY: LiveKit API 密钥
   - LIVEKIT_API_SECRET: LiveKit API 密钥
3. 运行此脚本: python azure_tts_client.py
"""

import asyncio
import logging
import os
import numpy as np
from dotenv import load_dotenv
from livekit import rtc
from livekit import api

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False
    print("⚠️  警告: sounddevice 未安装，无法播放音频")
    print("💡 请安装: pip install sounddevice")

load_dotenv()

# 配置日志输出到控制台
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("azure-tts-client")

# LiveKit 配置
LIVEKIT_URL = os.environ.get("LIVEKIT_URL")
LIVEKIT_API_KEY = os.environ.get("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET")

# 检查环境变量并给出清晰的错误信息
if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
    print("=" * 60)
    print("❌ 错误: 缺少必需的环境变量")
    print("=" * 60)
    print("\n请设置以下环境变量:")
    print(f"  LIVEKIT_URL: {'✅ 已设置' if LIVEKIT_URL else '❌ 未设置'}")
    print(f"  LIVEKIT_API_KEY: {'✅ 已设置' if LIVEKIT_API_KEY else '❌ 未设置'}")
    print(f"  LIVEKIT_API_SECRET: {'✅ 已设置' if LIVEKIT_API_SECRET else '❌ 未设置'}")
    print("\n💡 本地开发模式配置示例 (.env 文件):")
    print("  LIVEKIT_URL=ws://127.0.0.1:7880")
    print("  LIVEKIT_API_KEY=devkey")
    print("  LIVEKIT_API_SECRET=secret")
    print("\n或者运行检查脚本: python check_config.py")
    print("=" * 60)
    raise ValueError(
        "请设置环境变量: LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET"
    )


async def main():
    """连接到 LiveKit 房间并收听 TTS 音频"""
    
    logger.info("=" * 60)
    logger.info("Azure TTS 客户端启动")
    logger.info("=" * 60)
    
    # 创建房间名称
    room_name ="mock_room" # "azure-tts-demo"
    logger.info(f"目标房间: {room_name}")
    
    # 验证并规范化 URL
    url = LIVEKIT_URL.strip()
    logger.info(f"原始 URL: {url}")
    if not url.startswith(("ws://", "wss://")):
        logger.error(f"❌ LIVEKIT_URL 必须以 ws:// 或 wss:// 开头，当前值: {url}")
        logger.info("💡 本地开发模式应使用: ws://127.0.0.1:7880")
        logger.info("💡 生产环境应使用: wss://your-server.com")
        print("\n❌ 连接失败: URL 格式错误")
        return
    
    # 确保 URL 格式正确（移除尾部斜杠和路径）
    if url.endswith("/"):
        url = url.rstrip("/")
    # 如果 URL 包含路径（除了协议和主机），移除它
    from urllib.parse import urlparse
    parsed = urlparse(url)
    if parsed.path and parsed.path != "/":
        logger.warning(f"URL 包含路径 {parsed.path}，将使用基础 URL")
        url = f"{parsed.scheme}://{parsed.netloc}"
    
    logger.info(f"使用 LiveKit URL: {url}")
    
    # 创建访问令牌
    try:
        token = api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET) \
            .with_identity("tts-client") \
            .with_name("TTS Client") \
            .with_grants(api.VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
            )).to_jwt()
    except Exception as e:
        logger.error(f"创建访问令牌失败: {e}")
        return
    
    # 创建房间并连接
    room = rtc.Room()
    
    # 音频播放相关变量
    audio_stream = None
    audio_output_stream = None
    audio_queue = asyncio.Queue()
    audio_playback_task = None
    
    async def play_audio_frames():
        """从队列中读取音频帧并播放到系统扬声器"""
        nonlocal audio_output_stream
        if not HAS_SOUNDDEVICE:
            logger.warning("sounddevice 未安装，无法播放音频")
            return
        
        try:
            # 获取第一个音频帧来确定采样率和通道数
            first_frame = await audio_queue.get()
            if first_frame is None:  # 如果立即收到结束标记
                return
                
            sample_rate = first_frame.sample_rate
            num_channels = first_frame.num_channels
            
            logger.info(f"初始化音频播放: 采样率={sample_rate}Hz, 通道数={num_channels}")
            
            # 创建音频输出流
            audio_output_stream = sd.OutputStream(
                samplerate=sample_rate,
                channels=num_channels,
                dtype='int16',
                blocksize=sample_rate // 20,  # 50ms blocks
            )
            audio_output_stream.start()
            logger.info("✅ 音频播放已启动")
            
            # 播放第一个帧
            audio_data = np.frombuffer(first_frame.data, dtype=np.int16)
            if num_channels == 1:
                audio_data = audio_data.reshape(-1, 1)
            else:
                audio_data = audio_data.reshape(-1, num_channels)
            audio_output_stream.write(audio_data)
            
            # 继续播放后续帧
            while True:
                try:
                    frame = await asyncio.wait_for(audio_queue.get(), timeout=1.0)
                    if frame is None:  # 结束标记
                        break
                    audio_data = np.frombuffer(frame.data, dtype=np.int16)
                    if num_channels == 1:
                        audio_data = audio_data.reshape(-1, 1)
                    else:
                        audio_data = audio_data.reshape(-1, num_channels)
                    audio_output_stream.write(audio_data)
                except asyncio.TimeoutError:
                    # 检查是否还有音频流
                    nonlocal audio_stream
                    if audio_stream is None:
                        break
                    continue
        except Exception as e:
            logger.error(f"音频播放错误: {e}")
        finally:
            if audio_output_stream:
                audio_output_stream.stop()
                audio_output_stream.close()
                logger.info("音频播放已停止")
    
    @room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        """当订阅到音频轨道时调用"""
        logger.info(f"已订阅轨道: {track.kind} from {participant.identity}")
        
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            logger.info("开始接收音频流...")
            
            if not HAS_SOUNDDEVICE:
                logger.warning("⚠️  sounddevice 未安装，无法播放音频")
                logger.info("💡 请安装: pip install sounddevice")
                return
            
            # 创建音频流并开始播放任务
            nonlocal audio_stream, audio_playback_task
            try:
                audio_stream = rtc.AudioStream.from_track(track=track)
                logger.info("✅ 音频流已创建")
                
                # 启动音频播放任务
                if audio_playback_task is None:
                    audio_playback_task = asyncio.create_task(play_audio_frames())
                    logger.info("✅ 音频播放任务已启动")
                
                # 启动音频帧接收任务
                async def receive_audio_frames():
                    try:
                        async for audio_event in audio_stream:
                            await audio_queue.put(audio_event.frame)
                    except Exception as e:
                        logger.error(f"接收音频帧错误: {e}")
                    finally:
                        # 发送结束标记
                        await audio_queue.put(None)
                
                asyncio.create_task(receive_audio_frames())
                logger.info("✅ 开始接收并播放音频")
            except Exception as e:
                logger.error(f"创建音频流失败: {e}")
    
    @room.on("track_unsubscribed")
    def on_track_unsubscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        """取消订阅轨道时调用"""
        logger.info(f"已取消订阅轨道: {track.kind} from {participant.identity}")
    
    @room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant):
        """参与者连接时调用"""
        logger.info(f"参与者已连接: {participant.identity}")
    
    @room.on("participant_disconnected")
    def on_participant_disconnected(participant: rtc.RemoteParticipant):
        """参与者断开连接时调用"""
        logger.info(f"参与者已断开: {participant.identity}")
    
    # 尝试先通过 API 创建房间（可选，但可能有助于解决连接问题）
    try:
        # 将 ws:// 转换为 http:// 用于 API 调用
        api_url = url.replace("ws://", "http://").replace("wss://", "https://")
        async with api.LiveKitAPI(api_url, LIVEKIT_API_KEY, LIVEKIT_API_SECRET) as lk_api:
            try:
                room_info = await lk_api.room.create_room(api.CreateRoomRequest(name=room_name))
                logger.info(f"房间已创建或已存在: {room_name}")
            except Exception as e:
                logger.debug(f"创建房间时出现错误（可能已存在）: {e}")
    except Exception as e:
        logger.warning(f"无法通过 API 创建房间（将尝试直接连接）: {e}")
    
    # 连接到房间
    logger.info(f"正在连接到房间: {room_name}")
    logger.info(f"使用 URL: {url}")
    try:
        await room.connect(url, token)
        logger.info("✅ 已成功连接到房间")
    except Exception as e:
        logger.error(f"❌ 连接失败: {e}")
        print("\n" + "=" * 60)
        print("❌ 连接失败")
        print("=" * 60)
        logger.error(f"错误详情: {type(e).__name__}: {e}")
        print("\n💡 故障排除建议:")
        print("1. 确认 LiveKit 服务器正在运行:")
        print("   livekit-server --dev")
        print("2. 检查 LIVEKIT_URL 是否正确:")
        print("   - 本地开发: ws://127.0.0.1:7880")
        print("   - 生产环境: wss://your-server.com")
        print("3. 检查 LIVEKIT_API_KEY 和 LIVEKIT_API_SECRET 是否正确")
        print("   开发模式默认: API_KEY=devkey, API_SECRET=secret")
        print("4. 确认服务器端 (azure_tts.py) 正在运行:")
        print("   python azure_tts.py dev")
        print("5. 尝试运行配置检查: python check_config.py")
        print("6. 检查服务器是否可访问:")
        print("   curl http://127.0.0.1:7880/")
        print("=" * 60)
        return
    
    # 检查是否有参与者（服务器端）
    participants = list(room.remote_participants.values())
    if not participants:
        logger.warning("⚠️  房间中没有其他参与者")
        logger.warning("⚠️  请确保服务器端 (azure_tts.py) 正在运行并已连接到房间")
        logger.info("等待 30 秒以查看是否有参与者加入...")
        await asyncio.sleep(30)
        participants = list(room.remote_participants.values())
        if not participants:
            logger.error("❌ 仍然没有参与者，可能服务器端未运行")
            print("\n❌ 未检测到服务器端连接")
            print("请确保在另一个终端运行: python azure_tts.py dev")
            await room.disconnect()
            return
    
    # 等待一段时间以接收音频
    logger.info("=" * 60)
    logger.info("等待接收 TTS 音频... (30秒)")
    logger.info("服务器会自动播放几个示例文本")
    logger.info("按 Ctrl+C 可以提前退出")
    logger.info("=" * 60)
    
    try:
        # 等待 30 秒以接收完整的演示音频
        await asyncio.sleep(30)
    except KeyboardInterrupt:
        logger.info("用户中断")
    finally:
        # 停止音频播放
        if audio_stream:
            try:
                await audio_stream.aclose()
            except Exception:
                pass
            audio_stream = None
        
        # 发送结束标记到队列
        try:
            await audio_queue.put(None)
        except Exception:
            pass
        
        if audio_playback_task and not audio_playback_task.done():
            await asyncio.sleep(0.5)  # 等待播放完成
            if not audio_playback_task.done():
                audio_playback_task.cancel()
                try:
                    await audio_playback_task
                except asyncio.CancelledError:
                    pass
        
        if audio_output_stream:
            try:
                audio_output_stream.stop()
                audio_output_stream.close()
            except Exception:
                pass
        
        # 断开连接
        logger.info("正在断开连接...")
        await room.disconnect()
        logger.info("✅ 已断开连接")
        logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

