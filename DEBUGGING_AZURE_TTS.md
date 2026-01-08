# Azure TTS 调试指南

## ✅ 问题已解决!

### 根本原因

错误信息: `APIError: no audio frames were pushed for text: Got it! Let me know if there's anything you'd like to test or ask.`

**真正的原因**: 这不是一个bug,而是**用户打断TTS的正常行为**被错误地当作了错误处理。

从日志中发现:
```
Azure TTS synthesis canceled: CancellationReason.CancelledByUser.
Error: Request closed by client (499). USP state: TurnStarted.
Received audio size: 0 bytes.
```

这表明:
1. 用户在TTS开始合成时就打断了(开始说话)
2. 系统正确地取消了TTS合成
3. 但是这个取消被当作错误,导致重试和最终的"no audio frames"错误

### 修复方案

我们修改了错误处理逻辑,正确区分**用户打断**(正常行为)和**真正的错误**:

1. **在SDK回调层面** - 识别用户打断,不将其当作错误
2. **在段落处理层面** - 用户打断时正常返回,不抛出异常
3. **在合成任务层面** - 用户打断时不重试,直接完成

## 原始问题分析 (已解决)

根据代码分析,我识别出以下可能的问题来源:

1. ✅ **用户打断处理不当** - 这是真正的原因!
2. **Azure SDK回调未触发** - `synthesizing_callback` 没有被调用
3. **Azure API认证/配置问题** - API密钥、区域或端点配置错误
4. **Azure SDK内部错误** - SDK失败但没有触发错误回调
5. **文本流处理问题** - 文本没有正确发送到SDK
6. **网络连接问题** - WebSocket连接失败或超时
7. **音频格式不匹配** - SDK配置的音频格式与预期不符

## 已添加的诊断日志

我在以下关键位置添加了详细的日志:

### 1. SDK线程 (`_run_sdk_synthesis`)
- `[SDK_THREAD]` - SDK合成线程的启动和状态
- `[SDK_CALLBACK]` - 所有SDK回调的触发情况

### 2. 文本输入 (`_stream_text_input`)
- `[TEXT_INPUT]` - 文本块的接收和发送情况

### 3. 音频接收 (`_receive_audio`)
- `[AUDIO_RX]` - 音频块的接收和推送情况
- **关键**: 如果没有收到音频块,会记录错误日志

### 4. 段落协调 (`_synthesize_segment`)
- `[SEGMENT]` - 整体合成流程的协调

### 5. Synthesizer创建和初始化
- `[CREATE_SYNTH]` - synthesizer的创建过程
- `[WARMUP]` - warmup过程的详细信息

## 如何使用这些日志进行诊断

### 步骤1: 运行应用并复现错误

```bash
cd examples
python azure_agent.py dev
```

### 步骤2: 查看日志输出

当错误发生时,查找以下关键日志序列:

#### 正常流程应该看到:
```
[SEGMENT] starting synthesis orchestration for segment xxx
[SDK_THREAD] Starting SDK synthesis thread
[SDK_THREAD] connecting event handlers
[TEXT_INPUT] starting text input streaming
[AUDIO_RX] starting audio reception
[TEXT_INPUT] received chunk 1: 'Got it! ...'
[SDK_CALLBACK] synthesizing_callback triggered  <-- 关键!
[SDK_CALLBACK] received audio chunk: XXX bytes
[AUDIO_RX] received audio chunk 1: XXX bytes
```

#### 如果出现问题,可能看到:
```
[SEGMENT] starting synthesis orchestration
[SDK_THREAD] Starting SDK synthesis thread
[TEXT_INPUT] received chunk 1: 'Got it! ...'
[SDK_THREAD] speak_async called
[SDK_THREAD] synthesis completed with reason: XXX
[AUDIO_RX] received completion signal, total chunks: 0  <-- 没有音频!
[AUDIO_RX] ⚠️ NO AUDIO CHUNKS RECEIVED!
```

### 步骤3: 根据日志定位问题

#### 场景A: SDK回调从未触发
如果看不到 `[SDK_CALLBACK] synthesizing_callback triggered`:
- **原因**: Azure SDK没有调用回调函数
- **可能问题**: 
  - 认证失败 (检查API密钥和区域)
  - 网络连接问题
  - SDK内部错误

#### 场景B: SDK回调触发但没有音频数据
如果看到 `[SDK_CALLBACK] synthesizing event has no audio_data`:
- **原因**: SDK触发了回调但没有提供音频数据
- **可能问题**: 音频格式配置问题

#### 场景C: SDK返回错误
如果看到 `[SDK_CALLBACK] canceled_callback triggered`:
- **原因**: SDK明确报告了错误
- **查看**: cancellation reason和error details

#### 场景D: 文本没有发送
如果看不到 `[TEXT_INPUT] received chunk`:
- **原因**: 文本流没有正确传递
- **可能问题**: 上游文本生成问题

## 下一步诊断建议

### 1. 确认问题场景
请运行应用并提供完整的日志输出,特别是包含以下标签的日志:
- `[SEGMENT]`
- `[SDK_THREAD]`
- `[SDK_CALLBACK]`
- `[TEXT_INPUT]`
- `[AUDIO_RX]`
- `[CREATE_SYNTH]`
- `[WARMUP]`

### 2. 检查配置
确认以下环境变量正确设置:
```bash
AZURE_SPEECH_KEY=your_key
AZURE_SPEECH_REGION=your_region  # 例如: eastus
AZURE_SPEECH_VOICE=your_voice    # 例如: en-US-JennyNeural
```

### 3. 测试基本连接
运行测试脚本验证Azure连接:
```bash
python tests/test_azure_tts_debug.py
```

### 4. 根据日志输出采取行动

一旦我们看到详细的日志输出,我们就能准确定位问题是在:
- Azure SDK层面 (认证、网络、配置)
- 回调处理层面 (事件未触发)
- 音频数据层面 (格式、编码)
- 文本流层面 (输入处理)

## 修复内容

### 1. 改进的SDK回调处理 (tts.py:626-651)

```python
def canceled_callback(evt):
    """Called when synthesis is canceled or fails."""
    cancellation = evt.result.cancellation_details
    error_details = cancellation.error_details if cancellation.error_details else ""
    
    # Check if this is a user interruption (client closed request)
    is_user_interruption = (
        "499" in error_details or
        "closed by client" in error_details.lower() or
        cancelled[0]  # We set this flag when user interrupts
    )
    
    if is_user_interruption:
        logger.info(f"[SDK_CALLBACK] synthesis cancelled due to user interruption (this is normal)")
        # Don't treat user interruption as an error
        asyncio.run_coroutine_threadsafe(audio_queue.put(None), loop)
    else:
        # This is a real error
        logger.error(f"[SDK_CALLBACK] synthesis canceled with error")
        error = APIStatusError(...)
        synthesis_error.append(error)
```

### 2. 改进的段落错误处理 (tts.py:875-920)

```python
except Exception as e:
    # Check if this is a user interruption error
    error_str = str(e).lower()
    is_user_interruption = (
        "499" in error_str or
        "closed by client" in error_str or
        "cancelledbyuser" in error_str.replace(" ", "").replace("_", "")
    )
    
    if is_user_interruption:
        # User interruption is normal, not an error
        logger.info(f"[SEGMENT] segment cancelled due to user interruption (normal behavior)")
        # Clean up and return normally - don't raise error
        return
    else:
        # This is a real error - handle it
        raise APIConnectionError(...)
```

### 3. 改进的合成任务处理 (tts.py:557-580)

```python
try:
    await self._synthesize_segment(output_emitter)
except (APIStatusError, APIConnectionError) as e:
    error_str = str(e).lower()
    is_user_interruption = (
        "499" in error_str and "closed by client" in error_str and
        ("cancelledbyuser" in error_str.replace(" ", "").replace("_", "") or
         "usp state: turnstarted" in error_str)
    )
    
    if is_user_interruption:
        # User interruption is normal - don't retry
        logger.info(f"[SYNTHESIS] synthesis cancelled due to user interruption (normal behavior), not retrying")
        return  # Exit normally without raising
    
    # Only retry on real connection errors
    raise
```

## 测试修复

运行应用并尝试在TTS说话时打断它:

```bash
cd examples
python azure_agent.py dev
```

现在应该看到:
- 用户打断时,日志显示 `[SYNTHESIS] synthesis cancelled due to user interruption (normal behavior)`
- 不再有错误重试
- 不再有"no audio frames were pushed"错误
- 系统优雅地处理打断,继续正常运行

## 诊断日志说明

修复后保留了详细的诊断日志,帮助未来调试:
- `[SDK_THREAD]` - SDK合成线程状态
- `[SDK_CALLBACK]` - SDK回调触发情况
- `[TEXT_INPUT]` - 文本输入流
- `[AUDIO_RX]` - 音频接收流
- `[SEGMENT]` - 段落协调
- `[CREATE_SYNTH]` - Synthesizer创建
- `[WARMUP]` - Warmup过程

这些日志在遇到真正的错误时仍然有用。