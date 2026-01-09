# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import os
import queue
import weakref
from dataclasses import dataclass, replace
from typing import Literal

import aiohttp

try:
    import azure.cognitiveservices.speech as speechsdk
except ImportError:
    raise ImportError(
        "azure-cognitiveservices-speech is required for streaming. "
        "Install with: pip install azure-cognitiveservices-speech"
    )

from livekit.agents import APIConnectionError, APIStatusError, APITimeoutError, tokenize, tts, utils
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from .log import logger

SUPPORTED_OUTPUT_FORMATS = {
    8000: "raw-8khz-16bit-mono-pcm",
    16000: "raw-16khz-16bit-mono-pcm",
    22050: "raw-22050hz-16bit-mono-pcm",
    24000: "raw-24khz-16bit-mono-pcm",
    44100: "raw-44100hz-16bit-mono-pcm",
    48000: "raw-48khz-16bit-mono-pcm",
}

# Azure SDK output format mapping - use Raw formats to avoid WAV headers
SDK_OUTPUT_FORMATS = {
    8000: speechsdk.SpeechSynthesisOutputFormat.Raw8Khz16BitMonoPcm,
    16000: speechsdk.SpeechSynthesisOutputFormat.Raw16Khz16BitMonoPcm,
    24000: speechsdk.SpeechSynthesisOutputFormat.Raw24Khz16BitMonoPcm,
    48000: speechsdk.SpeechSynthesisOutputFormat.Raw48Khz16BitMonoPcm,
}


@dataclass
class ProsodyConfig:
    rate: Literal["x-slow", "slow", "medium", "fast", "x-fast"] | float | None = None
    volume: Literal["silent", "x-soft", "soft", "medium", "loud", "x-loud"] | float | None = None
    pitch: Literal["x-low", "low", "medium", "high", "x-high"] | None = None

    def validate(self) -> None:
        if self.rate:
            if isinstance(self.rate, float) and not 0.5 <= self.rate <= 2:
                raise ValueError("Prosody rate must be between 0.5 and 2")
            if isinstance(self.rate, str) and self.rate not in [
                "x-slow",
                "slow",
                "medium",
                "fast",
                "x-fast",
            ]:
                raise ValueError(
                    "Prosody rate must be one of 'x-slow', 'slow', 'medium', 'fast', 'x-fast'"
                )
        if self.volume:
            if isinstance(self.volume, float) and not 0 <= self.volume <= 100:
                raise ValueError("Prosody volume must be between 0 and 100")
            if isinstance(self.volume, str) and self.volume not in [
                "silent",
                "x-soft",
                "soft",
                "medium",
                "loud",
                "x-loud",
            ]:
                raise ValueError(
                    "Prosody volume must be one of 'silent', 'x-soft', 'soft', 'medium', 'loud', 'x-loud'"  # noqa: E501
                )
        if self.pitch and self.pitch not in [
            "x-low",
            "low",
            "medium",
            "high",
            "x-high",
        ]:
            raise ValueError(
                "Prosody pitch must be one of 'x-low', 'low', 'medium', 'high', 'x-high'"
            )

    def __post_init__(self) -> None:
        self.validate()


@dataclass
class StyleConfig:
    style: str
    degree: float | None = None

    def validate(self) -> None:
        if self.degree is not None and not 0.1 <= self.degree <= 2.0:
            raise ValueError("Style degree must be between 0.1 and 2.0")

    def __post_init__(self) -> None:
        self.validate()


@dataclass
class _TTSOptions:
    sample_rate: int
    subscription_key: str | None
    region: str | None
    voice: str
    language: str | None
    speech_endpoint: str | None
    deployment_id: str | None
    prosody: NotGivenOr[ProsodyConfig]
    style: NotGivenOr[StyleConfig]
    auth_token: str | None = None

    def get_endpoint_url(self) -> str:
        base = (
            self.speech_endpoint
            or f"https://{self.region}.tts.speech.microsoft.com/cognitiveservices/v1"
        )
        if self.deployment_id:
            return f"{base}?deploymentId={self.deployment_id}"
        return base


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice: str = "en-US-JennyNeural",
        language: str | None = None,
        sample_rate: int = 24000,
        prosody: NotGivenOr[ProsodyConfig] = NOT_GIVEN,
        style: NotGivenOr[StyleConfig] = NOT_GIVEN,
        speech_key: str | None = None,
        speech_region: str | None = None,
        speech_endpoint: str | None = None,
        deployment_id: str | None = None,
        speech_auth_token: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=1,
        )
        if sample_rate not in SUPPORTED_OUTPUT_FORMATS:
            raise ValueError(
                f"Unsupported sample rate {sample_rate}. Supported: {list(SUPPORTED_OUTPUT_FORMATS)}"  # noqa: E501
            )

        if not speech_key:
            speech_key = os.environ.get("AZURE_SPEECH_KEY")

        if not speech_region:
            speech_region = os.environ.get("AZURE_SPEECH_REGION")

        if not speech_endpoint:
            speech_endpoint = os.environ.get("AZURE_SPEECH_ENDPOINT")

        has_endpoint = bool(speech_endpoint)
        has_key_and_region = bool(speech_key and speech_region)
        has_token_and_region = bool(speech_auth_token and speech_region)
        if not (has_endpoint or has_key_and_region or has_token_and_region):
            raise ValueError(
                "Authentication requires one of: speech_endpoint (AZURE_SPEECH_ENDPOINT), "
                "speech_key & speech_region (AZURE_SPEECH_KEY & AZURE_SPEECH_REGION), "
                "or speech_auth_token & speech_region."
            )

        if is_given(prosody):
            prosody.validate()
        if is_given(style):
            style.validate()

        self._session = http_session
        self._opts = _TTSOptions(
            sample_rate=sample_rate,
            subscription_key=speech_key,
            region=speech_region,
            speech_endpoint=speech_endpoint,
            voice=voice,
            deployment_id=deployment_id,
            language=language,
            prosody=prosody,
            style=style,
            auth_token=speech_auth_token,
        )
        self._streams = weakref.WeakSet[SynthesizeStream]()
        # Shared synthesizer and warmup state across all streams
        self._synthesizer: speechsdk.SpeechSynthesizer | None = None
        self._warmup_done = False

    @property
    def model(self) -> str:
        return "unknown"

    @property
    def provider(self) -> str:
        return "Azure TTS"

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        prosody: NotGivenOr[ProsodyConfig] = NOT_GIVEN,
        style: NotGivenOr[StyleConfig] = NOT_GIVEN,
    ) -> None:
        if is_given(voice):
            self._opts.voice = voice
        if is_given(language):
            self._opts.language = language
        if is_given(prosody):
            prosody.validate()
            self._opts.prosody = prosody
        if is_given(style):
            style.validate()
            self._opts.style = style

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()


class ChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    def _build_ssml(self) -> str:
        lang = self._opts.language or "en-US"
        ssml = (
            f'<speak version="1.0" '
            f'xmlns="http://www.w3.org/2001/10/synthesis" '
            f'xmlns:mstts="http://www.w3.org/2001/mstts" '
            f'xml:lang="{lang}">'
        )
        ssml += f'<voice name="{self._opts.voice}">'
        if is_given(self._opts.style):
            degree = f' styledegree="{self._opts.style.degree}"' if self._opts.style.degree else ""
            ssml += f'<mstts:express-as style="{self._opts.style.style}"{degree}>'

        if is_given(self._opts.prosody):
            p = self._opts.prosody

            rate_attr = f' rate="{p.rate}"' if p.rate is not None else ""
            vol_attr = f' volume="{p.volume}"' if p.volume is not None else ""
            pitch_attr = f' pitch="{p.pitch}"' if p.pitch is not None else ""
            ssml += f"<prosody{rate_attr}{vol_attr}{pitch_attr}>{self.input_text}</prosody>"
        else:
            ssml += self.input_text

        if is_given(self._opts.style):
            ssml += "</mstts:express-as>"

        ssml += "</voice></speak>"
        return ssml

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        headers = {
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": SUPPORTED_OUTPUT_FORMATS[self._opts.sample_rate],
            "User-Agent": "LiveKit Agents",
        }
        if self._opts.auth_token:
            headers["Authorization"] = f"Bearer {self._opts.auth_token}"
        elif self._opts.subscription_key:
            headers["Ocp-Apim-Subscription-Key"] = self._opts.subscription_key

        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
        )

        try:
            ssml_content = self._build_ssml()
            logger.debug(f"[Azure TTS] SSML length: {len(ssml_content)} bytes")
            
            async with self._tts._ensure_session().post(
                url=self._opts.get_endpoint_url(),
                headers=headers,
                data=ssml_content,
                timeout=aiohttp.ClientTimeout(total=30, sock_connect=self._conn_options.timeout),
            ) as resp:
                logger.debug(f"[Azure TTS] Response status: {resp.status}")
                logger.debug(f"[Azure TTS] Response headers: {dict(resp.headers)}")
                
                if resp.status != 200:
                    response_text = await resp.text()
                    logger.error(f"[Azure TTS] Error response body: {response_text}")
                
                resp.raise_for_status()
                async for data, _ in resp.content.iter_chunks():
                    output_emitter.push(data)

        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            logger.error(f"[Azure TTS] HTTP error: {e.status} - {e.message}")
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=None,
                body=None,
            ) from None
        except Exception as e:
            raise APIConnectionError(str(e)) from e


class SynthesizeStream(tts.SynthesizeStream):
    """Streaming TTS using Azure Speech SDK with TextStream input."""

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)
        self._text_ch = utils.aio.Chan[str]()
        # Use shared synthesizer from TTS instance
        if self._tts._synthesizer is None:
            self._tts._synthesizer = self._create_synthesizer()
            # Warm up only once
            self._warmup_synthesizer()

    def _recreate_synthesizer(self) -> None:
        """Recreate the synthesizer after connection issues."""
        logger.info("recreating azure tts synthesizer after connection error")
        try:
            if self._tts._synthesizer:
                # Clean up old synthesizer
                del self._tts._synthesizer
        except:
            pass
        self._tts._synthesizer = self._create_synthesizer()
        # DO NOT reset warmup_done flag - we don't want to warmup again during retry
        # The warmup is only needed once per TTS instance, not per synthesizer
        # Resetting it here causes segment state issues when warmup is called during retry
        logger.debug("[RECREATE] synthesizer recreated, skipping warmup to avoid segment state issues")
 
    def _create_synthesizer(self) -> speechsdk.SpeechSynthesizer:
        """Create and configure the Azure Speech synthesizer."""
        logger.info("[CREATE_SYNTH] creating new Azure Speech synthesizer")
        
        # Build WebSocket v2 endpoint
        if self._opts.speech_endpoint:
            endpoint = self._opts.speech_endpoint.replace(
                "/cognitiveservices/v1", "/cognitiveservices/websocket/v2"
            )
            logger.info(f"[CREATE_SYNTH] using custom endpoint: {endpoint}")
        else:
            endpoint = f"wss://{self._opts.region}.tts.speech.microsoft.com/cognitiveservices/websocket/v2"
            logger.info(f"[CREATE_SYNTH] using default endpoint for region: {self._opts.region}")

        # Create speech config
        logger.debug("[CREATE_SYNTH] creating speech config")
        speech_config = speechsdk.SpeechConfig(
            endpoint=endpoint,
            subscription=self._opts.subscription_key or "",
        )

        # Set deployment ID if provided
        if self._opts.deployment_id:
            logger.info(f"[CREATE_SYNTH] setting deployment ID: {self._opts.deployment_id}")
            speech_config.endpoint_id = self._opts.deployment_id

        # Set voice and output format
        logger.info(f"[CREATE_SYNTH] setting voice: {self._opts.voice}")
        speech_config.speech_synthesis_voice_name = self._opts.voice
        
        # Use SDK format if available
        if self._opts.sample_rate in SDK_OUTPUT_FORMATS:
            format_name = SDK_OUTPUT_FORMATS[self._opts.sample_rate]
            logger.info(f"[CREATE_SYNTH] setting output format for {self._opts.sample_rate}Hz: {format_name}")
            speech_config.set_speech_synthesis_output_format(format_name)
        else:
            logger.info(f"[CREATE_SYNTH] using default 24kHz format (requested: {self._opts.sample_rate}Hz)")
            # Default to 24kHz raw format
            speech_config.set_speech_synthesis_output_format(
                speechsdk.SpeechSynthesisOutputFormat.Raw24Khz16BitMonoPcm
            )

        # Create synthesizer (no audio config - we'll use events)
        logger.info("[CREATE_SYNTH] creating synthesizer instance")
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config, audio_config=None
        )
        logger.info("[CREATE_SYNTH] synthesizer created successfully")
        return synthesizer

    def _warmup_synthesizer(self) -> None:
        """Warm up the synthesizer by synthesizing a short text."""
        if self._tts._warmup_done:
            logger.debug("azure tts synthesizer already warmed up, skipping")
            return

        import time

        # Warm-up: synthesize a short text first to establish connection
        logger.info("warming up azure tts synthesizer - starting warmup process")
        warmup_start = time.time()
        
        try:
            logger.debug("[WARMUP] creating warmup synthesis request")
            warmup_request = speechsdk.SpeechSynthesisRequest(
                input_type=speechsdk.SpeechSynthesisRequestInputType.TextStream
            )
            
            logger.debug("[WARMUP] starting async warmup synthesis")
            warmup_task = self._tts._synthesizer.speak_async(warmup_request)
            
            logger.debug("[WARMUP] writing warmup text to input stream")
            warmup_request.input_stream.write("Warm up.")
            warmup_request.input_stream.close()
            
            logger.debug("[WARMUP] waiting for warmup synthesis result (this may take several seconds)")
            warmup_result = warmup_task.get()
            
            warmup_time = time.time() - warmup_start
            
            if warmup_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                logger.info(
                    "[WARMUP] azure tts warmup completed successfully",
                    extra={
                        "duration": f"{warmup_time:.3f}s",
                        "audio_length": len(warmup_result.audio_data) if warmup_result.audio_data else 0
                    }
                )
            else:
                logger.warning(
                    "[WARMUP] azure tts warmup completed with unexpected reason",
                    extra={
                        "duration": f"{warmup_time:.3f}s",
                        "reason": warmup_result.reason,
                        "cancellation_details": warmup_result.cancellation_details if hasattr(warmup_result, 'cancellation_details') else None
                    }
                )

            logger.debug("[WARMUP] warmup completed, setting warmup_done flag")
            self._tts._warmup_done = True
            
        except Exception as e:
            warmup_time = time.time() - warmup_start
            logger.error(
                "azure tts warmup failed",
                extra={
                    "duration": f"{warmup_time:.3f}s",
                    "error": str(e),
                    "error_type": type(e).__name__
                },
                exc_info=True
            )
            raise

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )

        async def _forward_input() -> None:
            """Forward text chunks directly to synthesis."""
            async for input in self._input_ch:
                if isinstance(input, str):
                    logger.debug("[FORWARD_INPUT] enqueue text chunk", extra={"len": len(input)})
                    self._text_ch.send_nowait(input)
                elif isinstance(input, self._FlushSentinel):
                    logger.debug("[FORWARD_INPUT] flush sentinel received -> segment boundary")
                    # Flush marks segment boundary
                    self._text_ch.send_nowait(None)
            logger.debug("[FORWARD_INPUT] closing text channel after input stream end")
            self._text_ch.close()

        async def _run_synthesis() -> None:
            """Run synthesis task - let base framework handle retries."""
            # Check if synthesizer needs recreation due to previous errors
            # This can happen when the base framework retries with a new AudioEmitter
            if self._tts._synthesizer is None:
                logger.debug("[SYNTHESIS] synthesizer is None, recreating")
                self._tts._synthesizer = self._create_synthesizer()
                if not self._tts._warmup_done:
                    self._warmup_synthesizer()
            
            try:
                await self._synthesize_segment(output_emitter)
            except (APIStatusError, APIConnectionError) as e:
                # Check if this is a user interruption (not a real error)
                error_str = str(e).lower()
                is_user_interruption = (
                    "499" in error_str and "closed by client" in error_str and
                    ("cancelledbyuser" in error_str.replace(" ", "").replace("_", "") or
                     "usp state: turnstarted" in error_str)
                )
                
                if is_user_interruption:
                    # User interruption is normal - don't treat as error, don't retry
                    logger.info(f"[SYNTHESIS] synthesis cancelled due to user interruption (normal behavior), not retrying")
                    return  # Exit normally without raising
                
                # On real connection errors, mark synthesizer for recreation on next attempt
                # The base framework will retry with a new AudioEmitter
                if ("timeout" in error_str or "usp error" in error_str or
                    ("499" in error_str and "closed by client" not in error_str)):
                    logger.warning(f"[SYNTHESIS] connection error, will recreate synthesizer on next retry: {e}")
                    # Clean up synthesizer so it gets recreated on next attempt
                    try:
                        if self._tts._synthesizer:
                            del self._tts._synthesizer
                            self._tts._synthesizer = None
                    except:
                        pass
                raise  # Let base framework handle retry

        tasks = [
            asyncio.create_task(_forward_input()),
            asyncio.create_task(_run_synthesis()),
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except (APIStatusError, APIConnectionError):
            raise  # Don't wrap these errors again
        except Exception as e:
            raise APIConnectionError(str(e)) from e
        finally:
            await utils.aio.gracefully_cancel(*tasks)

    async def _synthesize_segment(
        self, output_emitter: tts.AudioEmitter
    ) -> None:
        """Synthesize using Azure SDK with streaming text and audio."""
        import time
        
        # Check connection health before synthesis
        # if not self._check_connection_health():
        #     logger.info("recreating stale azure tts connection")
        #     self._recreate_synthesizer()
        
        segment_id = utils.shortuuid()
        logger.debug(f"[SEGMENT] starting new segment: {segment_id}")
        output_emitter.start_segment(segment_id=segment_id)
        logger.debug(f"[SEGMENT] segment {segment_id} started successfully")
        
        # Track synthesis start time
        # synthesis_start = time.time()

        # Use asyncio Queue to bridge sync callbacks to async
        audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        synthesis_error: list[Exception] = []
        text_queue: queue.Queue[str | None] = queue.Queue()  # Sync queue for thread-safe access
        cancelled = [False]  # Flag to signal cancellation to SDK thread

        # Get the event loop before entering the thread
        loop = asyncio.get_event_loop()

        def _run_sdk_synthesis() -> None:
            """Run Azure SDK synthesis in sync mode with streaming callbacks."""
            logger.info(f"[SDK_THREAD] Starting SDK synthesis thread for segment {segment_id}")
            
            def synthesizing_callback(evt):
                """Called when audio chunks are available during synthesis."""
                logger.info(f"[SDK_CALLBACK] synthesizing_callback triggered, cancelled={cancelled[0]}")
                if cancelled[0]:
                    logger.debug("[SDK_CALLBACK] discarding audio due to cancellation")
                    return  # Discard audio if cancelled
                if evt.result.audio_data:
                    # Raw PCM format - no headers to strip
                    audio_chunk = evt.result.audio_data
                    logger.info(f"[SDK_CALLBACK] received audio chunk: {len(audio_chunk)} bytes")
                    # Send audio to async queue (thread-safe)
                    asyncio.run_coroutine_threadsafe(audio_queue.put(audio_chunk), loop)
                else:
                    logger.warning("[SDK_CALLBACK] synthesizing event has no audio_data")

            def completed_callback(evt):
                """Called when synthesis completes successfully."""
                # logger.info(f"[SDK_CALLBACK] completed_callback triggered, reason={evt.result.reason}")
                # Signal completion with None
                asyncio.run_coroutine_threadsafe(audio_queue.put(None), loop)

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
                    logger.debug(f"[SDK_CALLBACK] cancellation details: {cancellation.reason}, {error_details}")
                    # Don't treat user interruption as an error - just signal completion
                    asyncio.run_coroutine_threadsafe(audio_queue.put(None), loop)
                else:
                    # This is a real error
                    # logger.error(f"[SDK_CALLBACK] synthesis canceled with error")
                    logger.error(f"[SDK_CALLBACK] cancellation reason: {cancellation.reason}, error: {error_details}")
                    error = APIStatusError(
                        f"Azure TTS synthesis canceled: {cancellation.reason}. "
                        f"Error: {error_details}"
                    )
                    synthesis_error.append(error)
                    # Signal error completion
                    asyncio.run_coroutine_threadsafe(audio_queue.put(None), loop)

            # Connect event handlers
            logger.info("[SDK_THREAD] connecting event handlers to synthesizer")
            self._tts._synthesizer.synthesizing.connect(synthesizing_callback)
            self._tts._synthesizer.synthesis_completed.connect(completed_callback)
            self._tts._synthesizer.synthesis_canceled.connect(canceled_callback)
            logger.info("[SDK_THREAD] event handlers connected successfully")

            # Create streaming request
            logger.info("[SDK_THREAD] creating speech synthesis request with TextStream input")
            tts_request = speechsdk.SpeechSynthesisRequest(
                input_type=speechsdk.SpeechSynthesisRequestInputType.TextStream
            )
            logger.info("[SDK_THREAD] speech synthesis request created")

            try:
                # Start synthesis (returns result future)
                logger.info("[SDK_THREAD] calling speak_async to start synthesis")
                result_future = self._tts._synthesizer.speak_async(tts_request)
                logger.info("[SDK_THREAD] speak_async called, synthesis task started, waiting for text input")

                # Stream text pieces as they arrive from the queue
                chunk_count = [0]
                while True:
                    if cancelled[0]:
                        logger.debug("sdk thread detected cancellation, stopping text streaming")
                        break
                    text_piece = text_queue.get()  # Blocking get in sync thread
                    if text_piece is None:
                        break
                    if cancelled[0]:
                        break
                    chunk_count[0] += 1
                    tts_request.input_stream.write(text_piece)
                    # Log each chunk as it's streamed
                    logger.info("streaming tts chunk", extra={"chunk": chunk_count[0], "text": text_piece})

                # Close input stream to signal completion
                logger.info("[SDK_THREAD] closing input stream to signal end of text")
                tts_request.input_stream.close()
                logger.info("[SDK_THREAD] input stream closed")
                
                # Wait for synthesis to complete
                # This blocks until the SDK finishes and callbacks fire
                logger.info("[SDK_THREAD] waiting for synthesis to complete (blocking call)")
                result = result_future.get()
                logger.info(f"[SDK_THREAD] synthesis completed with reason: {result.reason}")
                
                # Ensure completion signal is sent
                if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                    logger.info("[SDK_THREAD] synthesis completed successfully, sending completion signal")
                    asyncio.run_coroutine_threadsafe(audio_queue.put(None), loop)
                else:
                    logger.warning(f"[SDK_THREAD] unexpected result reason: {result.reason}")
                    # Still send completion signal to avoid hanging
                    asyncio.run_coroutine_threadsafe(audio_queue.put(None), loop)
                
            except Exception as e:
                logger.error(f"[SDK_THREAD] exception in SDK synthesis: {e}", exc_info=True)
                synthesis_error.append(e)
                asyncio.run_coroutine_threadsafe(audio_queue.put(None), loop)

        async def _stream_text_input() -> None:
            """Stream text chunks to the SDK as they arrive."""

            chunk_count = 0
            total_text_length = 0
            async for text_chunk in self._text_ch:
                if text_chunk is None:
                    logger.debug("[TEXT_INPUT] received segment terminator (None)")
                    # End of segment
                    break
                chunk_count += 1
                total_text_length += len(text_chunk)
                self._mark_started()
                # Send to sync thread via sync queue
                logger.debug(f"[TEXT_INPUT] putting chunk {chunk_count} into text_queue", extra={"len": len(text_chunk)})
                text_queue.put(text_chunk)
                logger.debug(f"[TEXT_INPUT] chunk {chunk_count} queued successfully")
            # Signal end of text
            logger.debug("[TEXT_INPUT] sending None to text_queue to close segment")
            text_queue.put(None)
  
        async def _receive_audio() -> None:
            """Receive audio chunks as they arrive."""
            audio_chunk_count = 0
            total_audio_bytes = 0
            try:
                while True:
                    if cancelled[0]:
                        # Drain remaining audio
                        while not audio_queue.empty():
                            try:
                                await asyncio.wait_for(audio_queue.get(), timeout=0.01)
                            except Exception:
                                break
                        break

                    logger.debug(f"[AUDIO_RX] waiting for audio chunk from queue (received {audio_chunk_count} so far)")
                    audio_chunk = await audio_queue.get()
                    logger.debug(f"[AUDIO_RX] got item from queue: {type(audio_chunk)}, is_none={audio_chunk is None}")

                    # None signals completion - but callbacks may still be firing
                    if audio_chunk is None:
                        # Wait briefly to collect any trailing chunks from callbacks
                        logger.debug("[AUDIO_RX] completion sentinel received, waiting 100ms for trailing chunks")
                        trailing_chunks = 0
                        try:
                            while True:
                                next_chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
                                if next_chunk is None:
                                    # Second sentinel, truly done
                                    break
                                if not cancelled[0]:
                                    trailing_chunks += 1
                                    audio_chunk_count += 1
                                    total_audio_bytes += len(next_chunk)
                                    logger.debug(
                                        "[AUDIO_RX] pushing trailing audio chunk",
                                        extra={"chunk_idx": audio_chunk_count, "len": len(next_chunk)},
                                    )
                                    output_emitter.push(next_chunk)
                        except asyncio.TimeoutError:
                            # No more chunks arriving
                            pass
                        
                        if trailing_chunks > 0:
                            logger.info(
                                "[AUDIO_RX] collected trailing chunks after completion sentinel",
                                extra={"trailing": trailing_chunks},
                            )
                        
                        if audio_chunk_count == 0:
                            logger.error(
                                "[AUDIO_RX] ⚠️ NO AUDIO CHUNKS RECEIVED! This is the root cause of 'no audio frames were pushed'"
                            )
                        logger.info(
                            "[AUDIO_RX] audio stream completed",
                            extra={"chunks": audio_chunk_count, "bytes": total_audio_bytes},
                        )
                        break

                    # Only push audio if not cancelled
                    if not cancelled[0]:
                        audio_chunk_count += 1
                        total_audio_bytes += len(audio_chunk)
                        logger.debug(
                            "[AUDIO_RX] pushing audio chunk",
                            extra={"chunk_idx": audio_chunk_count, "len": len(audio_chunk)},
                        )
                        output_emitter.push(audio_chunk)
            except asyncio.CancelledError:
                cancelled[0] = True
                raise
            except Exception as e:
                raise

        try:
            # Start SDK synthesis in thread pool
            synthesis_task = loop.run_in_executor(None, _run_sdk_synthesis)
            # Run text streaming and audio receiving concurrently
            await asyncio.gather(
                _stream_text_input(),
                _receive_audio()
            )
            # Check for errors
            if synthesis_error:
                logger.error(f"[SEGMENT] synthesis error detected: {synthesis_error[0]}")
                raise synthesis_error[0]

            # Wait for synthesis thread to complete
            await synthesis_task
            logger.info("[SEGMENT] SDK synthesis thread completed")

            logger.info(f"[SEGMENT] ending segment {segment_id}")
            output_emitter.end_segment()
            logger.info(f"[SEGMENT] segment {segment_id} ended successfully")

        except asyncio.CancelledError:
            # Clean up on interruption
            logger.info(f"[SEGMENT] synthesis interrupted for segment {segment_id}, stopping synthesizer")
            cancelled[0] = True

            # Stop the Azure synthesizer to terminate ongoing synthesis
            # This only stops the current operation, synthesizer can be reused
            try:
                if self._tts._synthesizer:
                    stop_future = self._tts._synthesizer.stop_speaking_async()
                    # Wait for stop to complete (with timeout to avoid hanging)
                    await asyncio.wait_for(
                        loop.run_in_executor(None, stop_future.get),
                        timeout=1.0
                    )
                    logger.debug("[SEGMENT] stopped azure synthesizer")
            except asyncio.TimeoutError:
                logger.warning("[SEGMENT] timeout stopping synthesizer")
            except Exception as e:
                logger.warning(f"[SEGMENT] error stopping synthesizer: {e}")

            # Signal SDK thread to stop
            text_queue.put(None)
            # Flush any queued audio immediately
            output_emitter.flush()
            # End segment on cancellation
            try:
                logger.debug(f"[SEGMENT] attempting to end segment {segment_id} after cancellation")
                output_emitter.end_segment()
                logger.debug(f"[SEGMENT] segment {segment_id} ended after cancellation")
            except Exception as e:
                logger.warning(f"[SEGMENT] failed to end segment {segment_id} after cancellation: {e}")
            # Drain queues
            while not text_queue.empty():
                try:
                    text_queue.get_nowait()
                except:
                    break
            while not audio_queue.empty():
                try:
                    await asyncio.wait_for(audio_queue.get(), timeout=0.01)
                except:
                    break
            raise
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
                logger.info(f"[SEGMENT] segment {segment_id} cancelled due to user interruption (normal behavior)")
                try:
                    logger.debug(f"[SEGMENT] ending segment {segment_id} after user interruption")
                    output_emitter.end_segment()
                    logger.debug(f"[SEGMENT] segment {segment_id} ended after user interruption")
                except Exception as end_error:
                    logger.debug(f"[SEGMENT] failed to end segment {segment_id} after interruption: {end_error}")
                # Clean up queues
                while not text_queue.empty():
                    try:
                        text_queue.get_nowait()
                    except:
                        break
                while not audio_queue.empty():
                    try:
                        await asyncio.wait_for(audio_queue.get(), timeout=0.01)
                    except:
                        break
                # Don't raise error for user interruption - just return normally
                return
            else:
                # This is a real error
                logger.error(f"[SEGMENT] error in segment {segment_id}: {e}", exc_info=True)
                try:
                    logger.debug(f"[SEGMENT] attempting to end segment {segment_id} after error")
                    output_emitter.end_segment()
                    logger.debug(f"[SEGMENT] segment {segment_id} ended after error")
                except Exception as end_error:
                    logger.error(f"[SEGMENT] failed to end segment {segment_id} after error: {end_error}")
                # Clean up queues on error
                while not text_queue.empty():
                    try:
                        text_queue.get_nowait()
                    except:
                        break
                while not audio_queue.empty():
                    try:
                        await asyncio.wait_for(audio_queue.get(), timeout=0.01)
                    except:
                        break
                raise APIConnectionError(f"Azure SDK synthesis failed: {e}") from e
