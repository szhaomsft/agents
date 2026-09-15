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
import threading
import weakref
from dataclasses import dataclass, replace
from typing import Literal
from urllib.parse import urlparse

import aiohttp

try:
    import azure.cognitiveservices.speech as speechsdk
except ImportError:
    raise ImportError(
        "azure-cognitiveservices-speech is required for streaming. "
        "Install with: pip install azure-cognitiveservices-speech"
    )

from livekit.agents import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
    tokenize,
    tts,
    utils,
)
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
    language: LanguageCode | None
    speech_endpoint: str | None
    deployment_id: str | None
    prosody: NotGivenOr[ProsodyConfig]
    style: NotGivenOr[StyleConfig]
    lexicon_uri: NotGivenOr[str]
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
        lexicon_uri: NotGivenOr[str] = NOT_GIVEN,
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
            language=LanguageCode(language) if language else None,
            prosody=prosody,
            style=style,
            lexicon_uri=lexicon_uri,
            auth_token=speech_auth_token,
        )
        self._streams = weakref.WeakSet[SynthesizeStream]()
        self._prewarmed: (
            tuple[_TTSOptions, speechsdk.SpeechSynthesizer, speechsdk.Connection] | None
        ) = None
        self._closed = False

    @property
    def model(self) -> str:
        if self._opts.speech_endpoint:
            endpoint = self._opts.speech_endpoint.strip()
            parsed = urlparse(endpoint if "://" in endpoint else f"//{endpoint}")
            location = parsed.hostname or endpoint
        elif self._opts.region:
            location = self._opts.region
        else:
            return "unknown"
        model = f"{location}:{self._opts.voice}"
        if self._opts.deployment_id:
            model += f":{self._opts.deployment_id}"
        return model

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
        lexicon_uri: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        if is_given(voice):
            self._opts.voice = voice
        if is_given(language):
            self._opts.language = LanguageCode(language)
        if is_given(prosody):
            prosody.validate()
            self._opts.prosody = prosody
        if is_given(style):
            style.validate()
            self._opts.style = style
        if is_given(lexicon_uri):
            self._opts.lexicon_uri = lexicon_uri

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

    def prewarm(self) -> None:
        """Open an idle SDK connection without synthesizing dummy speech."""
        if self._closed:
            return
        if self._prewarmed is not None:
            if self._prewarmed[0] == self._opts:
                return
            self._prewarmed[2].close()
            self._prewarmed = None
        self._prewarmed = self._open_synthesizer(self._opts)

    def _open_synthesizer(
        self, opts: _TTSOptions
    ) -> tuple[_TTSOptions, speechsdk.SpeechSynthesizer, speechsdk.Connection]:
        synthesizer = self._create_synthesizer(opts)
        connection = speechsdk.Connection.from_speech_synthesizer(synthesizer)
        # SDK open() starts connection establishment without waiting for it.
        connection.open(True)
        return replace(opts), synthesizer, connection

    def _take_synthesizer(
        self, opts: _TTSOptions
    ) -> tuple[_TTSOptions, speechsdk.SpeechSynthesizer, speechsdk.Connection]:
        warmed, self._prewarmed = self._prewarmed, None
        if warmed is not None:
            if warmed[0] == opts:
                return warmed
            warmed[2].close()
        return self._open_synthesizer(opts)

    def _create_synthesizer(self, opts: _TTSOptions) -> speechsdk.SpeechSynthesizer:
        if opts.speech_endpoint:
            endpoint = opts.speech_endpoint.replace(
                "/cognitiveservices/v1", "/cognitiveservices/websocket/v2"
            )
        else:
            endpoint = f"wss://{opts.region}.tts.speech.microsoft.com/cognitiveservices/websocket/v2"

        speech_config = speechsdk.SpeechConfig(
            endpoint=endpoint,
            subscription=opts.subscription_key or "",
        )
        if opts.deployment_id:
            speech_config.endpoint_id = opts.deployment_id
        speech_config.speech_synthesis_voice_name = opts.voice
        speech_config.set_speech_synthesis_output_format(
            SDK_OUTPUT_FORMATS.get(
                opts.sample_rate,
                speechsdk.SpeechSynthesisOutputFormat.Raw24Khz16BitMonoPcm,
            )
        )
        return speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)

    async def aclose(self) -> None:
        self._closed = True
        if self._prewarmed is not None:
            self._prewarmed[2].close()
            self._prewarmed = None
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

        if is_given(self._opts.lexicon_uri):
            ssml += f'<lexicon uri="{self._opts.lexicon_uri}"/>'

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
            async with self._tts._ensure_session().post(
                url=self._opts.get_endpoint_url(),
                headers=headers,
                data=self._build_ssml(),
                timeout=aiohttp.ClientTimeout(total=30, sock_connect=self._conn_options.timeout),
            ) as resp:
                resp.raise_for_status()
                async for data, _ in resp.content.iter_chunks():
                    output_emitter.push(data)

        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
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
        self._text_ch = utils.aio.Chan[str | None]()
        # SDK events and stop_speaking_async apply to the entire synthesizer.
        # Each stream needs its own instance to isolate overlapping requests.
        self._connection: speechsdk.Connection | None = None
        self._synthesizer = self._create_synthesizer()

    async def aclose(self) -> None:
        await super().aclose()
        if self._connection is not None:
            self._connection.close()
            self._connection = None
            # Warm the next request while the current response is playing.
            self._tts.prewarm()

    def _recreate_synthesizer(self) -> None:
        """Recreate the synthesizer after connection issues."""
        logger.info("recreating azure tts synthesizer after connection error")
        if self._connection is not None:
            self._connection.close()
        self._synthesizer = self._create_synthesizer()

    def _create_synthesizer(self) -> speechsdk.SpeechSynthesizer:
        _, synthesizer, self._connection = self._tts._take_synthesizer(self._opts)
        return synthesizer

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
                    self._text_ch.send_nowait(input)
                elif isinstance(input, self._FlushSentinel):
                    # Flush marks segment boundary
                    self._text_ch.send_nowait(None)
            self._text_ch.close()

        async def _run_synthesis() -> None:
            """Run synthesis task with retry on 499 errors."""
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    await self._synthesize_segment(output_emitter)
                    break  # Success, exit retry loop
                except (APIStatusError, APIConnectionError) as e:
                    # Check if it's a 499 error (client closed connection)
                    error_str = str(e)
                    if "499" in error_str or "closed by client" in error_str.lower():
                        if attempt < max_retries - 1:
                            logger.warning(
                                f"azure tts 499 error, recreating synthesizer and retrying (attempt {attempt + 1}/{max_retries})"
                            )
                            # Recreate synthesizer
                            self._recreate_synthesizer()
                            await asyncio.sleep(1.0)  # Wait before retry
                            continue
                    raise  # Re-raise if not retryable or out of retries

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
        segment_id = utils.shortuuid()
        output_emitter.start_segment(segment_id=segment_id)

        # Use asyncio Queue to bridge sync callbacks to async
        audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        synthesis_error: list[Exception] = []
        text_queue: queue.Queue[str | None] = queue.Queue()  # Sync queue for thread-safe access
        cancelled = [False]  # Flag to signal cancellation to SDK thread

        # Get the event loop before entering the thread
        loop = asyncio.get_event_loop()
        synthesizer = self._synthesizer

        def _run_sdk_synthesis() -> None:
            """Run Azure SDK synthesis in sync mode with streaming callbacks."""
            callback_lock = threading.Lock()
            callback_audio = bytearray()
            accepting_audio = True

            def synthesizing_callback(evt):
                """Called when audio chunks are available during synthesis."""
                with callback_lock:
                    if cancelled[0] or not accepting_audio:
                        return
                    if audio_chunk := evt.result.audio_data:
                        callback_audio.extend(audio_chunk)
                        loop.call_soon_threadsafe(audio_queue.put_nowait, audio_chunk)

            def canceled_callback(evt):
                """Called when synthesis is canceled or fails."""
                cancellation = evt.result.cancellation_details
                error = APIStatusError(
                    f"Azure TTS synthesis canceled: {cancellation.reason}. "
                    f"Error: {cancellation.error_details}"
                )
                synthesis_error.append(error)

            # Connect event handlers
            synthesizer.synthesizing.connect(synthesizing_callback)
            synthesizer.synthesis_canceled.connect(canceled_callback)

            try:
                tts_request = speechsdk.SpeechSynthesisRequest(
                    input_type=speechsdk.SpeechSynthesisRequestInputType.TextStream
                )
                # Start synthesis (returns result future)
                result_future = synthesizer.speak_async(tts_request)

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
                tts_request.input_stream.close()
                
                # Wait for the complete result, not just the completion event.
                result = result_future.get()
                if (
                    result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted
                    and not synthesis_error
                ):
                    raise APIConnectionError(f"Azure TTS synthesis did not complete: {result.reason}")
                with callback_lock:
                    # SDK completion can precede the final audio callbacks.
                    # Freeze delivery before reconciling with the complete PCM
                    # result, so a late callback cannot duplicate the tail.
                    accepting_audio = False
                    if not cancelled[0] and not synthesis_error:
                        result_audio = result.audio_data
                        if not result_audio.startswith(callback_audio):
                            raise APIConnectionError(
                                "Azure TTS callback audio differs from the final synthesis result"
                            )
                        if tail := result_audio[len(callback_audio) :]:
                            loop.call_soon_threadsafe(audio_queue.put_nowait, tail)
                            logger.debug(
                                "azure tts forwarded final audio from SDK result",
                                extra={"tail_bytes": len(tail)},
                            )
            except Exception as e:
                synthesis_error.append(e)
            finally:
                with callback_lock:
                    accepting_audio = False
                synthesizer.synthesizing.disconnect_all()
                synthesizer.synthesis_canceled.disconnect_all()
                # Only the finished SDK request may end this stream's audio.
                loop.call_soon_threadsafe(audio_queue.put_nowait, None)

        async def _stream_text_input() -> None:
            """Stream text chunks to the SDK as they arrive."""
            async for text_chunk in self._text_ch:
                if text_chunk is None:
                    # End of segment
                    break
                self._mark_started()
                # Send to sync thread via sync queue
                text_queue.put(text_chunk)
            # Signal end of text
            text_queue.put(None)
  
        async def _receive_audio() -> None:
            """Receive audio chunks as they arrive."""
            try:
                while True:
                    if cancelled[0]:
                        logger.debug("audio reception cancelled, discarding remaining chunks")
                        # Drain remaining audio
                        while not audio_queue.empty():
                            try:
                                await asyncio.wait_for(audio_queue.get(), timeout=0.01)
                            except:
                                break
                        break

                    audio_chunk = await audio_queue.get()

                    # None signals completion
                    if audio_chunk is None:
                        break

                    # Only push audio if not cancelled
                    if not cancelled[0]:
                        output_emitter.push(audio_chunk)
            except asyncio.CancelledError:
                cancelled[0] = True
                text_queue.put(None)
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
                raise synthesis_error[0]

            # Wait for synthesis thread to complete
            await synthesis_task

            output_emitter.end_segment()

        except asyncio.CancelledError:
            # Clean up on interruption
            logger.info("synthesis interrupted, stopping synthesizer")
            cancelled[0] = True

            # Stop the Azure synthesizer to terminate ongoing synthesis
            # This only stops the current operation, synthesizer can be reused
            try:
                if synthesizer:
                    stop_future = synthesizer.stop_speaking_async()
                    # Wait for stop to complete (with timeout to avoid hanging)
                    await asyncio.wait_for(
                        loop.run_in_executor(None, stop_future.get),
                        timeout=1.0
                    )
                    logger.debug("stopped azure synthesizer")
            except asyncio.TimeoutError:
                logger.warning("timeout stopping synthesizer")
            except Exception as e:
                logger.warning(f"error stopping synthesizer: {e}")

            # Signal SDK thread to stop
            text_queue.put(None)
            # Flush any queued audio immediately
            output_emitter.flush()
            # End segment on cancellation
            try:
                output_emitter.end_segment()
            except:
                pass  # Segment might already be ended
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
            # End segment on error
            try:
                output_emitter.end_segment()
            except:
                pass  # Segment might already be ended
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
