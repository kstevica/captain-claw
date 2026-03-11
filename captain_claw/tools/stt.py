"""Speech-to-text: audio recording and transcription.

Provider resolution order (first available wins):
  1. Explicit ``stt_provider`` in screen_capture config
  2. Soniox  (``SONIOX_API_KEY``)
  3. OpenAI Whisper (``OPENAI_API_KEY``)
  4. Gemini multimodal (``GEMINI_API_KEY``)
  5. Explicit ``model_type: "stt"`` from ``config.model.allowed``
"""

from __future__ import annotations

import asyncio
import io
import os
import struct
from typing import Any

try:
    from structlog import get_logger

    log = get_logger(__name__)
except Exception:  # pragma: no cover
    import logging

    log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

try:
    import numpy as _np
    import sounddevice as _sd

    _HAS_AUDIO = True
except ImportError:
    _HAS_AUDIO = False

try:
    from soniox import AsyncSonioxClient as _AsyncSonioxClient  # noqa: F401

    _HAS_SONIOX = True
except ImportError:
    _HAS_SONIOX = False


# ---------------------------------------------------------------------------
# Audio recording (synchronous — run via asyncio.to_thread)
# ---------------------------------------------------------------------------


def _record_audio_sync(
    duration_limit: float = 30.0,
    sample_rate: int = 16000,
    channels: int = 1,
    stop_event: Any = None,  # threading.Event
) -> bytes:
    """Record audio from the default microphone until *stop_event* is set
    or *duration_limit* seconds have elapsed.

    Returns WAV bytes (PCM 16-bit).  This is a blocking call — wrap with
    ``asyncio.to_thread``.
    """
    if not _HAS_AUDIO:
        raise RuntimeError(
            "Audio recording requires 'sounddevice' and 'numpy'. "
            "Install with: pip install captain-claw[screen]"
        )

    frames: list[Any] = []

    def _callback(indata: Any, frame_count: int, time_info: Any, status: Any) -> None:
        frames.append(indata.copy())

    stream = _sd.InputStream(
        samplerate=sample_rate,
        channels=channels,
        dtype="int16",
        callback=_callback,
    )
    stream.start()

    # Wait for stop signal or timeout.
    if stop_event is not None:
        stop_event.wait(timeout=duration_limit)
    else:
        import time

        time.sleep(duration_limit)

    stream.stop()
    stream.close()

    if not frames:
        raise RuntimeError("No audio frames captured.")

    audio_data = _np.concatenate(frames, axis=0)
    return _encode_wav(audio_data, sample_rate, channels)


# ---------------------------------------------------------------------------
# WAV encoding (in-memory, no temp files)
# ---------------------------------------------------------------------------


def _encode_wav(samples: Any, sample_rate: int, channels: int) -> bytes:
    """Encode int16 numpy array as WAV bytes."""
    raw = samples.tobytes()
    data_size = len(raw)

    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    # fmt chunk
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))  # chunk size
    buf.write(struct.pack("<H", 1))  # PCM format
    buf.write(struct.pack("<H", channels))
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", sample_rate * channels * 2))  # byte rate
    buf.write(struct.pack("<H", channels * 2))  # block align
    buf.write(struct.pack("<H", 16))  # bits per sample
    # data chunk
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(raw)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Speech-to-text transcription
# ---------------------------------------------------------------------------


def _resolve_stt_api_key(provider: str) -> str:
    """Resolve an API key for the given STT provider."""
    env_map = {
        "soniox": "SONIOX_API_KEY",
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }
    env_var = env_map.get(provider, "")
    if env_var:
        val = os.environ.get(env_var, "").strip()
        if val:
            return val

    # Fall back to captain_claw.llm key resolver for openai/gemini.
    if provider in ("openai", "gemini"):
        try:
            from captain_claw.llm import _resolve_api_key

            key = _resolve_api_key(provider, None)
            if key:
                return key
        except Exception:
            pass
    return ""


async def transcribe_audio(wav_bytes: bytes) -> str:
    """Transcribe WAV audio to text using the best available provider.

    Resolution order:
      1. Explicit ``stt_provider`` in ``tools.screen_capture`` config
      2. Soniox  (if ``SONIOX_API_KEY`` set and ``soniox`` package installed)
      3. OpenAI Whisper (if ``OPENAI_API_KEY`` set)
      4. Gemini multimodal (if ``GEMINI_API_KEY`` set)
      5. Explicit ``model_type: "stt"`` from ``config.model.allowed``
    """
    from captain_claw.config import get_config

    cfg = get_config()

    # --- Check for explicit provider in config ---
    explicit_provider = cfg.tools.screen_capture.stt_provider.strip().lower()
    if explicit_provider:
        return await _transcribe_with_provider(explicit_provider, wav_bytes)

    # --- Auto-detect: try providers in order ---

    # 1. Soniox (preferred)
    soniox_key = _resolve_stt_api_key("soniox")
    if soniox_key and _HAS_SONIOX:
        return await _transcribe_soniox(soniox_key, wav_bytes)

    # 2. OpenAI Whisper
    openai_key = _resolve_stt_api_key("openai")
    if openai_key:
        return await _transcribe_openai_whisper(openai_key, wav_bytes)

    # 3. Gemini multimodal
    gemini_key = _resolve_stt_api_key("gemini")
    if gemini_key:
        return await _transcribe_gemini(gemini_key, wav_bytes)

    # 4. Explicit model_type: "stt" from model.allowed
    for m in cfg.model.allowed:
        if getattr(m, "model_type", "llm") == "stt":
            return await _transcribe_with_model(m, wav_bytes)

    raise RuntimeError(
        "No STT provider available. Set one of: SONIOX_API_KEY, "
        "OPENAI_API_KEY, or GEMINI_API_KEY. Or configure stt_provider "
        "in tools.screen_capture settings."
    )


async def _transcribe_with_provider(provider: str, wav_bytes: bytes) -> str:
    """Route to a specific STT provider by name."""
    key = _resolve_stt_api_key(provider)
    if not key:
        raise RuntimeError(
            f"STT provider '{provider}' configured but no API key found. "
            f"Set the appropriate environment variable."
        )

    if provider == "soniox":
        if not _HAS_SONIOX:
            raise RuntimeError(
                "Soniox configured but 'soniox' package not installed. "
                "Install with: pip install soniox"
            )
        return await _transcribe_soniox(key, wav_bytes)
    elif provider == "openai":
        model = ""
        try:
            from captain_claw.config import get_config

            model = get_config().tools.screen_capture.stt_model
        except Exception:
            pass
        return await _transcribe_openai_whisper(key, wav_bytes, model=model or "whisper-1")
    elif provider in ("gemini", "google"):
        return await _transcribe_gemini(key, wav_bytes)
    else:
        raise RuntimeError(f"Unknown STT provider: {provider}")


async def _transcribe_with_model(model_cfg: Any, wav_bytes: bytes) -> str:
    """Transcribe using an explicitly configured STT model."""
    provider = str(getattr(model_cfg, "provider", "")).lower()
    key = _resolve_stt_api_key(provider)

    if provider == "openai":
        base_url = str(getattr(model_cfg, "base_url", "") or "").strip()
        model_name = str(getattr(model_cfg, "model", "whisper-1"))
        return await _transcribe_openai_whisper(
            key or "",
            wav_bytes,
            model=model_name,
            base_url=base_url or None,
        )
    elif provider in ("gemini", "google"):
        return await _transcribe_gemini(key or "", wav_bytes)
    elif provider == "soniox":
        if not _HAS_SONIOX:
            raise RuntimeError("Soniox package not installed.")
        return await _transcribe_soniox(key or "", wav_bytes)
    else:
        raise RuntimeError(f"STT not supported for provider: {provider}")


# ---------------------------------------------------------------------------
# Soniox realtime STT (mic → WebSocket → text, no audio files)
# ---------------------------------------------------------------------------


def realtime_stt_sync(
    api_key: str,
    stop_event: Any,  # threading.Event
    sample_rate: int = 16000,
    max_duration: float = 30.0,
) -> str:
    """Stream microphone audio to Soniox realtime STT, return transcription.

    Audio is streamed directly from the microphone to Soniox via WebSocket.
    **No audio files are created or uploaded.**

    Blocks until *stop_event* is set or *max_duration* seconds elapse.
    Meant to be called via ``asyncio.to_thread(realtime_stt_sync, ...)``.
    """
    if not _HAS_AUDIO:
        raise RuntimeError(
            "Audio recording requires 'sounddevice' and 'numpy'. "
            "Install with: pip install captain-claw[screen]"
        )
    if not _HAS_SONIOX:
        raise RuntimeError(
            "Soniox realtime STT requires the 'soniox' package. "
            "Install with: pip install soniox"
        )

    import queue as _queue
    import time

    from soniox import SonioxClient
    from soniox.types import RealtimeSTTConfig, Token
    from soniox.utils import render_tokens, start_audio_thread

    client = SonioxClient(api_key=api_key)
    config = RealtimeSTTConfig(
        model="stt-rt-v4",
        audio_format="pcm_s16le",
        sample_rate=sample_rate,
        num_channels=1,
    )

    final_tokens: list[Token] = []
    non_final_tokens: list[Token] = []

    def _mic_iter():
        """Yield raw PCM int16 chunks from the mic until stop_event fires."""
        audio_q: _queue.Queue[bytes] = _queue.Queue()
        block_size = int(sample_rate * 0.1)  # 100 ms chunks

        def _cb(indata: Any, _frames: int, _time_info: Any, _status: Any) -> None:
            audio_q.put(indata.copy().tobytes())

        stream = _sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            callback=_cb,
            blocksize=block_size,
        )
        stream.start()

        deadline = time.monotonic() + max_duration
        try:
            while not stop_event.is_set() and time.monotonic() < deadline:
                try:
                    yield audio_q.get(timeout=0.1)
                except _queue.Empty:
                    continue
        finally:
            stream.stop()
            stream.close()

    with client.realtime.stt.connect(config=config) as session:
        # Audio sender runs in a background thread; it calls session.finish()
        # automatically when the mic iterator is exhausted (key released).
        start_audio_thread(session, _mic_iter())

        # Receive transcription events in *this* thread until the session
        # closes (triggered by finish() above).
        for event in session.receive_events():
            for token in event.tokens:
                if token.is_final:
                    final_tokens.append(token)
                else:
                    non_final_tokens.append(token)
            non_final_tokens.clear()

    text = render_tokens(final_tokens, []).strip()
    log.info("soniox_realtime_complete", chars=len(text))
    return text


# ---------------------------------------------------------------------------
# Provider implementations (file-upload fallback for non-Soniox providers)
# ---------------------------------------------------------------------------


async def _transcribe_soniox(api_key: str, wav_bytes: bytes) -> str:
    """Transcribe using the Soniox async file-based API.

    Uses ``AsyncSonioxClient.stt.transcribe()`` which handles upload +
    create + wait in one call.  The actual text is in a separate
    ``TranscriptionTranscript`` object fetched via ``get_transcript()``.
    """
    from soniox import AsyncSonioxClient

    client = AsyncSonioxClient(api_key=api_key)

    wav_size_kb = len(wav_bytes) / 1024.0
    log.info("soniox_transcribing", wav_size_kb=f"{wav_size_kb:.1f}")

    transcription = await client.stt.transcribe(
        file=wav_bytes,
        filename="recording.wav",
    )

    log.info(
        "soniox_job_done",
        transcription_id=transcription.id,
        status=str(getattr(transcription, "status", "unknown")),
        audio_duration_ms=getattr(transcription, "audio_duration_ms", None),
        error_message=getattr(transcription, "error_message", None),
    )

    # transcribe() may return before the job finishes (status="queued").
    # Explicitly wait for completion before fetching the transcript.
    status = str(getattr(transcription, "status", "")).lower()
    if status != "completed":
        log.info("soniox_waiting_for_completion", transcription_id=transcription.id)
        transcription = await client.stt.wait(transcription.id)
        log.info(
            "soniox_wait_done",
            status=str(getattr(transcription, "status", "unknown")),
        )

    # The Transcription object does NOT contain the text directly.
    # The text lives in TranscriptionTranscript, fetched separately.
    transcript = await client.stt.get_transcript(transcription.id)
    text = str(getattr(transcript, "text", "") or "").strip()

    log.info("soniox_transcription_complete", chars=len(text))
    return text


async def _transcribe_openai_whisper(
    api_key: str,
    wav_bytes: bytes,
    *,
    model: str = "whisper-1",
    base_url: str | None = None,
) -> str:
    """Call the OpenAI-compatible ``/v1/audio/transcriptions`` endpoint."""
    import aiohttp

    url = (base_url or "https://api.openai.com").rstrip("/")
    url = f"{url}/v1/audio/transcriptions"

    form = aiohttp.FormData()
    form.add_field(
        "file",
        wav_bytes,
        filename="recording.wav",
        content_type="audio/wav",
    )
    form.add_field("model", model)

    async with aiohttp.ClientSession() as session:
        async with session.post(
            url,
            data=form,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            text = str(data.get("text", "")).strip()
            log.info("whisper_transcription_complete", chars=len(text))
            return text


async def _transcribe_gemini(api_key: str, wav_bytes: bytes) -> str:
    """Transcribe using Gemini's multimodal audio input via litellm."""
    import base64

    from litellm import completion as litellm_completion

    b64 = base64.b64encode(wav_bytes).decode("ascii")

    response = await asyncio.to_thread(
        litellm_completion,
        model="gemini/gemini-2.0-flash",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Transcribe the following audio recording exactly. "
                            "Return only the transcribed text, nothing else."
                        ),
                    },
                    {
                        "type": "audio_url",
                        "audio_url": {
                            "url": f"data:audio/wav;base64,{b64}",
                        },
                    },
                ],
            }
        ],
        api_key=api_key,
        timeout=60,
    )

    choices = getattr(response, "choices", None)
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    text = str(getattr(message, "content", "") or "").strip()
    log.info("gemini_transcription_complete", chars=len(text))
    return text
