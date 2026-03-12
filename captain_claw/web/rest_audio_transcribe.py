"""REST handler for browser audio transcription (voice input)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiohttp import web

from captain_claw.logging import get_logger

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer

log = get_logger(__name__)


async def transcribe_audio_handler(
    server: "WebServer", request: web.Request
) -> web.Response:
    """POST /api/audio/transcribe — receive browser audio, transcribe, return text.

    Accepts multipart form data with an ``audio`` field containing the
    browser-recorded audio blob (typically webm/opus from MediaRecorder).

    Returns JSON ``{"text": "..."}`` or ``{"error": "..."}``.
    """
    try:
        reader = await request.multipart()
        if reader is None:
            return web.json_response(
                {"error": "Multipart body required"}, status=400
            )

        audio_field = None
        while True:
            field = await reader.next()
            if field is None:
                break
            if field.name == "audio":
                audio_field = field
                break

        if audio_field is None:
            return web.json_response(
                {"error": "No 'audio' field in upload"}, status=400
            )

        # Read audio data.
        chunks: list[bytes] = []
        while True:
            chunk = await audio_field.read_chunk(8192)
            if not chunk:
                break
            chunks.append(chunk)
        audio_bytes = b"".join(chunks)

        if not audio_bytes:
            return web.json_response({"error": "Empty audio"}, status=400)

        # Size guard: reject unreasonably large uploads (> 10 MB).
        if len(audio_bytes) > 10 * 1024 * 1024:
            return web.json_response(
                {"error": "Audio too large (max 10 MB)"}, status=400
            )

        log.info(
            "audio_transcribe_request",
            size_kb=f"{len(audio_bytes) / 1024:.1f}",
            content_type=audio_field.headers.get("Content-Type", "unknown"),
        )

        # Convert webm/ogg to WAV if needed — transcribe_audio expects WAV.
        wav_bytes = await _ensure_wav(audio_bytes)

        # Reuse the existing STT pipeline.
        from captain_claw.tools.stt import transcribe_audio

        text = await transcribe_audio(wav_bytes)

        log.info("audio_transcribe_complete", chars=len(text))
        return web.json_response({"text": text})

    except RuntimeError as exc:
        # STT provider errors (no provider configured, API key missing, etc.)
        log.warning("audio_transcribe_stt_error", error=str(exc))
        return web.json_response({"error": str(exc)}, status=503)
    except web.HTTPException:
        raise
    except Exception as exc:
        log.error("audio_transcribe_failed", error=str(exc))
        return web.json_response(
            {"error": f"Transcription failed: {exc}"}, status=500
        )


async def _ensure_wav(audio_bytes: bytes) -> bytes:
    """Convert audio to WAV if not already in WAV format.

    Uses ffmpeg subprocess for conversion.  Falls back to passing bytes
    through if ffmpeg is not available (some providers accept webm natively).
    """
    import asyncio
    import shutil

    # Already WAV — pass through.
    if audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE":
        return audio_bytes

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        log.warning(
            "ffmpeg_not_found",
            msg="ffmpeg not available; passing raw audio to STT provider. "
            "Install ffmpeg for reliable browser audio transcription.",
        )
        return audio_bytes

    # Convert to 16 kHz mono PCM WAV via ffmpeg.
    proc = await asyncio.create_subprocess_exec(
        ffmpeg_bin,
        "-i", "pipe:0",
        "-ar", "16000",
        "-ac", "1",
        "-f", "wav",
        "-acodec", "pcm_s16le",
        "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate(input=audio_bytes)

    if proc.returncode != 0:
        err_msg = stderr.decode("utf-8", errors="replace")[:500]
        log.error(
            "ffmpeg_conversion_failed",
            returncode=proc.returncode,
            stderr=err_msg,
        )
        raise RuntimeError(f"Audio format conversion failed: {err_msg}")

    if not stdout:
        raise RuntimeError("Audio conversion produced empty output.")

    log.info(
        "audio_converted_to_wav",
        input_size=len(audio_bytes),
        output_size=len(stdout),
    )
    return stdout
