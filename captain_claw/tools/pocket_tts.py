"""Pocket TTS tool for local text-to-speech synthesis."""

import asyncio
import importlib
import shutil
import subprocess
from array import array
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from captain_claw.config import get_config
from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult
from captain_claw.tools.write import WriteTool

log = get_logger(__name__)


def _iter_scalar_samples(value: Any):
    """Yield float samples from arbitrarily nested list/tensor-like values."""
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_scalar_samples(item)
        return
    try:
        yield float(value)
    except Exception:
        return


def _coerce_samples(audio: Any) -> list[float]:
    """Normalize model audio output into a flat float sample list."""
    value = audio
    if hasattr(value, "detach"):
        try:
            value = value.detach()
        except Exception:
            pass
    if hasattr(value, "cpu"):
        try:
            value = value.cpu()
        except Exception:
            pass
    if hasattr(value, "reshape") and hasattr(value, "tolist"):
        try:
            value = value.reshape(-1).tolist()
        except Exception:
            pass
    elif hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:
            pass

    if not isinstance(value, (list, tuple)):
        try:
            value = list(value)
        except Exception:
            value = [value]

    return [sample for sample in _iter_scalar_samples(value)]


def _samples_to_pcm16(samples: list[float]) -> bytes:
    """Convert float/pcm-like samples to signed 16-bit PCM bytes."""
    pcm = array("h")
    for sample in samples:
        if abs(sample) <= 1.5:
            scaled = int(round(sample * 32767.0))
        else:
            scaled = int(round(sample))
        clipped = max(-32768, min(32767, scaled))
        pcm.append(clipped)
    return pcm.tobytes()


def _encode_mp3_bytes(pcm_data: bytes, sample_rate: int, bitrate_kbps: int) -> bytes:
    """Encode mono PCM16 bytes to MP3 using lameenc or ffmpeg fallback."""
    try:
        lameenc = importlib.import_module("lameenc")
        encoder = lameenc.Encoder()
        encoder.set_bit_rate(max(32, int(bitrate_kbps)))
        encoder.set_in_sample_rate(max(1, int(sample_rate)))
        encoder.set_channels(1)
        encoder.set_quality(2)
        encoded = encoder.encode(pcm_data) + encoder.flush()
        if not encoded:
            raise RuntimeError("MP3 encoding produced empty output.")
        return encoded
    except Exception:
        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            raise RuntimeError(
                "MP3 encoding requires either Python package 'lameenc' or system 'ffmpeg'. "
                "On macOS install ffmpeg with: brew install ffmpeg"
            )
        bitrate = f"{max(32, int(bitrate_kbps))}k"
        cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "s16le",
            "-ar",
            str(max(1, int(sample_rate))),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-vn",
            "-f",
            "mp3",
            "-b:a",
            bitrate,
            "pipe:1",
        ]
        try:
            proc = subprocess.run(
                cmd,
                input=pcm_data,
                capture_output=True,
                check=True,
            )
        except Exception as e:
            stderr_text = ""
            if hasattr(e, "stderr") and getattr(e, "stderr"):
                try:
                    stderr_text = str(getattr(e, "stderr").decode("utf-8", errors="replace")).strip()
                except Exception:
                    stderr_text = ""
            detail = f": {stderr_text}" if stderr_text else ""
            raise RuntimeError(f"ffmpeg MP3 encoding failed{detail}") from e
        encoded = bytes(proc.stdout or b"")
        if not encoded:
            raise RuntimeError("ffmpeg MP3 encoding produced empty output.")
        return encoded


def _write_bytes(path: Path, payload: bytes) -> None:
    """Write bytes to file."""
    path.write_bytes(payload)


class PocketTTSTool(Tool):
    """Generate local speech audio (MP3) from text via pocket-tts."""

    name = "pocket_tts"
    description = (
        "Convert text to speech using local pocket-tts and save an MP3 (128 kbps) under saved/media."
    )
    parameters = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Text to synthesize into speech.",
            },
            "voice": {
                "type": "string",
                "description": "Optional voice preset (for example af_bella).",
            },
            "output_path": {
                "type": "string",
                "description": (
                    "Optional output path. Paths are normalized under saved/media/<session-id> and saved as .mp3."
                ),
            },
            "sample_rate": {
                "type": "number",
                "description": "Optional sample rate override (Hz).",
            },
        },
        "required": ["text"],
    }

    def __init__(self):
        self._model: Any | None = None
        self._model_load_error: str | None = None

    def _resolve_output_path(self, output_path: str | None, **kwargs: Any) -> tuple[Path, str]:
        """Resolve safe output location under saved root."""
        saved_root = WriteTool._resolve_saved_root(kwargs)
        session_id = WriteTool._normalize_session_id(str(kwargs.get("_session_id", "")))

        requested = str(output_path or "").strip()
        if not requested:
            stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
            requested = f"media/{session_id}/pocket-tts-{stamp}.mp3"

        resolved = WriteTool._normalize_under_saved(requested, saved_root, session_id)
        if resolved.suffix.lower() != ".mp3":
            resolved = resolved.with_suffix(".mp3")
        return resolved, requested

    async def _ensure_model(self) -> Any:
        """Load and cache pocket-tts model lazily."""
        if self._model is not None:
            return self._model
        if self._model_load_error:
            raise RuntimeError(self._model_load_error)
        try:
            module = importlib.import_module("pocket_tts")
        except Exception:
            msg = (
                "Pocket TTS requires dependency 'pocket-tts'. "
                "Install it with: ./venv/bin/pip install pocket-tts"
            )
            self._model_load_error = msg
            raise RuntimeError(msg)

        model_cls = getattr(module, "TTSModel", None)
        if model_cls is None:
            msg = "Installed pocket_tts package is incompatible (missing TTSModel)."
            self._model_load_error = msg
            raise RuntimeError(msg)

        try:
            model = await asyncio.to_thread(model_cls.load_model)
        except Exception as e:
            msg = f"Failed to load pocket-tts model: {e}"
            self._model_load_error = msg
            raise RuntimeError(msg)
        self._model = model
        return model

    @staticmethod
    def _generate_audio(model: Any, text: str, state: Any | None) -> Any:
        """Run model synthesis with light signature compatibility handling."""
        if state is not None:
            call_attempts = [
                lambda: model.generate_audio(state, text),
                lambda: model.generate_audio(text, state),
                lambda: model.generate_audio(voice_state=state, text=text),
                lambda: model.generate_audio(text=text, state=state),
            ]
            for call in call_attempts:
                try:
                    return call()
                except TypeError:
                    continue
            raise TypeError("Unsupported pocket-tts generate_audio signature")
        try:
            return model.generate_audio(text)
        except TypeError:
            return model.generate_audio(text=text)

    async def execute(
        self,
        text: str,
        voice: str = "",
        output_path: str = "",
        sample_rate: int | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Synthesize text and persist an MP3 file."""
        text_value = str(text or "").strip()
        if not text_value:
            return ToolResult(success=False, error="Missing required argument: text")

        cfg = get_config()
        max_chars = max(1, int(getattr(cfg.tools.pocket_tts, "max_chars", 4000)))
        if len(text_value) > max_chars:
            return ToolResult(
                success=False,
                error=(
                    f"Text too long for pocket_tts ({len(text_value)} chars). "
                    f"Max allowed is {max_chars}."
                ),
            )

        output_file, requested_path = self._resolve_output_path(output_path, **kwargs)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            model = await self._ensure_model()
            configured_voice = str(getattr(cfg.tools.pocket_tts, "default_voice", "")).strip()
            default_voice = str(getattr(model, "default_voice", "")).strip()
            selected_voice = str(voice or "").strip() or configured_voice or default_voice

            state = None
            if selected_voice:
                state = await asyncio.to_thread(model.get_state_for_audio_prompt, selected_voice)

            audio = await asyncio.to_thread(self._generate_audio, model, text_value, state)
            samples = await asyncio.to_thread(_coerce_samples, audio)
            if not samples:
                return ToolResult(success=False, error="Pocket TTS returned empty audio output.")

            pcm_data = await asyncio.to_thread(_samples_to_pcm16, samples)
            model_sample_rate = int(getattr(model, "sample_rate", 0) or 0)
            configured_rate = int(getattr(cfg.tools.pocket_tts, "sample_rate", 24000))
            effective_sample_rate = int(sample_rate or model_sample_rate or configured_rate or 24000)
            bitrate_kbps = int(getattr(cfg.tools.pocket_tts, "mp3_bitrate_kbps", 128) or 128)
            mp3_data = await asyncio.to_thread(
                _encode_mp3_bytes,
                pcm_data,
                effective_sample_rate,
                bitrate_kbps,
            )
            await asyncio.to_thread(_write_bytes, output_file, mp3_data)

            redirect_note = ""
            if requested_path != str(output_file):
                redirect_note = f" (requested: {requested_path})"

            duration_seconds = len(samples) / float(max(1, effective_sample_rate))
            voice_label = selected_voice or "default"
            return ToolResult(
                success=True,
                content=(
                    "Generated speech audio with pocket-tts.\n"
                    f"Path: {output_file}{redirect_note}\n"
                    f"Voice: {voice_label}\n"
                    f"Format: mp3\n"
                    f"Bitrate: {bitrate_kbps} kbps\n"
                    f"Sample rate: {effective_sample_rate} Hz\n"
                    f"Duration: {duration_seconds:.2f}s"
                ),
            )
        except Exception as e:
            log.error("Pocket TTS synthesis failed", error=str(e))
            return ToolResult(success=False, error=str(e))
