"""video_vision — analyze a video by sampling frames + transcribing audio.

Pipeline (v1):
  1. ffprobe the duration; pick a frame interval (2-10s) so we take ≤20 frames.
  2. ffmpeg extracts the frames.
  3. ffmpeg extracts the audio; Soniox transcribes it WITH timestamps.
  4. each frame is described by the agent's vision (image_vision), given the
     transcript around that timestamp as extra context.
  5. all frame descriptions + the transcript are synthesized into one
     coherent video description by the agent's own model.

Frame vision is run with limited concurrency. Requires system ``ffmpeg`` +
``ffprobe`` (same assumption the audio pipeline already makes). Soniox
(``SONIOX_API_KEY``) is optional — without it, analysis is frames-only.
"""

from __future__ import annotations

import asyncio
import shutil
import uuid
from pathlib import Path
from typing import Any

from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)

_MAX_FRAMES = 20
_MIN_INTERVAL = 2.0
_MAX_INTERVAL = 10.0
_FRAME_CONCURRENCY = 4
_VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}
_SONIOX_API_BASE = "https://api.soniox.com"
_SONIOX_STT_MODEL = "stt-async-v4"


# ── subprocess helpers ────────────────────────────────────────────────


async def _run(cmd: list[str], *, input_bytes: bytes | None = None, timeout: float = 120.0) -> tuple[int, bytes, bytes]:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE if input_bytes is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(input=input_bytes), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        return 1, b"", b"timeout"
    return proc.returncode or 0, out or b"", err or b""


async def _ffprobe_duration(ffprobe: str, path: str) -> float:
    rc, out, _ = await _run([
        ffprobe, "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", path,
    ], timeout=30.0)
    if rc != 0:
        return 0.0
    try:
        return float(out.decode().strip())
    except (ValueError, AttributeError):
        return 0.0


def _seek_args(ss: float, t: float | None) -> tuple[list[str], list[str]]:
    """Return (pre_input_args, post_input_args) for an optional [ss, ss+t] window."""
    pre = ["-ss", f"{ss:.3f}"] if ss and ss > 0 else []
    post = ["-t", f"{t:.3f}"] if t and t > 0 else []
    return pre, post


async def _extract_frames(
    ffmpeg: str, path: str, interval: float, count: int, out_dir: Path,
    *, ss: float = 0.0, t: float | None = None,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "frame_%03d.jpg")
    pre, post = _seek_args(ss, t)
    # One frame every `interval` seconds, capped at `count`, within the window.
    # -ss before -i = fast seek; scale caps the long edge before vision resize.
    rc, _, err = await _run([
        ffmpeg, "-hide_banner", "-loglevel", "error", *pre, "-i", path, *post,
        "-vf", f"fps=1/{interval:.4f},scale='min(1280,iw)':-2",
        "-frames:v", str(count), "-y", pattern,
    ], timeout=180.0)
    if rc != 0:
        log.warning("ffmpeg frame extraction failed: %s", err.decode(errors="replace")[:200])
    return sorted(out_dir.glob("frame_*.jpg"))


async def _extract_audio_wav(ffmpeg: str, path: str, *, ss: float = 0.0, t: float | None = None) -> bytes:
    pre, post = _seek_args(ss, t)
    rc, out, _ = await _run([
        ffmpeg, "-hide_banner", "-loglevel", "error", *pre, "-i", path, *post,
        "-vn", "-ar", "16000", "-ac", "1", "-f", "wav", "pipe:1",
    ], timeout=180.0)
    return out if rc == 0 else b""


def _parse_time(val: Any) -> float | None:
    """Parse a time spec: seconds (number/'90') or 'MM:SS' / 'HH:MM:SS'. None if blank/invalid."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val) if val >= 0 else None
    s = str(val).strip()
    if not s:
        return None
    try:
        if ":" in s:
            parts = [float(p) for p in s.split(":")]
            secs = 0.0
            for p in parts:
                secs = secs * 60 + p
            return secs if secs >= 0 else None
        return max(0.0, float(s))
    except ValueError:
        return None


# ── Soniox transcription WITH timestamps ──────────────────────────────


async def _transcribe_timestamped(wav_bytes: bytes) -> tuple[str, list[tuple[float, str]]]:
    """Return (full_text, [(start_seconds, token_text), ...]) or ('', [])."""
    import os

    import httpx

    api_key = os.environ.get("SONIOX_API_KEY", "").strip()
    if not api_key or not wav_bytes:
        return "", []
    headers = {"Authorization": f"Bearer {api_key}"}
    file_id = transcription_id = ""
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            up = await client.post(
                f"{_SONIOX_API_BASE}/v1/files", headers=headers,
                files={"file": ("audio.wav", wav_bytes, "audio/wav")},
            )
            up.raise_for_status()
            file_id = str((up.json() or {}).get("id") or "")
            if not file_id:
                return "", []
            cr = await client.post(
                f"{_SONIOX_API_BASE}/v1/transcriptions",
                headers={**headers, "Content-Type": "application/json"},
                json={"model": _SONIOX_STT_MODEL, "file_id": file_id,
                      "enable_language_identification": True},
            )
            cr.raise_for_status()
            transcription_id = str((cr.json() or {}).get("id") or "")
            if not transcription_id:
                return "", []
            for _ in range(120):
                p = await client.get(
                    f"{_SONIOX_API_BASE}/v1/transcriptions/{transcription_id}", headers=headers)
                p.raise_for_status()
                st = str((p.json() or {}).get("status") or "")
                if st in ("completed", "error"):
                    break
                await asyncio.sleep(1)
            text, tokens = "", []
            tr = await client.get(
                f"{_SONIOX_API_BASE}/v1/transcriptions/{transcription_id}/transcript", headers=headers)
            if tr.status_code == 200:
                data = tr.json() or {}
                text = str(data.get("text") or "").strip()
                for tok in (data.get("tokens") or []):
                    t = str(tok.get("text") or "")
                    start_ms = tok.get("start_ms")
                    if t and start_ms is not None:
                        tokens.append((float(start_ms) / 1000.0, t))
            # cleanup
            for pth in (f"/v1/transcriptions/{transcription_id}", f"/v1/files/{file_id}"):
                try:
                    await client.delete(f"{_SONIOX_API_BASE}{pth}", headers=headers)
                except Exception:
                    pass
            return text, tokens
    except Exception as exc:
        log.warning("soniox timestamped transcription failed: %s", exc)
        return "", []


def _transcript_window(tokens: list[tuple[float, str]], center: float, half: float) -> str:
    """Join tokens whose start is within [center-half, center+half]."""
    parts = [t for (ts, t) in tokens if center - half <= ts <= center + half]
    return "".join(parts).strip()


class VideoVisionTool(Tool):
    """Analyze a video: sample frames + transcribe audio, then describe it."""

    name = "video_vision"
    timeout_seconds = 600.0
    description = (
        "Analyze and describe a VIDEO file. Samples up to 20 frames across the "
        "video, transcribes the audio (with timestamps), describes each frame "
        "with vision, and synthesizes a coherent description of the whole video "
        "(visuals + spoken content over time). Use for .mp4/.mov/.webm/etc. "
        "Pass the video 'path'. Slow (many vision calls) — call once per video."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the video file (saved/ workspace or absolute)."},
            "prompt": {
                "type": "string",
                "description": "Optional focus for the description (e.g. 'what is the person doing?').",
            },
            "start": {
                "type": "string",
                "description": "Optional start of the segment to analyze: seconds ('90') or 'MM:SS'/'HH:MM:SS'. Omit for the whole video.",
            },
            "end": {
                "type": "string",
                "description": "Optional end of the segment to analyze (same format as start). Omit to go to the end of the video.",
            },
        },
        "required": ["path"],
    }

    async def execute(self, path: str, prompt: str = "", start: Any = None, end: Any = None, **kwargs: Any) -> ToolResult:
        path_str = str(path or "").strip()
        if not path_str:
            return ToolResult(success=False, error="Missing required argument: path")

        ffmpeg = shutil.which("ffmpeg")
        ffprobe = shutil.which("ffprobe")
        if not ffmpeg or not ffprobe:
            return ToolResult(success=False, error="video_vision requires system 'ffmpeg' and 'ffprobe' to be installed.")

        # Resolve the video file.
        from captain_claw.tools.image_ocr import _require_existing_file
        runtime_base = kwargs.get("_runtime_base_path")
        file_path, error = _require_existing_file(path_str, runtime_base_path=runtime_base)
        if error:
            return ToolResult(success=False, error=error)
        if file_path.suffix.lower() not in _VIDEO_EXTS:
            return ToolResult(
                success=False,
                error=f"Not a video ('{file_path.suffix}'). Supported: {', '.join(sorted(_VIDEO_EXTS))}.",
            )

        duration = await _ffprobe_duration(ffprobe, str(file_path))
        if duration <= 0:
            return ToolResult(success=False, error="Could not read the video's duration (corrupt or unsupported codec?).")

        # v2: optional [start, end] segment. Clamp to the video; default = whole.
        seg_start = _parse_time(start) or 0.0
        seg_start = max(0.0, min(seg_start, duration))
        parsed_end = _parse_time(end)
        seg_end = duration if parsed_end is None else max(0.0, min(parsed_end, duration))
        if seg_end <= seg_start:
            seg_end = duration  # invalid/zero window → fall back to the rest of the video
        seg_dur = seg_end - seg_start
        is_segment = seg_start > 0.0 or seg_end < duration

        interval = min(_MAX_INTERVAL, max(_MIN_INTERVAL, seg_dur / _MAX_FRAMES))
        n_frames = max(1, min(_MAX_FRAMES, int(seg_dur // interval) + 1))
        ss = seg_start if is_segment else 0.0
        t = seg_dur if is_segment else None

        # Frame output dir under saved/.
        saved_base = kwargs.get("_saved_base_path")
        base = Path(saved_base) if saved_base else (Path(runtime_base) / "saved" if runtime_base else file_path.parent)
        frame_dir = Path(base) / "video_frames" / uuid.uuid4().hex[:8]

        log.info("video_vision: analyzing", path=str(file_path), duration=round(duration, 1),
                 segment=f"{round(seg_start,1)}-{round(seg_end,1)}s" if is_segment else "full",
                 interval=round(interval, 1), frames=n_frames)

        # Extract frames + audio (for the segment) concurrently.
        frames, wav = await asyncio.gather(
            _extract_frames(ffmpeg, str(file_path), interval, n_frames, frame_dir, ss=ss, t=t),
            _extract_audio_wav(ffmpeg, str(file_path), ss=ss, t=t),
        )
        if not frames:
            return ToolResult(success=False, error="Frame extraction produced no frames (ffmpeg failed?).")

        full_text, tokens = await _transcribe_timestamped(wav)

        # Describe each frame (with nearby transcript) — limited concurrency.
        from captain_claw.tools.image_ocr import ImageVisionTool
        ivt = ImageVisionTool()
        sem = asyncio.Semaphore(_FRAME_CONCURRENCY)
        half = max(interval / 2.0, 1.5)

        async def describe(idx: int, fp: Path) -> dict[str, Any]:
            rel = idx * interval                      # offset within the segment
            abs_ts = round(seg_start + rel, 1)        # absolute time in the video
            # Transcript tokens are segment-relative (audio extracted from seg_start).
            audio = _transcript_window(tokens, rel, half) if tokens else ""
            fprompt = "Describe this single video frame concisely: people, objects, on-screen text, action."
            if audio:
                fprompt += f" Audio spoken around this moment: \"{audio}\"."
            async with sem:
                try:
                    res = await ivt.execute(path=str(fp), prompt=fprompt, **kwargs)
                    desc = res.content if res.success else f"(frame not described: {res.error})"
                except Exception as exc:
                    desc = f"(frame error: {exc})"
            return {"t": abs_ts, "desc": desc.strip(), "audio": audio}

        analyzed = await asyncio.gather(*(describe(i, fp) for i, fp in enumerate(frames)))

        # Build the per-frame breakdown.
        lines = []
        for a in analyzed:
            seg = f"[{a['t']}s] {a['desc']}"
            if a["audio"]:
                seg += f"  (audio: \"{a['audio']}\")"
            lines.append(seg)
        breakdown = "\n".join(lines)

        # Step 5 — synthesize a coherent description via the agent's own model.
        scope = (f"segment {round(seg_start)}-{round(seg_end)}s of a {round(duration)}s video"
                 if is_segment else f"a {round(duration)}s video")
        summary = await self._synthesize(kwargs.get("_agent"), scope, full_text, breakdown, prompt)

        seg_label = f" of {round(seg_start)}-{round(seg_end)}s" if is_segment else ""
        out = summary or "(synthesis unavailable)"
        out += f"\n\n— Frame-by-frame{seg_label} ({len(frames)} frames @ ~{round(interval,1)}s) —\n{breakdown}"
        if full_text:
            out += f"\n\n— Transcript —\n{full_text}"
        return ToolResult(success=True, content=out)

    @staticmethod
    async def _synthesize(agent: Any, scope: str, transcript: str, breakdown: str, focus: str) -> str:
        if agent is None or not getattr(agent, "provider", None):
            return ""
        from captain_claw.llm import Message
        sys = (
            "You are given a frame-by-frame visual analysis of a video (or a "
            "segment of it) plus its audio transcript. Write a clear, coherent "
            "description: what it shows, what happens over time, and the key "
            "spoken content. Be concise and concrete; do not invent details."
        )
        user = f"This is {scope}.\n\n"
        if focus.strip():
            user += f"Focus on: {focus.strip()}\n\n"
        user += f"Frame-by-frame analysis:\n{breakdown}\n\n"
        user += f"Full transcript:\n{transcript or '(no speech detected)'}"
        try:
            resp = await agent.provider.complete(
                [Message(role="system", content=sys), Message(role="user", content=user)],
            )
            return str(getattr(resp, "content", "") or "").strip()
        except Exception as exc:
            log.warning("video_vision synthesis failed: %s", exc)
            return ""
