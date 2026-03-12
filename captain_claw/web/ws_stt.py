"""WebSocket handler for live speech-to-text from browser microphone.

Browser streams raw PCM int16 audio at 16 kHz over the WebSocket as binary
frames.  If Soniox realtime is available, audio is forwarded to Soniox and
partial transcription results stream back in real time.  Otherwise, audio is
collected and batch-transcribed when the user stops recording.

Protocol:
    → (binary)              Raw PCM int16 audio chunk
    → {"type": "stop"}      User stopped recording
    ← {"type": "stt_ready", "realtime": bool}   Sent on connect
    ← {"type": "stt_partial", "text": "..."}     Partial transcript (realtime only)
    ← {"type": "stt_final",   "text": "..."}     Final transcript
    ← {"type": "stt_error",   "error": "..."}    Error
"""

from __future__ import annotations

import asyncio
import io
import json
import queue
import struct
import threading
from typing import TYPE_CHECKING, Any

import aiohttp
from aiohttp import web

from captain_claw.logging import get_logger

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer

log = get_logger(__name__)


async def handle_stt_ws(
    server: "WebServer", request: web.Request
) -> web.WebSocketResponse:
    """WebSocket endpoint ``/ws/stt`` — live audio transcription."""
    ws = web.WebSocketResponse(max_msg_size=10 * 1024 * 1024)
    await ws.prepare(request)

    # Detect whether Soniox realtime is available.
    from captain_claw.tools.stt import _HAS_SONIOX, _resolve_stt_api_key

    soniox_key = _resolve_stt_api_key("soniox")
    use_realtime = bool(soniox_key and _HAS_SONIOX)

    await ws.send_json({"type": "stt_ready", "realtime": use_realtime})
    log.info("stt_ws_connected", realtime=use_realtime)

    try:
        if use_realtime:
            await _handle_realtime(ws, soniox_key)
        else:
            await _handle_batch(ws)
    except Exception as exc:
        log.error("stt_ws_error", error=str(exc))
        if not ws.closed:
            await ws.send_json({"type": "stt_error", "error": str(exc)})
    finally:
        if not ws.closed:
            await ws.close()

    return ws


# ---------------------------------------------------------------------------
# Realtime mode (Soniox streaming)
# ---------------------------------------------------------------------------


async def _handle_realtime(ws: web.WebSocketResponse, api_key: str) -> None:
    """Stream browser audio → Soniox realtime → partial results back to WS."""
    audio_q: queue.Queue[bytes | None] = queue.Queue()
    stop_event = threading.Event()
    loop = asyncio.get_event_loop()

    def _soniox_worker() -> None:
        """Run Soniox realtime session in a background thread."""
        from soniox import SonioxClient
        from soniox.types import RealtimeSTTConfig
        from soniox.utils import render_tokens, start_audio_thread

        client = SonioxClient(api_key=api_key)
        config = RealtimeSTTConfig(
            model="stt-rt-v4",
            audio_format="pcm_s16le",
            sample_rate=16000,
            num_channels=1,
        )

        def _audio_iter():
            """Yield PCM chunks from the queue until stop."""
            while not stop_event.is_set():
                try:
                    chunk = audio_q.get(timeout=0.2)
                    if chunk is None:
                        break
                    yield chunk
                except queue.Empty:
                    continue

        final_tokens: list[Any] = []
        non_final_tokens: list[Any] = []

        try:
            with client.realtime.stt.connect(config=config) as session:
                start_audio_thread(session, _audio_iter())

                for event in session.receive_events():
                    non_final_tokens.clear()
                    for token in event.tokens:
                        if token.is_final:
                            final_tokens.append(token)
                        else:
                            non_final_tokens.append(token)

                    text = render_tokens(final_tokens, non_final_tokens).strip()
                    if text and not ws.closed:
                        asyncio.run_coroutine_threadsafe(
                            ws.send_json({"type": "stt_partial", "text": text}),
                            loop,
                        )

            # Session closed — send final result.
            final_text = render_tokens(final_tokens, []).strip()
            if not ws.closed:
                asyncio.run_coroutine_threadsafe(
                    ws.send_json({"type": "stt_final", "text": final_text}),
                    loop,
                )

        except Exception as exc:
            log.error("soniox_realtime_error", error=str(exc))
            if not ws.closed:
                asyncio.run_coroutine_threadsafe(
                    ws.send_json(
                        {"type": "stt_error", "error": f"Realtime STT error: {exc}"}
                    ),
                    loop,
                )

    # Start Soniox worker in a thread.
    worker_future = loop.run_in_executor(None, _soniox_worker)

    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.BINARY:
                audio_q.put(msg.data)
            elif msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(msg.data)
                if data.get("type") == "stop":
                    break
            elif msg.type in (
                aiohttp.WSMsgType.ERROR,
                aiohttp.WSMsgType.CLOSE,
            ):
                break
    finally:
        stop_event.set()
        audio_q.put(None)  # unblock _audio_iter
        await worker_future


# ---------------------------------------------------------------------------
# Batch mode (fallback for non-Soniox providers)
# ---------------------------------------------------------------------------


async def _handle_batch(ws: web.WebSocketResponse) -> None:
    """Collect all audio, then batch-transcribe when user stops."""
    audio_chunks: list[bytes] = []

    async for msg in ws:
        if msg.type == aiohttp.WSMsgType.BINARY:
            audio_chunks.append(msg.data)
        elif msg.type == aiohttp.WSMsgType.TEXT:
            data = json.loads(msg.data)
            if data.get("type") == "stop":
                break
        elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSE):
            break

    if not audio_chunks:
        await ws.send_json({"type": "stt_final", "text": ""})
        return

    # Combine raw PCM chunks → WAV → transcribe.
    pcm_data = b"".join(audio_chunks)
    wav_bytes = _pcm_to_wav(pcm_data, sample_rate=16000, channels=1)

    log.info("stt_batch_transcribe", pcm_kb=f"{len(pcm_data)/1024:.1f}")

    from captain_claw.tools.stt import transcribe_audio

    text = await transcribe_audio(wav_bytes)
    await ws.send_json({"type": "stt_final", "text": text})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pcm_to_wav(
    pcm_bytes: bytes, sample_rate: int = 16000, channels: int = 1
) -> bytes:
    """Wrap raw PCM int16 bytes in a WAV header."""
    data_size = len(pcm_bytes)
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))
    buf.write(struct.pack("<H", 1))  # PCM
    buf.write(struct.pack("<H", channels))
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", sample_rate * channels * 2))
    buf.write(struct.pack("<H", channels * 2))
    buf.write(struct.pack("<H", 16))  # bits per sample
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(pcm_bytes)
    return buf.getvalue()
