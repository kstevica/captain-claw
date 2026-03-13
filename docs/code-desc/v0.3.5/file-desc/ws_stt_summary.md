# Summary: ws_stt.py

# ws_stt.py Summary

This module implements a WebSocket handler for live speech-to-text transcription from browser microphone input, supporting both real-time streaming (via Soniox API) and batch processing fallback modes. The system streams raw PCM int16 audio at 16 kHz from the browser and either forwards it to Soniox for real-time partial/final transcription results or collects it for batch processing when the user stops recording.

## Purpose

Solves the problem of capturing live microphone audio from a web browser and converting it to text transcriptions with minimal latency. Provides dual-mode operation: real-time streaming transcription when Soniox is available (with partial results streamed back as they arrive), and graceful fallback to batch transcription using alternative STT providers when real-time capability is unavailable.

## Most Important Functions/Classes/Procedures

1. **`handle_stt_ws(server, request)`** — Main WebSocket endpoint handler (`/ws/stt`). Detects Soniox availability, sends readiness status to client, routes to appropriate handler (realtime or batch), manages error handling and WebSocket lifecycle. Acts as the entry point and orchestrator for the entire STT flow.

2. **`_handle_realtime(ws, api_key)`** — Manages real-time streaming transcription via Soniox. Spawns a background worker thread running the Soniox client, maintains an audio queue for thread-safe communication, receives partial and final transcription events, and streams results back to the WebSocket client in real-time. Handles graceful shutdown via stop events.

3. **`_soniox_worker()`** — Background thread function that establishes a Soniox realtime STT session, consumes PCM audio from a queue via `_audio_iter()`, processes token events (distinguishing final vs. non-final tokens), renders transcription text, and sends results back to the WebSocket via `asyncio.run_coroutine_threadsafe()` for thread-safe async communication.

4. **`_handle_batch(ws)`** — Fallback batch transcription handler. Collects all binary audio chunks from the WebSocket until the user sends a "stop" message, combines them into a single PCM stream, converts to WAV format, and calls `transcribe_audio()` for batch processing before sending the final result.

5. **`_pcm_to_wav(pcm_bytes, sample_rate, channels)`** — Utility function that wraps raw PCM int16 audio bytes in a valid WAV file header (RIFF/WAVE format) with proper metadata (sample rate, channels, bit depth). Enables batch mode to work with STT providers expecting WAV input.

## Architecture & Dependencies

**Key Dependencies:**
- `aiohttp` — WebSocket server framework and async HTTP client
- `soniox` — Real-time STT API client (optional; gracefully degraded if unavailable)
- `asyncio` — Async event loop management and thread-safe coroutine execution
- `threading` — Background worker thread for blocking Soniox operations
- `queue` — Thread-safe audio chunk buffering between async and sync contexts
- `struct` — Binary WAV header encoding
- `captain_claw.tools.stt` — Batch transcription backend and API key resolution

**Architecture Pattern:**
- **Dual-mode design**: Real-time path uses thread + async bridge; batch path is pure async
- **Thread-safe communication**: `queue.Queue` for audio, `asyncio.run_coroutine_threadsafe()` for results
- **Protocol**: Binary frames for audio, JSON for control messages and results
- **Error isolation**: Exceptions caught and reported via WebSocket without crashing the handler