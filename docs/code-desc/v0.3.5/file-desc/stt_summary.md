# Summary: stt.py

# stt.py Summary

Comprehensive speech-to-text module providing audio recording, real-time transcription, and multi-provider STT support with intelligent provider resolution. Handles microphone input via sounddevice, WAV encoding, and integrates with Soniox (real-time WebSocket), OpenAI Whisper, and Gemini multimodal APIs with fallback logic.

## Purpose

Solves the problem of converting spoken audio to text through multiple transcription backends, enabling flexible deployment across different STT providers while maintaining a unified interface. Supports both batch transcription (file upload) and real-time streaming (Soniox WebSocket), with zero temporary file creation.

## Architecture & Dependencies

**External Dependencies:**
- `sounddevice` + `numpy` (optional): microphone audio capture
- `soniox` (optional): real-time STT via WebSocket
- `aiohttp`: HTTP client for OpenAI API calls
- `litellm`: Gemini multimodal API wrapper
- `structlog`: structured logging

**Internal Dependencies:**
- `captain_claw.config`: configuration resolution
- `captain_claw.llm`: API key fallback resolution

**Key Design Patterns:**
- Optional dependency guards (`_HAS_AUDIO`, `_HAS_SONIOX`) prevent import failures
- Provider resolution cascade: explicit config → Soniox → OpenAI → Gemini → model.allowed
- Synchronous blocking functions wrapped for async via `asyncio.to_thread()`
- In-memory WAV encoding (no temp files)
- Async/await throughout transcription layer

## Most Important Functions/Classes

1. **`transcribe_audio(wav_bytes: bytes) → str`** (async)
   - Main entry point for batch transcription. Implements provider resolution cascade (explicit config → Soniox → OpenAI → Gemini → model.allowed). Raises RuntimeError if no provider available.

2. **`realtime_stt_sync(api_key, stop_event, ...) → str`** (sync, meant for `asyncio.to_thread`)
   - Streams live microphone audio directly to Soniox via WebSocket without creating files. Uses threading.Event for stop signaling and time-based deadline. Returns final transcribed text. Handles background audio thread and token accumulation.

3. **`_record_audio_sync(duration_limit, sample_rate, channels, stop_event) → bytes`** (sync)
   - Captures raw PCM int16 audio from default microphone using sounddevice callback pattern. Returns WAV-encoded bytes. Blocks until stop_event or timeout. Requires numpy/sounddevice.

4. **`_encode_wav(samples, sample_rate, channels) → bytes`**
   - Encodes numpy int16 array as WAV bytes using struct packing. Constructs RIFF header, fmt chunk, and data chunk in-memory. No file I/O.

5. **`_transcribe_soniox(api_key, wav_bytes) → str`** (async)
   - Calls AsyncSonioxClient.stt.transcribe() for file upload, polls wait() until completion, fetches TranscriptionTranscript separately (text not in Transcription object). Logs job ID and audio duration.

6. **`_transcribe_openai_whisper(api_key, wav_bytes, model, base_url) → str`** (async)
   - Posts multipart form to OpenAI-compatible `/v1/audio/transcriptions` endpoint. Supports custom base_url for self-hosted Whisper. Uses aiohttp with 60s timeout.

7. **`_transcribe_gemini(api_key, wav_bytes) → str`** (async)
   - Base64-encodes WAV, calls litellm.completion with gemini-2.0-flash model using audio_url content type. Offloads to thread via `asyncio.to_thread()`. Extracts text from response.choices[0].message.content.

8. **`_resolve_stt_api_key(provider: str) → str`**
   - Maps provider name to environment variable (SONIOX_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY). Falls back to captain_claw.llm._resolve_api_key for openai/gemini. Returns empty string if not found.

9. **`_transcribe_with_provider(provider, wav_bytes) → str`** (async)
   - Routes to specific provider by name string. Validates API key presence and package availability (e.g., soniox installed). Handles provider-specific model/base_url config.

10. **`_transcribe_with_model(model_cfg, wav_bytes) → str`** (async)
    - Transcribes using explicitly configured STT model from config.model.allowed. Extracts provider, base_url, model name from config object. Delegates to provider-specific function.