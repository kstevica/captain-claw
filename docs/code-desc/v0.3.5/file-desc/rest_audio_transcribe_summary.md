# Summary: rest_audio_transcribe.py

# rest_audio_transcribe.py Summary

**Summary:**
This module implements a REST API endpoint for browser-based audio transcription, accepting multipart form uploads of browser-recorded audio (typically WebM/Opus from MediaRecorder) and returning transcribed text. It handles audio format conversion to WAV via ffmpeg and integrates with the existing STT (speech-to-text) pipeline, with comprehensive error handling and size validation.

**Purpose:**
Solves the problem of capturing voice input directly from web browsers and converting it to text through a unified REST interface. Bridges the gap between browser audio recording capabilities (MediaRecorder API) and backend speech-to-text processing, handling format incompatibilities and providing a clean JSON API for frontend consumption.

**Most Important Functions/Classes:**

1. **`transcribe_audio_handler(server, request)`**
   - Main async REST handler for POST `/api/audio/transcribe` endpoint
   - Parses multipart form data, extracts audio field, validates size (max 10 MB), converts format to WAV, invokes STT pipeline, and returns JSON response with transcribed text or error details
   - Handles multiple error scenarios (missing field, empty audio, oversized uploads, STT provider errors) with appropriate HTTP status codes

2. **`_ensure_wav(audio_bytes)`**
   - Utility function that detects WAV format by magic bytes (RIFF/WAVE headers) and converts non-WAV audio to 16 kHz mono PCM WAV format
   - Uses ffmpeg subprocess for format conversion with graceful fallback if ffmpeg unavailable
   - Logs conversion metrics and handles ffmpeg errors with detailed error reporting

3. **Error Handling & Validation Layer**
   - Multipart parsing with field validation (checks for "audio" field presence)
   - Size guard against oversized uploads (10 MB limit)
   - Content-Type logging for debugging
   - Structured exception handling distinguishing between STT provider errors (503), HTTP exceptions (re-raised), and generic failures (500)

**Architecture & Dependencies:**
- Built on aiohttp async web framework for non-blocking I/O
- Depends on `captain_claw.tools.stt.transcribe_audio()` for actual speech-to-text processing
- Uses ffmpeg subprocess for audio format conversion (optional but recommended)
- Integrates with captain_claw logging system for structured observability
- Follows async/await pattern throughout for scalability