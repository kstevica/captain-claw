# Summary: pocket_tts.py

# Pocket TTS Tool Summary

## Overview
A local text-to-speech synthesis tool that converts text to MP3 audio files using the pocket-tts model. Handles model loading, voice selection, audio format conversion, and MP3 encoding with fallback mechanisms for encoding dependencies.

## Purpose
Solves the problem of generating high-quality speech audio locally without external API calls. Provides flexible voice selection (8 predefined voices or custom .wav files), configurable sample rates and bitrates, and safe file output management under a session-based directory structure.

## Architecture & Dependencies

**Key Dependencies:**
- `pocket_tts` — ML model for speech synthesis
- `lameenc` (optional) — MP3 encoding; falls back to `ffmpeg` if unavailable
- `captain_claw.tools.registry` — Tool framework integration
- `captain_claw.config` — Configuration management
- `asyncio` — Async execution for long-running synthesis operations

**System Role:**
Implements the `Tool` interface as a registered capability within the captain_claw framework. Operates asynchronously to prevent blocking during model inference and file I/O. Integrates with WriteTool for safe path resolution under saved media directories.

---

## Most Important Functions/Classes

### 1. **PocketTTSTool.execute()**
Main entry point for text-to-speech synthesis. Validates input text length against config limits, resolves output file path, loads/caches the model, selects voice, generates audio, converts to PCM16, encodes to MP3, and writes to disk. Returns detailed ToolResult with metadata (duration, bitrate, sample rate).

### 2. **PocketTTSTool._ensure_model()**
Lazy-loads and caches the pocket-tts TTSModel on first use. Handles import errors gracefully with helpful installation instructions. Prevents redundant model loading across multiple synthesis calls. Stores error state to avoid repeated failed load attempts.

### 3. **_encode_mp3_bytes()**
Converts PCM16 audio bytes to MP3 format. Attempts lameenc first (faster, pure Python), falls back to ffmpeg subprocess if unavailable. Validates encoding success and provides detailed error messages with installation guidance for missing dependencies.

### 4. **_coerce_samples()**
Normalizes arbitrary model output (tensors, nested lists, GPU tensors) into a flat list of float samples. Handles PyTorch tensors by detaching and moving to CPU, reshapes multidimensional arrays, and gracefully degrades for unexpected types.

### 5. **_samples_to_pcm16()**
Converts float audio samples to signed 16-bit PCM bytes. Scales samples intelligently (assumes normalized [-1, 1] range but handles larger values), clips to valid int16 range [-32768, 32767], and packs into binary format for encoding.

---

## Configuration Integration
Reads from `cfg.tools.pocket_tts`:
- `timeout_seconds` — Execution timeout (default 600s)
- `default_voice` — Fallback voice if none specified
- `max_chars` — Input text length limit (default 4000)
- `sample_rate` — Audio sample rate override (default 24000 Hz)
- `mp3_bitrate_kbps` — MP3 encoding bitrate (default 128 kbps)

## Error Handling Strategy
- Graceful voice fallback: If requested voice fails, retries with default voice
- Dual encoding paths: lameenc → ffmpeg → detailed error with installation instructions
- Input validation: Text length checks, empty audio detection
- Async safety: All blocking operations (model inference, file I/O, encoding) run in thread pool