# Captain Claw v0.3.0 Release Notes

**Release title:** Chunked Processing, Personality System & Termux

**Release date:** 2026-03-01

## Highlights

Major feature release — small-context models (20k-32k tokens) can now handle arbitrarily large content via automatic chunked processing, a dual-profile personality system tailors responses per user, and the new Termux tool brings Captain Claw to Android devices. Plus: web UI authentication, XLSX upload to datastore, and image auto-resize before vision calls.

## New Features

### Chunked Processing Pipeline
- Enables small-context models (20k-32k tokens) to process content that far exceeds their context window
- Context budget guard automatically detects when content exceeds available space and triggers chunking
- Map phase splits content into sequential overlapping chunks and processes each with full instructions via isolated LLM calls
- Reduce phase combines partial results via LLM synthesis (summarize) or simple join (concatenate)
- Hooks into both the normal conversation flow (tool result interception) and the scale loop micro-loop
- Content-extraction tools automatically chunked: `read`, `pdf_extract`, `docx_extract`, `xlsx_extract`, `pptx_extract`, `web_fetch`, `web_get`
- Configurable: auto-threshold, output reserve, chunk overlap, max chunks, and combine strategy
- New settings UI section under Model & LLM for all chunked processing parameters
- Tested: 108k token file split into 9 chunks, processed in 48.5 seconds, reduced to 5.7k tokens

### Personality System
- Dual-profile architecture: global agent identity (name, background, expertise) plus per-user profiles
- Per-user profiles tailor responses to each user's perspective and preferences
- New `personality` tool for reading and updating profiles from conversation
- REST API endpoints for personality management
- Settings page integration for editing agent and user profiles from the web UI
- Telegram users get automatic per-user profiles

### Termux Tool (Android)
- Run Captain Claw on Android via Termux
- Take photos with front/back camera (auto-sent to Telegram)
- Get GPS location, check battery status, toggle flashlight
- All operations via Termux API integration

### Web UI Authentication
- Optional authentication layer for the web UI
- Protects web interface when exposed on a network

### XLSX Upload to Datastore
- Upload Excel files directly to datastore tables from the web dashboard
- Automatic column detection and data import

### Image Auto-Resize
- Images are automatically resized before being sent to vision-capable LLMs
- Reduces token usage and speeds up processing for large images

## Improvements

### Gemini Timeout Handling
- Added 180-second timeout wrapper for Gemini sync completion calls
- Previously the agent could hang indefinitely if the Gemini API was slow or unresponsive
- Both `asyncio.wait_for` and litellm `timeout` kwarg applied

### Scale Loop Enhancements
- Improved micro-loop integration with chunked processing support
- Better handling of large content items in multi-step processing

### Mobile Usage
- Updated usage instructions with mobile-specific guidance

## Internal

- New `agent_chunked_processing_mixin.py` (837 lines) — full chunked processing pipeline
- New `personality.py` — personality profile management
- New `tools/personality.py` — personality tool implementation
- New `tools/termux.py` — Termux API tool
- New `web/auth.py` — web authentication middleware
- New `web/rest_personality.py` — personality REST endpoints
- New `web/rest_file_upload.py` — file upload handling for datastore
- Updated `agent.py` — added `AgentChunkedProcessingMixin` to Agent class MRO
- Updated `agent_tool_loop_mixin.py` — chunked processing hook in `_handle_tool_calls()`
- Updated `agent_scale_loop_mixin.py` — scale loop micro-loop chunked processing integration
- Updated `config.py` — added `ChunkedProcessingConfig` with 6 settings
- Updated `llm/__init__.py` — Gemini timeout fix
- Updated `web/rest_settings.py` — chunked processing settings UI section
- Updated `README.md` and `USAGE.md` — documentation for all new features
- 43 files changed, 4,035 insertions, 88 deletions
