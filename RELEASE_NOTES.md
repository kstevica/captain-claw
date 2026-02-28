# Captain Claw v0.2.5 Release Notes

**Release date:** 2026-02-28

## Highlights

Per-user Telegram sessions with concurrent agent execution, image generation and vision tools, and datastore table export.

## New Features

### Telegram: Per-User Sessions & Concurrent Agents
- Each Telegram user now gets a dedicated Agent instance with its own session
- Different Telegram users can interact concurrently (no more "Agent is busy" blocking between users)
- Sessions are isolated: users cannot see or switch to other users' sessions
- `/new` and session switching commands are disabled on Telegram
- `/clear`, `/history`, `/compact`, `/session` operate on the user's own session
- Per-user asyncio locks serialise requests from the same user while allowing parallelism across users
- Agent instances follow the lightweight AgentPool pattern: shared provider and tools, independent session and state

### Image Generation Tool
- New `image_gen` tool for AI-powered image generation
- Supports multiple providers via config
- Generated images are saved to the session media folder
- Generated images are automatically sent back to Telegram users

### Image OCR Tool
- New `image_ocr` tool for extracting text from images
- Supports local files and URLs

### Image Vision Tool
- New `image_vision` tool for visual analysis and understanding of images
- Telegram photo attachments are now processed through the vision pipeline

### Datastore Table Export
- Datastore tables can now be exported via the web UI
- New REST endpoint for datastore data export

### Web UI: Image Upload
- New REST endpoint for uploading images directly through the web UI

## Fixes

- Fixed LLM image generation provider integration
- Fixed Telegram handling of image attachments (`image_*` tools)
- Fixed scale loop passthrough for orchestrator workers
- Disabled Vertex AI provider (stability issues)

## Internal

- New `platform_adapter` functions for collecting generated image paths from session turns
- Telegram bridge extended with photo sending and file download capabilities
- Config extended with image generation settings
