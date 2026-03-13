# Summary: image_gen.py

# image_gen.py Summary

**Summary:**
ImageGenTool is an async-capable tool that generates images from text prompts using LiteLLM, supporting multiple image generation providers (DALL-E 3, etc.). It handles image retrieval via base64 encoding or URL download, persists images to the local filesystem under a session-organized directory structure, and registers generated files with a file registry for web UI serving.

**Purpose:**
Solves the problem of integrating AI image generation capabilities into an agent/tool system by abstracting provider differences, managing file persistence with security constraints, handling both base64 and URL-based image responses, and providing a standardized interface for prompt-based image creation with configurable dimensions and quality settings.

**Most Important Functions/Classes/Procedures:**

1. **`ImageGenTool` (class)** - Main tool class inheriting from `Tool` base class. Defines the tool's name, timeout, description, and JSON schema for parameters (prompt, size, quality, output_path). Implements the core `execute()` async method that orchestrates the entire image generation workflow.

2. **`execute()` (async method)** - Primary orchestration method that: validates the prompt, locates the configured image model, resolves safe output paths, calls LiteLLM's image generation API with provider-specific parameters, extracts image data from responses (handling both base64 and URL formats), downloads/decodes images, persists to disk, registers with file registry, and returns detailed success/error results with metadata.

3. **`_resolve_output_path()` (method)** - Security-critical path resolution that normalizes user-provided output paths under the `saved/media/` directory structure, prevents directory traversal attacks via `WriteTool._normalize_under_saved()`, auto-generates timestamped filenames if none provided, and enforces PNG/JPG/JPEG/WebP file extensions.

4. **`_find_image_model()` (static method)** - Configuration utility that searches the allowed models list for the first model with `model_type == 'image'`, enabling dynamic provider selection without hardcoding specific model names.

5. **Response handling logic** (in execute) - Flexible extraction from LiteLLM responses that handles both object attributes and dictionary access patterns, prioritizes base64 data (avoiding URL expiration issues), falls back to async HTTP retrieval for URL-based responses, and captures revised prompts from providers that modify user input.

**Architecture & Dependencies:**
- Depends on `captain_claw.config` (configuration management), `captain_claw.logging` (structured logging), `captain_claw.tools.registry` (Tool base class), `captain_claw.tools.write` (path normalization utilities), and `litellm` (multi-provider image generation abstraction)
- Uses `httpx.AsyncClient` for async image URL downloads
- Integrates with file registry system for web UI asset serving
- Respects session isolation via `_session_id` parameter
- Supports configurable timeouts and provider-specific parameters (base_url, api_key)