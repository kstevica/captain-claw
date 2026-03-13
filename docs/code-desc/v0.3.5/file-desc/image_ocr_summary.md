# Summary: image_ocr.py

# Image OCR and Vision Analysis Module

## Summary

This module provides OCR and vision analysis capabilities by routing images to multimodal LLMs (Large Language Models). It implements intelligent image resizing strategies (Pillow → ImageMagick → raw fallback), supports both standard OpenAI API and ChatGPT Responses API (for bearer/OAuth auth), and exposes two concrete tools: `ImageOcrTool` for text extraction and `ImageVisionTool` for general image analysis. The architecture uses a base class pattern to minimize code duplication while supporting flexible model selection and configuration.

## Purpose

Solves the problem of extracting text from images (OCR) and analyzing visual content through vision-capable LLMs. Handles image preprocessing (resizing to optimize token usage and API limits), authentication routing (API keys vs. bearer tokens), and response parsing across different LLM providers. Designed for integration into an agent/tool system where autonomous workflows need to process images.

## Most Important Functions/Classes

### 1. **`_BaseImageLLMTool` (class)**
   - **Description**: Abstract base class for vision-capable tools. Implements the complete execution pipeline: file validation, model selection, image preprocessing, LLM routing, and response extraction.
   - **Key responsibilities**: Configuration resolution, model discovery, image encoding/resizing orchestration, dual API routing (ChatGPT Responses vs. litellm), response parsing across different formats.
   - **Role**: Core architecture pattern—eliminates duplication between `ImageOcrTool` and `ImageVisionTool` by centralizing all vision logic.

### 2. **`_maybe_resize_image()` (function)**
   - **Description**: Orchestrates image resizing with fallback strategy: attempts Pillow first, then ImageMagick CLI, returns raw bytes if both fail.
   - **Key responsibilities**: Determines if resizing is needed (max_pixels threshold), delegates to appropriate backend, logs size reduction metrics.
   - **Role**: Optimizes API costs and token usage by reducing image dimensions while maintaining quality (JPEG compression with configurable quality).

### 3. **`_chatgpt_responses_vision()` (async function)**
   - **Description**: Direct HTTP client for ChatGPT Responses API (non-standard OpenAI endpoint). Handles SSE (Server-Sent Events) streaming, parses delta events and completion events, extracts text content.
   - **Key responsibilities**: Payload construction with multimodal input (text + base64 image), bearer token header injection, streaming response parsing, event type discrimination.
   - **Role**: Enables authentication via OAuth/bearer tokens (Codex auth) which standard OpenAI API doesn't support—critical for enterprise/restricted environments.

### 4. **`ImageOcrTool` (class)**
   - **Description**: Concrete tool for text extraction from images. Inherits all execution logic from `_BaseImageLLMTool`, configures OCR-specific defaults.
   - **Configuration**: Searches for models with `model_type: "ocr"` or `"vision"` (in priority order), default prompt: "Extract all text from this image."
   - **Role**: Exposes OCR as a callable tool in agent workflows with standardized parameter schema.

### 5. **`ImageVisionTool` (class)**
   - **Description**: Concrete tool for general image analysis and description. Inherits execution logic, configures vision-specific defaults.
   - **Configuration**: Searches for models with `model_type: "vision"` or `"ocr"` (reversed priority), default prompt: "Describe this image in detail."
   - **Role**: Enables flexible image understanding tasks (object detection, scene description, chart analysis, visual Q&A) in agent workflows.

## Architecture & Dependencies

**Key Dependencies**:
- `litellm`: Unified LLM API abstraction (handles provider normalization, model routing)
- `httpx`: Async HTTP client for ChatGPT Responses API streaming
- `PIL (Pillow)`: Optional image processing (graceful degradation if missing)
- `ImageMagick` (convert/magick CLI): Fallback image resizing via subprocess
- `captain_claw.config`: Configuration system (model selection, tool settings, API keys)
- `captain_claw.llm`: Provider/model utilities (normalization, API key resolution)
- `captain_claw.tools.registry`: Tool base class and result wrapper

**Design Patterns**:
- **Template Method**: `_BaseImageLLMTool.execute()` defines the complete workflow; subclasses only override configuration attributes.
- **Strategy Pattern**: Image resizing uses pluggable backends (Pillow → ImageMagick → none) with graceful fallback.
- **Conditional Routing**: Provider + auth detection determines API path (ChatGPT Responses API vs. litellm Chat Completions).
- **Configuration-Driven**: Model selection, timeouts, image quality, max characters all externalized to config system.

**Supported Image Formats**: `.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`, `.bmp`

**Configuration Keys**: `image_ocr` and `image_vision` (separate tool configs with `timeout_seconds`, `max_pixels`, `jpeg_quality`, `max_chars`, `default_prompt`)