# Summary: browser_vision.py

# browser_vision.py Summary

**Summary:**
This module provides vision-capable LLM integration for analyzing browser screenshots in the Captain Claw automation framework. It handles screenshot preprocessing (resizing, JPEG compression), model selection from configured vision/OCR/multimodal models, and sends images to LLMs via LiteLLM with support for both standard chat completions and OpenAI's ChatGPT Responses API.

**Purpose:**
Solves the problem of programmatic web page understanding by converting raw browser screenshots into semantic descriptions. Serves as a visual complement to accessibility tree parsing, enabling the automation system to understand page layout, UI elements, forms, data displays, and interactive components through vision-based analysis rather than DOM inspection alone.

**Most Important Functions/Classes:**

1. **`BrowserVision.analyze_screenshot(screenshot_bytes, prompt, max_pixels, jpeg_quality)`**
   - Core async method that orchestrates the entire vision analysis pipeline. Accepts raw screenshot bytes, resizes/compresses them, base64-encodes the image, routes to appropriate LLM endpoint (ChatGPT Responses API for OpenAI+bearer auth, standard LiteLLM for others), and returns extracted text description. Includes fallback to empty string if no vision model configured.

2. **`BrowserVision.find_vision_model()`**
   - Static method that searches the configuration's allowed models list for a vision-capable model, checking model_type in priority order: "vision" → "ocr" → "multimodal". Returns the matching `AllowedModelConfig` or None, enabling flexible model selection without hardcoding.

3. **`_resize_screenshot(image_bytes, max_pixels, jpeg_quality)`**
   - Helper function that conditionally resizes screenshots using Pillow to stay within pixel constraints while maintaining aspect ratio. Converts RGBA/palette images to RGB, compresses to JPEG at specified quality, and gracefully degrades if Pillow unavailable. Returns tuple of (processed_bytes, mime_type).

**Architecture & Dependencies:**
- **External:** LiteLLM for LLM communication, Pillow (optional) for image processing, asyncio for async operations
- **Internal:** Depends on `captain_claw.config` (model/tool configuration), `captain_claw.logging` (structured logging), `captain_claw.llm` (provider normalization, API key resolution), `captain_claw.tools.image_ocr` (ChatGPT Responses API handler for OpenAI)
- **Design Pattern:** Mirrors `image_ocr.py`'s architecture for consistency; uses optional dependency pattern for Pillow with graceful degradation; supports multiple auth mechanisms (API keys vs. bearer tokens/OAuth headers)