# Summary: rest_visualization_style.py

# rest_visualization_style.py Summary

## Summary
REST API handler module for managing visualization style profiles in the Captain Claw system. Provides endpoints to view, update, and intelligently extract visual style information (colors, fonts, layout rules) from uploaded images and documents using LLM vision capabilities and document parsing.

## Purpose
Solves the problem of maintaining consistent visual styling across AI-generated data visualizations and dashboards. Enables users to upload design references (images, PDFs, DOCX files), automatically extract style parameters via LLM analysis, update style configurations, and refine style descriptions through AI rephrasing—creating a centralized style management system that propagates changes to all dependent agents and instruction caches.

## Most Important Functions/Classes/Procedures

1. **`get_visualization_style(server, request)`**
   - GET endpoint returning current visualization style as JSON object
   - Loads style from persistent storage and serializes to response
   - Entry point for retrieving active style configuration

2. **`put_visualization_style(server, request)`**
   - PUT endpoint for updating style fields via JSON body (partial updates supported)
   - Merges provided fields into existing style, saves to disk, clears instruction caches
   - Triggers cache invalidation so all agents rebuild prompts with new styling

3. **`analyze_visualization_style(server, request)`**
   - POST endpoint accepting multipart file uploads (images: PNG/JPG/WEBP; documents: PDF/DOCX)
   - Routes to specialized analyzers (`_analyze_image` or `_analyze_document`)
   - Returns extracted style JSON with color palettes, fonts, layout rules, and design notes

4. **`_analyze_image(file_bytes, ext, server)`**
   - Leverages vision-capable LLM models to analyze design reference images
   - Handles image resizing/optimization, base64 encoding, and litellm API integration
   - Parses LLM response into structured style JSON with fallback to text-only analysis

5. **`_analyze_document(file_bytes, ext, server)`**
   - Extracts text from PDF/DOCX documents using `document_extract` tools
   - Sends extracted content + analysis prompt to main LLM provider
   - Handles temporary file management and content truncation for large documents

6. **`rephrase_visualization_style(server, request)`**
   - POST endpoint for AI-powered refinement of style field descriptions
   - Accepts field name and text, returns rephrased content optimized for AI chart generation
   - Uses design systems expert system prompt to improve clarity and actionability

7. **`_merge_style_fields(s, body)`**
   - Helper applying JSON body fields to VisualizationStyle object in-place
   - Handles type coercion and normalization (color palette parsing, string trimming)
   - Supports flexible input formats (lists, comma-separated strings)

8. **`_parse_style_json(raw)`**
   - Robust JSON extraction from LLM responses with markdown fence stripping
   - Implements fallback regex-based JSON object detection
   - Normalizes color_palette to list of strings; returns empty result on parse failure

9. **`_clear_instruction_caches(server)`**
   - Clears cached system prompts from main agent and Telegram agents
   - Forces rebuild of prompts with updated visualization style on next use
   - Critical for propagating style changes across all agent instances

## Architecture & Dependencies

**Key Dependencies:**
- `aiohttp.web` — async HTTP request/response handling
- `litellm` — unified LLM API abstraction for vision models
- `captain_claw.visualization_style` — style persistence and serialization
- `captain_claw.tools.image_ocr` — image resizing/optimization
- `captain_claw.tools.document_extract` — PDF/DOCX text extraction
- `captain_claw.llm` — provider abstraction, message handling

**System Role:**
- Sits in REST API layer, bridging user uploads/updates with core style management
- Integrates with LLM providers (vision + text) for intelligent style extraction
- Maintains cache invalidation contracts with agent instruction system
- Supports multipart file handling with validation and size limits

**Data Flow:**
1. User uploads design reference → multipart parsing → file type validation
2. Image/document → specialized analyzer → LLM processing → JSON extraction
3. Style updates → merge into model → persist → clear caches → rebuild prompts
4. Style rephrasing → LLM refinement with design expert context → return improved text