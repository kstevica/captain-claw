# Summary: visualization_style.py

# visualization_style.py Summary

**Summary:**
This module manages visualization style profiles for the Captain Claw system, providing persistent storage and in-memory caching of design preferences (colors, fonts, backgrounds, chart styles). It handles serialization/deserialization between Markdown files, Python dataclasses, dictionaries, and system prompt injection formats, with automatic cache invalidation based on file modification timestamps.

**Purpose:**
Solves the problem of maintaining consistent visual styling across generated outputs (HTML charts, dashboards, DOCX, PPTX documents) by centralizing style configuration in a user-editable Markdown file (`~/.captain-claw/visualization_style.md`) while minimizing disk I/O through intelligent caching and providing multiple format conversions for different consumption contexts (LLM prompts, JSON APIs, file storage).

**Most Important Functions/Classes/Procedures:**

1. **`VisualizationStyle` (dataclass)**
   - Core data model storing all style attributes: name, color_palette, font_primary, font_headings, font_mono, background_style, chart_style, layout_notes, additional_rules, source_description. Provides a single source of truth for style configuration with sensible defaults.

2. **`load_visualization_style()`**
   - Implements intelligent file I/O with modification-time-based caching. Returns cached style if file unchanged, loads from disk on modification, gracefully falls back to default style if file missing or unreadable. Critical for performance in high-frequency operations.

3. **`parse_visualization_style_markdown(text: str) -> VisualizationStyle`**
   - Parses Markdown section format (`# Section Name`) into structured VisualizationStyle object. Handles color palette as bullet lists and normalizes all string fields. Silently ignores unrecognized sections for forward compatibility.

4. **`visualization_style_to_prompt_block(s: VisualizationStyle) -> str`**
   - Converts style configuration into a formatted text block for LLM system prompt injection. Escapes curly braces to prevent `str.format_map()` conflicts and returns empty string when no meaningful style exists to keep prompts clean.

5. **`visualization_style_to_markdown(s: VisualizationStyle) -> str` & `save_visualization_style(s: VisualizationStyle)`**
   - Bidirectional serialization: converts dataclass to canonical Markdown format and persists to disk with cache invalidation. Ensures file directory exists and updates global cache state after write.

**Architecture Notes:**
- **Caching Strategy:** Global module-level variables (`_cached_style`, `_cached_mtime`) track loaded state; comparison of file modification times determines whether reload is necessary
- **Format Flexibility:** Supports three serialization formats (Markdown, dict/JSON, prompt text) enabling integration with file storage, REST APIs, and LLM systems
- **Robustness:** Graceful degradation on file I/O errors; missing files default to empty style; unrecognized Markdown sections ignored
- **Dependencies:** Minimal—only standard library (dataclasses, pathlib, re, typing)
- **Role in System:** Configuration management layer that decouples style definition from output generation, allowing non-technical users to customize visual appearance via Markdown while providing programmatic access for code-based styling