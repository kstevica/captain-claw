# Summary: telegram_bridge.py

# Telegram Bridge Summary

**Summary:** A minimal async Telegram Bot API wrapper providing long-polling message retrieval and rich message sending capabilities. Handles text normalization, Markdown-to-HTML conversion, inline keyboards, file uploads, and graceful fallbacks for API errors. Supports both standard and Business Account messaging.

**Purpose:** Abstracts Telegram Bot API complexity into a clean, developer-friendly interface for building chatbots and automated messaging systems. Solves the problems of: (1) parsing nested Telegram API JSON responses into normalized dataclasses, (2) converting Markdown formatting to Telegram-compatible HTML, (3) handling message chunking for the 4096-character limit, (4) implementing retry logic for malformed HTML and stale message references, and (5) managing file uploads/downloads with proper MIME type detection.

---

## Most Important Functions/Classes/Procedures

### 1. **`_md_to_telegram_html(text: str) -> str`**
Converts common Markdown syntax (**bold**, *italic*, `code`, ```code blocks```, [links](url), ~~strikethrough~~, headings) to Telegram-compatible HTML tags. Escapes HTML entities first to prevent injection. Uses regex patterns with proper ordering (code blocks before inline code, bold before italic) to avoid nested pattern conflicts. Critical for safe, formatted message rendering in Telegram clients.

### 2. **`TelegramBridge.get_updates(offset, timeout) -> list[TelegramMessage | TelegramCallbackQuery]`**
Implements long-polling against Telegram's `getUpdates` endpoint. Parses raw API response and normalizes into two dataclass types: `TelegramMessage` (text/photo messages) and `TelegramCallbackQuery` (inline button presses). Handles nested dict validation, photo array extraction (picks largest size), caption fallback for image messages, and Business Account message routing. Returns empty list on API errors rather than raising exceptions.

### 3. **`TelegramBridge.send_message(chat_id, text, reply_to_message_id) -> None`**
Sends text messages with intelligent chunking (splits at newlines near 3800-char boundary to stay under 4096 limit). Converts Markdown to HTML via `_md_to_telegram_html()`. Implements two-level fallback: if HTML parsing fails (400 status), resends as plain text; if reply target is stale, retries without `reply_to_message_id`. Disables web page preview by default.

### 4. **`TelegramBridge.send_message_with_inline_keyboard(chat_id, text, buttons, reply_to_message_id) -> None`**
Sends formatted messages with inline keyboard buttons (callback-based interactions). Accepts button structure as `list[list[dict]]` (rows of buttons with `text` and `callback_data` keys). Uses zero-width space fallback if text is empty. Applies same HTML conversion and error fallback logic as `send_message()`.

### 5. **`TelegramBridge.send_photo() / send_audio_file() / download_file()`**
File handling suite: `send_photo()` uploads images with MIME type detection (.png, .jpg, .webp, .gif), `send_audio_file()` uploads audio, `download_file()` retrieves files via two-step process (getFile API call to get server path, then HTTP download). All validate file existence, handle captions/replies, and raise `FileNotFoundError` or `RuntimeError` on failures.

---

## Architecture & Dependencies

**Core Dependencies:**
- `httpx` (async HTTP client, 40-second timeout)
- `dataclasses` (normalized message/query payloads)
- `re`, `html` (Markdown parsing and HTML escaping)
- `pathlib` (file path handling)

**Design Patterns:**
- **Async-first:** All I/O operations are async (`AsyncClient`), suitable for concurrent bot handling
- **Defensive parsing:** Deep dict validation at each nesting level; returns empty results on malformed responses rather than crashing
- **Graceful degradation:** Multi-level fallbacks (HTML → plain text → retry without reply reference)
- **Dataclass normalization:** Raw Telegram API responses mapped to clean `TelegramMessage` and `TelegramCallbackQuery` types for downstream consumption

**Role in System:** Acts as the I/O boundary layer between a chatbot application and Telegram's servers. Handles all API communication, message formatting, and file transfer. Designed to be stateless and reusable across multiple bot instances via token injection.