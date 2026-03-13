# Summary: discord_bridge.py

# discord_bridge.py Summary

Minimal Discord REST API bridge providing asynchronous polling of DMs and guild channels, message retrieval with offset tracking, and message/audio sending capabilities. Handles message normalization, bot mention detection, and large message chunking for Discord's 2000-character limit.

## Purpose

Solves the problem of integrating Discord as a communication channel for bot applications by abstracting Discord's REST API into a simple polling-based interface with normalized message objects, eliminating the need for WebSocket gateway connections and providing straightforward message sending with automatic chunking and reply support.

## Most Important Functions/Classes

1. **DiscordBridge class** – Main API wrapper managing authentication, HTTP client lifecycle, and all Discord interactions. Maintains bot user ID caching and provides async context for all operations.

2. **get_updates()** – Core polling method that retrieves messages from all accessible DM and guild text channels, tracks message offsets per channel, filters out bot messages, detects bot mentions, strips leading mentions in guild contexts, and returns normalized DiscordMessage objects sorted by ID with refreshed offsets for next poll.

3. **send_message()** – Sends text to a channel with automatic chunking at newline boundaries (max 1800 chars per chunk), supports reply-to functionality on first chunk only, and disables mention parsing for safety.

4. **_list_dm_channels() / _list_guild_text_channels()** – Channel discovery methods that fetch DM channels (type 1, 3) and guild text channels (type 0, 5) respectively, with guild enumeration and deduplication logic.

5. **send_audio_file()** – Uploads MP3 files to channels with multipart form-data, supports optional captions and reply-to references, validates file existence before upload.

## Architecture & Dependencies

- **HTTP Client**: Uses `httpx.AsyncClient` with 40-second timeout for all REST calls
- **Authentication**: Bot token-based authorization via `Authorization: Bot {token}` header
- **Data Model**: `DiscordMessage` dataclass normalizes incoming payloads with fields for ID, channel/user/guild IDs, username, text content, and bot mention flag
- **Offset Tracking**: Dictionary-based message ID tracking per channel enables resumable polling without duplicate processing
- **Error Handling**: Graceful degradation with exception catching on API calls; returns empty lists/strings on failures rather than propagating errors
- **Type Safety**: Full type hints with `dict[str, Any]` for flexible JSON payloads and proper async/await patterns

## Key Design Patterns

- **Stateless polling**: No persistent connection; caller manages offset state between polls
- **Defensive parsing**: Extensive type checking and `.strip()` calls on all string conversions to handle malformed API responses
- **ID comparison**: Custom `_id_to_int()` static method for numeric comparison of Discord's string-based snowflake IDs
- **Username normalization**: Prefers `global_name` over legacy `username#discriminator` format
- **Mention stripping**: Removes leading `<@...>` mentions from guild messages when bot is mentioned, enabling cleaner prompt text