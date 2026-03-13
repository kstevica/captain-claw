# Summary: openai_proxy.py

# openai_proxy.py Summary

This module implements OpenAI-compatible API proxy handlers for the Captain Claw system, enabling clients to interact with the agent infrastructure using standard OpenAI chat completion endpoints. It provides session-based authentication, request validation, and both standard and streaming response formats while maintaining API compatibility with OpenAI's specification.

## Purpose

Solves the problem of integrating Captain Claw's agent system with OpenAI-compatible clients by:
- Translating incoming OpenAI API requests into agent operations
- Managing session authentication via Bearer tokens
- Supporting both standard JSON and Server-Sent Events (SSE) streaming responses
- Providing model discovery and listing capabilities
- Handling error cases with OpenAI-compatible error responses

## Most Important Functions/Classes

1. **`extract_api_session_id(request: web.Request) -> str | None`**
   - Extracts Bearer token from Authorization header for session identification
   - Validates header format and returns token or None if missing/invalid
   - Critical for authentication flow in all API endpoints

2. **`api_chat_completions(server: WebServer, request: web.Request) -> web.Response | web.StreamResponse`**
   - Main endpoint handler for `POST /v1/chat/completions`
   - Orchestrates full request lifecycle: validation → agent creation → completion → response formatting
   - Handles both streaming and non-streaming modes; manages agent pool lifecycle with try/finally release pattern
   - Validates session ID, parses messages array, extracts user message, and delegates to agent.complete()

3. **`write_sse_streaming_response(...) -> web.StreamResponse`**
   - Streams completed responses as Server-Sent Events with proper HTTP headers (Content-Type, Cache-Control, Connection)
   - Chunks response text by 5-word segments and emits individual delta objects
   - Includes usage statistics in final completion chunk and [DONE] marker for stream termination

4. **`build_chat_completion_response(content: str, model: str, usage: dict, completion_id: str | None) -> dict`**
   - Constructs OpenAI-compatible chat completion response object
   - Generates unique completion IDs and timestamps; structures choice/message/usage fields per OpenAI spec

5. **`api_list_models(server: WebServer, request: web.Request) -> web.Response`**
   - Endpoint handler for `GET /v1/models`
   - Returns available models from agent's allowed_models list with OpenAI-compatible metadata
   - Ensures "captain-claw" default model is always present

## Architecture & Dependencies

- **Framework**: aiohttp (async HTTP server)
- **Dependencies**: captain_claw.logging, captain_claw.web_server
- **Key Integration Points**: 
  - `server._api_pool` for agent lifecycle management (get_or_create, release)
  - `agent.complete(message)` for LLM inference
  - `agent.get_runtime_model_details()` for model metadata
  - `agent.last_usage` for token counting
- **Error Handling**: Comprehensive validation with OpenAI-compatible error codes (missing_api_key, invalid_json, no_user_message, agent_init_failed, completion_failed)
- **Session Management**: Pool-based agent lifecycle with guaranteed cleanup via finally blocks