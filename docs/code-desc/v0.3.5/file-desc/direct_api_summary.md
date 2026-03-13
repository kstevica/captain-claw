# Summary: direct_api.py

# Direct API Tool Summary

## Overview
A comprehensive HTTP API endpoint management and execution tool that enables registration, storage, and execution of direct API calls with full CRUD operations, authentication handling, and usage tracking. Integrates with browser session management to capture authentication tokens automatically.

## Purpose
Solves the problem of managing and executing multiple API endpoints without hardcoding them into scripts. Provides a persistent registry for API calls with metadata (schemas, auth, headers), execution capabilities with statistics tracking, and intelligent auth token capture from active browser sessions. Particularly useful for testing APIs, integrating external services, and automating API workflows.

## Architecture & Dependencies
- **Core Dependency**: `captain_claw.session.SessionManager` — handles persistent storage of API call definitions and usage statistics
- **Integration**: `captain_claw.tools.browser_api_replay.ApiReplayEngine` — executes HTTP requests with proper header/auth handling
- **Browser Integration**: Accesses browser tool's network captures and cookie storage for auth extraction
- **Logging**: Uses `captain_claw.logging` for error tracking
- **Tool Framework**: Extends `captain_claw.tools.registry.Tool` base class with standardized parameter schema and async execution

## Most Important Functions/Classes

### 1. **DirectApiTool (Main Class)**
   - Extends Tool base class with name="direct_api" and comprehensive parameter schema
   - Routes 8 different actions through async `execute()` dispatcher
   - Enforces security by restricting HTTP methods to GET/POST/PUT/PATCH (DELETE explicitly blocked)
   - Timeout: 60 seconds for all operations

### 2. **_add() — API Call Registration**
   - Creates new API endpoint entries with validation (name, URL, method required)
   - Stores metadata: description, input/output schemas, headers, auth credentials, app grouping, tags
   - Sets auth_source="manual" when credentials provided directly
   - Returns confirmation with truncated ID and endpoint details

### 3. **_execute_call() — HTTP Request Execution**
   - Dual-mode operation: `record=True` for production calls (tracks stats), `record=False` for testing
   - Resolves authentication headers via ApiReplayEngine (bearer, api_key, basic, cookie, none)
   - Merges extra headers with auth headers; parses JSON payload and query parameters
   - Splits URL into base_url/endpoint for ApiReplayEngine compatibility
   - Records usage stats (status code, response preview ≤500 chars) when record=True
   - Returns formatted result with status, elapsed time, error details, and response body (truncated at 2000 chars)

### 4. **_auth_from_browser() — Intelligent Auth Token Capture**
   - Extracts authentication from active browser session without manual entry
   - Primary strategy: searches network captures for matching domain, infers auth type (bearer/api_key/basic/cookie)
   - Fallback strategy: extracts domain-specific cookies from browser storage
   - Updates API call entry with captured credentials and auth_source="browser"
   - Returns descriptive success/failure messages with auth type and domain info

### 5. **_list() — API Call Inventory**
   - Retrieves up to 50 registered API calls with optional app_name filtering
   - Displays compact table format: index, method, name, auth type, app grouping, URL, usage count, truncated ID
   - Supports browsing and discovery of registered endpoints

### 6. **_show() — Detailed API Call Inspection**
   - Retrieves full metadata for single API call by ID, index, or name
   - Displays: name, ID, method, URL, description, auth details, payload schemas, headers, app/tags, usage stats, last execution details (status, response preview), creation timestamp
   - Comprehensive reference for understanding endpoint requirements

### 7. **_update() — Metadata Modification**
   - Modifies any field of existing API call (name, URL, method, auth, headers, schemas, tags, etc.)
   - Validates HTTP method against allowed set
   - Requires at least one field update; returns success confirmation

### 8. **_remove() — API Call Deletion**
   - Deletes API call entry by ID, index, or name
   - Returns confirmation with truncated ID and name

## Key Design Patterns
- **Action Dispatcher**: Single execute() method routes to specific handlers based "action" parameter
- **Flexible Identification**: Supports lookup by call_id (UUID), #index, or friendly name
- **Security-First**: Explicit method whitelist, DELETE forbidden, auth token handling via SessionManager
- **Browser Integration**: Seamless extraction of auth from active sessions without user intervention
- **Usage Analytics**: Tracks call frequency, last execution time, status codes, response previews
- **Error Resilience**: Graceful JSON parsing fallbacks, comprehensive error messages, detailed logging