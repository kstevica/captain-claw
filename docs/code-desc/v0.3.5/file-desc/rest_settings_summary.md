# Summary: rest_settings.py

# rest_settings.py Summary

Provides REST API endpoints for web-based configuration management, enabling real-time editing and validation of Captain Claw settings through a structured schema-driven UI. Handles GET/PUT operations for settings schema, current values (with secret masking), and persistent YAML configuration updates with hot-reload capabilities.

## Purpose

Solves the problem of managing complex, hierarchical application configuration through a web interface while maintaining security (masking secrets), validation (Pydantic), and runtime flexibility (hot-reloading tools/hotkeys without restart). Bridges the gap between YAML file-based config and interactive UI editing.

## Most Important Functions/Classes/Procedures

1. **`_build_schema()`** – Constructs the complete settings schema (13 major groups: Personality, Provider Keys, Model/LLM, Memory, Tools, Email, Safety, Integrations, General, Features, Scale/Orchestrator, Web Server, Deep Memory, Skills, BotPort). Returns hierarchical list of groups/sections/fields with UI metadata (type, options, hints, conditional visibility). This is the single source of truth for UI rendering and field validation.

2. **`put_settings(server, request)`** – Core handler for applying configuration changes. Workflow: (1) parse JSON changes, (2) read existing YAML from disk, (3) merge changes with dotted-key notation, (4) validate merged config against Pydantic model, (5) save to `~/.captain-claw/config.yaml`, (6) reload in-memory singleton, (7) trigger hot-reloads (tools, hotkey daemon). Enforces readonly fields, skips unchanged secrets, and ensures always-on tools remain enabled.

3. **`_flatten_values(cfg, schema)`** – Converts nested Pydantic Config object into flat dictionary (`{"model.provider": "openai", ...}`) for API response. Masks secret field values with `••••••••` and handles Pydantic model serialization. Enables UI to display current state without exposing sensitive data.

4. **`_get_nested(obj, dotted_key)` / `_set_nested(data, dotted_key, value)`** – Utility pair for navigating/mutating nested dictionaries and objects using dot notation (e.g., `"model.api_key"`). Abstracts complexity of multi-level config structure, supporting both dict and attribute access patterns.

5. **`get_settings_schema(server, request)` / `get_settings_values(server, request)`** – Simple GET endpoints returning schema structure and current masked values respectively. Enable UI bootstrap and state synchronization.

## Architecture & Dependencies

- **Dependency chain**: Imports `Config` (Pydantic model), `get_config`/`set_config` (singleton management), `yaml` (persistence), `aiohttp.web` (HTTP framework)
- **Security model**: `_SECRET_FIELDS` frozenset (23 fields) and `_READONLY_FIELDS` frozenset (2 fields) define access control; secrets masked in responses and skipped if unchanged in updates
- **Hot-reload integration**: Conditionally imports and invokes `reload_tools()` on agent and hotkey daemon restart functions when relevant settings change
- **Config persistence**: Reads/writes `~/.captain-claw/config.yaml` (home overlay) while respecting local `captain-claw.yaml` (base) via `Config._deep_merge()` and `Config._read_yaml_data()`
- **Validation**: Leverages Pydantic's `Config(**merged_data)` constructor to validate merged config before persistence; returns 422 on validation failure
- **Role in system**: Acts as the configuration management API layer for the web UI, enabling non-technical users to adjust settings without manual YAML editing while maintaining data integrity and security