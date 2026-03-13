# Summary: personality.py

# personality.py Summary

This module implements a tool for reading and updating personality profiles in the Captain Claw agent system, supporting both global agent personality and per-user profiles in multi-user contexts (e.g., Telegram agents). It provides dual-mode operation that switches between agent and user profile management based on context.

## Purpose

Solves the problem of maintaining and modifying agent identity and user context information. Enables agents to have configurable personalities (name, description, background, expertise, custom instructions) and allows tracking of user profiles for personalized interactions. Supports both global agent configuration and per-user customization in multi-user deployment scenarios.

## Most Important Functions/Classes

1. **PersonalityTool (class)**
   - Main tool class inheriting from Tool framework. Manages dual-mode operation (agent vs. user profile) via `_user_id` state. Exposes `execute()` method that routes to get/update operations with comprehensive error handling and logging.

2. **execute() (async method)**
   - Entry point for tool invocation. Accepts action parameter ("get" or "update") plus optional profile fields (name, description, background, expertise, instructions). Delegates to `_get_personality()` or `_update_personality()` and wraps exceptions in ToolResult objects.

3. **_get_personality() (method)**
   - Retrieves current personality profile (agent or user-specific) via `load_effective_personality()`. Formats profile data into human-readable output showing scope, name, description, background, expertise list, and instructions. Returns ToolResult with formatted content.

4. **_update_personality() (method)**
   - Modifies personality fields with validation (empty name rejection, whitespace trimming, expertise parsing from comma-separated string). Conditionally loads user or global personality, applies changes, persists via `save_personality()` or `save_user_personality()`, and returns confirmation with changed field list.

5. **set_user_mode() (method)**
   - Switches tool from agent mode to user-profile mode by setting `_user_id` and updating description/parameters to reflect user-focused operations. Called during agent registration when user context is available.

## Architecture & Dependencies

- **Dependencies**: Imports from `captain_claw.logging` (logging), `captain_claw.tools.registry` (Tool base class, ToolResult), and `captain_claw.personality` (personality model and persistence functions)
- **Role**: Middleware tool providing CLI-like interface to personality management subsystem
- **Design Pattern**: Dual-mode tool with state-based behavior switching; parameter schemas differ between agent and user modes
- **Data Model**: Works with Personality objects containing name, description, background, expertise (list), and instructions (optional)
- **Persistence**: Delegates to external personality module for load/save operations, supporting both global and per-user storage