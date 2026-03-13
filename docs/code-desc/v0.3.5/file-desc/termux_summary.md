# Summary: termux.py

# Termux.py Summary

**Summary:**
A comprehensive Termux API integration tool that enables interaction with Android device hardware features (camera, battery, GPS, flashlight) through CLI commands. Implements async command execution with timeout and abort handling, file registry integration for media artifacts, and structured parameter validation for four distinct device actions.

**Purpose:**
Solves the problem of programmatically accessing native Android device capabilities from within a Termux environment by wrapping the termux-api package commands. Enables AI agents or automation systems to capture photos, query device status, retrieve location data, and control hardware without direct Android SDK integration.

---

## Most Important Functions/Classes/Procedures

### 1. **TermuxTool (Class)**
Core tool implementation inheriting from `Tool` base class. Defines the tool interface with metadata (name, timeout, description), parameter schema for all four actions, and orchestrates action routing. Manages tool lifecycle and integrates with the captain_claw framework's tool registry system.

### 2. **_run_command() (Async Method)**
Low-level command execution wrapper handling subprocess lifecycle with sophisticated async control flow. Implements timeout enforcement, abort event signaling, proper process cleanup, and UTF-8 output decoding. Returns tuple of (success: bool, stdout: str, stderr: str) with graceful error handling for timeouts and cancellations.

### 3. **_action_photo() (Async Method)**
Captures device photos via `termux-camera-photo` with intelligent file organization. Manages output directory creation (saved/media/{session_id}), generates timestamped filenames with camera labels, registers files in the file registry system, and validates successful photo creation. Supports both back (id=0) and front (id=1) cameras with 30-second timeout.

### 4. **_action_location() (Async Method)**
Retrieves device location using `termux-location` with provider selection (gps/network/passive). Validates provider parameter with fallback to GPS, constructs command with `-r once` flag for single-shot retrieval, and returns raw JSON output. Implements 15-second timeout suitable for network-dependent operations.

### 5. **execute() (Async Method)**
Main entry point implementing action dispatcher pattern. Routes requests to appropriate action handler (_action_photo, _action_battery, _action_location, _action_torch) based on normalized action parameter. Validates action enum and delegates all kwargs (including _abort_event, _file_registry, _session_id) to handlers for framework integration.

---

## Architecture & Dependencies

**Framework Integration:**
- Inherits from `captain_claw.tools.registry.Tool` base class
- Returns `ToolResult` objects for standardized response handling
- Uses `captain_claw.logging.get_logger()` for structured logging
- Integrates with file registry system via `_file_registry` kwarg for artifact tracking

**Async Architecture:**
- Fully async/await implementation using asyncio primitives
- Supports concurrent command execution with timeout enforcement
- Implements abort event pattern for graceful cancellation
- Uses `asyncio.wait()` with `FIRST_COMPLETED` for race condition handling between process completion and abort signals

**External Dependencies:**
- Termux-api package (termux-camera-photo, termux-battery-status, termux-location, termux-torch CLI commands)
- Termux:API companion Android app with appropriate permissions (camera, location, hardware control)
- Standard library: asyncio, os, pathlib, datetime

**System Role:**
Acts as a hardware abstraction layer enabling AI agents or automation workflows to interact with Android device capabilities. Designed for integration into larger agent systems where device sensor data and hardware control are required for task execution.