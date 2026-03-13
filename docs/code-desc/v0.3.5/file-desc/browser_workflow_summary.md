# Summary: browser_workflow.py

# browser_workflow.py Summary

## Summary
A browser workflow recording and replay engine that captures user interactions (clicks, typing, navigation) through injected JavaScript event listeners and replays them later with variable substitution. Implements a "teach once, run many" pattern where users demonstrate tasks once and the system executes them with different data inputs. The architecture separates recording (WorkflowRecorder) from replay (WorkflowReplayEngine) with robust selector fallback strategies for element identification.

## Purpose
Solves the problem of automating repetitive browser-based tasks without requiring manual script writing. Enables non-technical users to record their workflow by interacting with a browser normally, then replay that workflow with different data (e.g., different customer names, addresses, or search terms) through template variable substitution. Particularly valuable for RPA (Robotic Process Automation) scenarios, data entry workflows, and cross-browser testing.

## Most Important Functions/Classes/Procedures

### 1. **WorkflowRecorder Class**
Captures user interactions by injecting JavaScript event listeners into the browser page. Manages lifecycle (attach/detach), recording control (start/stop), and step collection. Key methods:
- `attach(page)`: Registers callback and injects recorder JS
- `start_recording()` / `stop_recording()`: Controls recording state
- `_on_step(step_json)`: Callback invoked from injected JS for each interaction; parses JSON and appends RecordedStep to internal list

### 2. **_RECORDER_JS (JavaScript Injection)**
Embedded JavaScript that runs in the browser to capture interactions. Implements event listeners for clicks, input changes, form submissions, and SPA navigation. Extracts rich selector information including CSS selectors, ARIA roles, accessible names, visible text, placeholders, and name attributes. Handles both traditional MPA (Multi-Page Application) and SPA (Single-Page Application) navigation patterns through history API interception.

### 3. **WorkflowReplayEngine.replay() Class Method**
Executes recorded steps sequentially with variable substitution. Replaces `{{variable_name}}` placeholders in step values and URLs. Implements action dispatching (navigate, click, type, select, press_key, submit) and returns detailed WorkflowReplayResult with per-step success/failure tracking. Includes configurable step delays for page stability.

### 4. **_click_with_fallback() / _type_with_fallback() Methods**
Implement resilient selector strategies with graceful degradation. Click fallback chain: ARIA role+name → visible text → CSS selector. Type fallback chain: ARIA role+name → placeholder attribute → name attribute → CSS selector. Each strategy includes timeout handling and comprehensive error logging for debugging selector failures.

### 5. **RecordedStep / StepResult / WorkflowReplayResult Data Classes**
Structured data containers for workflow state. RecordedStep captures individual interactions (action type, URL, value, selectors, timestamp). StepResult tracks replay outcome per step. WorkflowReplayResult aggregates workflow-level results with human-readable summary generation via `to_summary()` method.

## Architecture & Dependencies

**Key Design Patterns:**
- **Lifecycle Pattern**: Recorder mirrors NetworkInterceptor with attach→start→stop→detach lifecycle
- **Fallback Strategy Pattern**: Multiple selector strategies with ordered fallback chains ensure robustness across different DOM structures
- **Variable Substitution**: Template-based variable replacement enables parameterization without code changes
- **Async/Await**: Full async implementation for non-blocking browser operations

**Dependencies:**
- Playwright/Puppeteer-compatible page API (exposed_function, evaluate, navigate, click, fill, select_option, press_key)
- BrowserSession abstraction (navigate, click_by_role, click_by_text, click, type_by_role, type_text, press_key, ensure_page)
- Standard library: asyncio, json, time, dataclasses, typing
- captain_claw.logging for structured logging

**System Role:**
Acts as the workflow automation layer in a larger browser automation framework. Sits between user interaction capture and programmatic replay, enabling non-technical workflow definition. Integrates with BrowserSession for actual browser control and supports both recording and playback phases of workflow lifecycle.