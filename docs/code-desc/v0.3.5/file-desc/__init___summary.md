# Summary: __init__.py

# Summary

This is the package initialization file for Captain Claw's tools module, serving as the central export hub for a comprehensive toolkit of 40+ specialized tools and core infrastructure components. The file aggregates imports from 30+ individual tool modules and the registry system, exposing them through a single public interface via `__all__`.

# Purpose

The `__init__.py` file solves the problem of scattered tool definitions across multiple modules by providing a unified, discoverable API for consumers of the tools package. It enables clean imports (e.g., `from captain_claw.tools import ReadTool, WebSearchTool`) rather than requiring knowledge of individual module locations, while maintaining clear separation of concerns across the codebase.

# Most Important Functions/Classes/Procedures

1. **Tool Registry Infrastructure** (`Tool`, `ToolRegistry`, `get_tool_registry`)
   - Core abstraction layer defining the tool interface and centralized registry for tool discovery and management. `ToolRegistry` maintains the catalog of available tools; `get_tool_registry()` provides singleton access.

2. **Tool Policy System** (`ToolPolicy`, `ToolPolicyChain`)
   - Security and access control framework that enforces constraints on tool execution. `ToolPolicyChain` enables composable policy validation before tool invocation.

3. **ToolResult**
   - Standard return type for all tool executions, encapsulating output, errors, and metadata in a consistent format across the heterogeneous tool ecosystem.

4. **File & Text Operations** (`ReadTool`, `WriteTool`, `EditTool`, `GlobTool`)
   - Foundation tools for filesystem manipulation and text processing; critical for agent workflows involving file I/O and pattern matching.

5. **Web & Data Integration** (`WebFetchTool`, `WebSearchTool`, `GoogleDriveTool`, `GoogleMailTool`, `TypesenseTool`, `DatastoreTool`)
   - External service connectors enabling agents to access web content, search capabilities, and cloud services (Google Workspace, search indexes, data persistence layers).

6. **Document Processing** (`PdfExtractTool`, `DocxExtractTool`, `XlsxExtractTool`, `PptxExtractTool`)
   - Multi-format document parsers for extracting structured data from enterprise file formats.

7. **AI/ML Capabilities** (`ImageGenTool`, `ImageOcrTool`, `ImageVisionTool`, `PocketTTSTool`)
   - Generative and perception tools for image generation, optical character recognition, vision analysis, and text-to-speech synthesis.

8. **Desktop & System Automation** (`BrowserTool`, `DesktopActionTool`, `ScreenCaptureTool`, `ClipboardTool`, `TermuxTool`)
   - Low-level system interaction tools for GUI automation, screen capture, and mobile/terminal command execution.

9. **Execution & Extensibility** (`ShellTool`, `ScriptsTool`, `PlaybooksTool`, `DirectApiTool`, `ApisTool`)
   - Mechanisms for executing arbitrary code, scripts, and API calls; enable dynamic behavior and integration with custom business logic.

10. **Productivity & Context** (`TodoTool`, `ContactsTool`, `PersonalityTool`, `GoogleCalendarTool`)
    - User-facing tools for managing tasks, contacts, calendar events, and maintaining agent personality/context state.