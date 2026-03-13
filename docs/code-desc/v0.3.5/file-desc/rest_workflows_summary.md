# Summary: rest_workflows.py

# rest_workflows.py Summary

Provides REST API endpoints for browsing and retrieving workflow execution results stored as JSON and Markdown files in the workspace directory. Enables clients to list all workflows with their metadata and output files, and retrieve individual output file contents with security validation.

## Purpose

Solves the problem of exposing workflow execution history and results through a web interface. Allows users to discover completed workflows, view their parameters and task counts, access timestamped output files, and retrieve full output content—all through a RESTful API that safely validates file paths to prevent directory traversal attacks.

## Most Important Functions/Classes

1. **`workflows_dir() → Path`**
   - Retrieves the workflows directory from configuration (workspace_path/workflows), creating it if necessary. Acts as a centralized reference point for file operations, mirroring SessionOrchestrator's directory structure.

2. **`list_workflow_outputs(server: WebServer, request: web.Request) → web.Response`**
   - Core endpoint (GET /api/workflow-browser) that aggregates workflow metadata and outputs. Scans for *.json workflow definition files and *-output-*.md result files, correlating them by filename patterns. Returns sorted list of workflows with their associated outputs, handling missing/corrupted files gracefully with fallback values.

3. **`get_workflow_output(server: WebServer, request: web.Request) → web.Response`**
   - Endpoint (GET /api/workflow-browser/output/{filename}) that retrieves individual output file contents. Implements security validation (rejects ".." and "/" in filenames) to prevent directory traversal, returns 404 for missing files and 400 for invalid requests.

## Architecture & Dependencies

- **Framework**: aiohttp web framework for async HTTP handling
- **Configuration**: Integrates with captain_claw.config for workspace path resolution
- **File System**: Direct Path-based file I/O with exception handling for robustness
- **Data Format**: JSON for workflow definitions, Markdown for outputs
- **Type Hints**: Full TYPE_CHECKING support with WebServer type annotation
- **Role**: Serves as the HTTP interface layer for the workflow browser UI, bridging file system storage with client-facing REST API