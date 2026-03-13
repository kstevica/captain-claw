# Summary: datastore.py

# Datastore.py Summary

This module implements a comprehensive LLM-managed relational database tool that enables AI agents to perform SQL-like operations on persistent data tables with built-in protection mechanisms and multi-format import/export capabilities.

## Purpose

Provides a `DatastoreTool` that abstracts complex database operations into 23 distinct actions, allowing LLMs to safely manage structured data through a standardized interface. Solves the problem of giving AI agents controlled, auditable access to persistent relational data while preventing unauthorized modifications through protection rules.

## Architecture & Dependencies

- **Core dependency**: `captain_claw.datastore.ProtectedError` and `get_datastore_manager()` for backend database operations
- **Logging**: Integrates with `captain_claw.logging` for operation tracking
- **Tool framework**: Extends `Tool` and `ToolResult` from `captain_claw.tools.registry`
- **Data handling**: Uses `json`, `pathlib.Path` for file operations and JSON parsing
- **Async-first**: All operations are async-compatible for non-blocking execution

## Most Important Functions/Classes

### 1. **DatastoreTool (class)**
Main tool class exposing 23 database actions via standardized `execute()` method. Handles parameter validation, error catching (especially `ProtectedError`), logging, and result formatting. Timeout set to 60 seconds. Acts as the primary interface between LLM and datastore backend.

### 2. **execute() (async method)**
Central dispatcher that routes action strings to appropriate handler methods. Implements comprehensive error handling with special treatment for `ProtectedError` (blocks operations and informs user). Logs all invocations and results with truncation for readability. Returns `ToolResult` objects indicating success/failure.

### 3. **_format_table() (function)**
Formats query results as compact markdown tables with dynamic column width calculation. Handles NULL values, row count totals, and proper alignment. Critical for presenting data results to LLMs in readable format. Returns formatted string or "No rows returned" message.

### 4. **_parse_json_str() & _parse_columns() (utility functions)**
Flexible parameter parsers handling multiple input formats (native dicts/lists, JSON strings, CSV). `_parse_json_str()` gracefully handles both pre-parsed objects and JSON strings. `_parse_columns()` accepts lists, JSON arrays, or comma-separated values. Essential for LLM parameter flexibility.

### 5. **Action Handler Methods (23 total)**
Individual async methods for each operation:
- **Schema operations**: `_create_table()`, `_drop_table()`, `_rename_table()`, `_add_column()`, `_rename_column()`, `_drop_column()`, `_change_column_type()`
- **Data operations**: `_insert()`, `_update()`, `_update_column()`, `_delete()`, `_query()`, `_sql()`
- **I/O operations**: `_import_file()` (CSV/XLSX), `_export()` (CSV/JSON/XLSX with auto-path generation)
- **Protection operations**: `_protect()`, `_unprotect()`, `_list_protections()`
- **Metadata**: `_list_tables()`, `_describe()`

Each validates required parameters, parses inputs, delegates to datastore manager, and returns formatted results.

## Key Features

**Multi-format support**: Import/export CSV, XLSX, JSON with intelligent path resolution relative to `saved/` directory and session-based auto-generation.

**Granular protection system**: Four-level protection (table, column, row, cell) with optional reasons, blocking operations at execution time with user-friendly error messages.

**Flexible querying**: Supports filtered queries with JSON-based WHERE clauses, ordering, pagination (limit/offset), and raw SQL SELECT for complex operations like JOINs.

**Robust parameter handling**: Accepts parameters in multiple formats (native objects, JSON strings, CSV lists) to accommodate LLM output variability.

**Comprehensive logging**: Tracks all operations with action, parameters, success/failure, and truncated content preview for debugging and audit trails.

**Path resolution**: Handles both absolute and relative file paths with intelligent base directory resolution for imports and exports.