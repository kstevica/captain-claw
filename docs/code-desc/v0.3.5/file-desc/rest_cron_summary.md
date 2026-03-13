# Summary: rest_cron.py

# rest_cron.py Summary

REST API handler module for cron job management in the Captain Claw system. Provides seven async endpoints for CRUD operations, execution control, and monitoring of scheduled cron jobs with full lifecycle management including creation, execution, pause/resume, and history tracking.

## Purpose

Exposes HTTP REST endpoints that enable external clients and UI dashboards to manage automated scheduled tasks (cron jobs) without direct database access. Handles validation, scheduling computation, job state transitions, and delegates execution to the cron dispatch system.

## Most Important Functions/Classes

1. **list_cron_jobs()** — Retrieves all cron jobs (up to 200) with complete metadata including schedule, status, execution history, and error logs. Returns JSON array with job details for dashboard/monitoring purposes.

2. **create_cron_job()** — Validates and creates new cron jobs with comprehensive input validation (kind, schedule format, payload structure, session_id). Computes next run time using schedule_to_text() and compute_next_run() utilities before persisting to database.

3. **run_cron_job()** — Immediately executes a specific job by ID via asyncio.create_task(), triggering execute_cron_job() with "manual" trigger flag. Enables ad-hoc execution independent of schedule.

4. **pause_cron_job() / resume_cron_job()** — Toggle job enabled state and manage status transitions. Resume recalculates next_run_at based on current schedule to prevent missed executions.

5. **update_cron_job_payload() / delete_cron_job()** — Modify job configuration or remove jobs entirely. Update validates payload presence before persisting changes.

6. **get_cron_job_history()** — Returns chat_history and monitor_history for a job, enabling audit trails and execution result inspection with JSON serialization fallback for non-standard types.

## Architecture & Dependencies

- **Framework**: aiohttp (async HTTP server)
- **Core Dependencies**: 
  - `captain_claw.web_server.WebServer` — server context providing agent/session_manager access
  - `captain_claw.cron` — schedule computation (compute_next_run, schedule_to_text, to_utc_iso)
  - `captain_claw.cron_dispatch.execute_cron_job` — job execution engine
  - `captain_claw.logging` — structured logging
- **Data Flow**: HTTP request → validation → session_manager CRUD → response JSON
- **Error Handling**: Consistent 400/403/404/503 status codes with descriptive error messages; agent initialization check on all endpoints
- **Concurrency**: Uses asyncio.create_task() for non-blocking job execution; all handlers are async-compatible