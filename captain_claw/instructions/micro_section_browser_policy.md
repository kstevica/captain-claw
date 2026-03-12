MANDATORY browser policy:
- Use `browser` ONLY for dynamic web apps needing JS/login/interaction. For simple reading, use `web_fetch`.
- Observe-act workflow: observe first → act on what you see → verify result → repeat.
- Element targeting order: 1) ARIA role+name, 2) text-based, 3) CSS selector, 4) nth disambiguation.
- Login: `browser(action='login', app_name='...')` tries cookies first, then form login. Store creds with `credentials_store`.
- Network capture: after browser workflows, `network_capture` saves API patterns. Then use `api_replay` for direct HTTP calls (skip browser).
- Multi-app: `switch_app` creates isolated browser contexts per app. `list_sessions` shows active sessions.
- Workflow recording: `workflow_record_start` → user interacts → `workflow_record_stop` → `workflow_save` with variables → `workflow_run` to replay.