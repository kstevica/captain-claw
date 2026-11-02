# Agent groups, end to end

Drives the kiosk's agent groups on a real deck: creating, dragging, the picker, folding,
search, reordering, editing, deleting, pins, the collapsed rail, a change from another device,
the phone layout, an admin in the desktop app's user view and in the web's simple layout,
and a deck without sign-in.

```bash
cd flight-deck && npm run build && cd ..
flight-deck/e2e/groups/run.sh                 # everything
flight-deck/e2e/groups/run.sh kiosk phone     # kiosk, phone, pins, admin, no-sign-in
```

Needs uv and Python Playwright with Chromium. Screenshots land in `.run/shots/`. From a
worktree, point uv at the main checkout's environment:
`UV_NO_SYNC=1 UV_PROJECT_ENVIRONMENT=<main checkout>/.venv`.

Sign-in on this deck: `maya@example.com` (a kiosk user) and `admin@example.com`, both with
the password in `run.sh`.
