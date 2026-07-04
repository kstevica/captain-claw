# Captain Claw v0.7.2 Release Notes

**Release title:** VFS Hosting — serve sites and apps straight from your VFS
**Release date:** 2026-07-04

Publish anything in your VFS to a public URL through Flight Deck. Additive and
backward compatible with 0.7.1 — but the new routes need a **Flight Deck restart**
to take effect.

## Highlights

- **VFS Hosting (new Workspace page).** Publish a VFS folder at a public,
  globally-unique name. Two kinds:
  - **Static site → `/vfs/<name>/…`** — files served directly, with `index.html`
    + SPA fallback, and a browsable **directory autoindex** when a folder has no
    `index.html`.
  - **Built app → `/vfs-apps/<name>/…`** — Flight Deck runs your start command as
    a managed subprocess and reverse-proxies **HTTP (all methods) + WebSocket** to
    it. Manual **Start/Stop**.
  - Publish, **edit**, Open, and Unpublish from the page; a **folder selector**
    picks which project subfolder to serve. Management is owner-gated
    (`/fd/hosting/*`); serving is public.

- **Base-path–aware apps.** Because apps are served under a path prefix, root-
  absolute URLs (`fetch("/api")`) break. Flight Deck now injects `PORT`, `HOST`,
  `FD_BASE_PATH` (`/vfs-apps/<name>/`) and `FD_VFS_APP` into each app process, and
  the **Code module** bakes this in: its plan/build/quick-edit prompts require
  binding `PORT` and prefixing absolute URLs with `FD_BASE_PATH`, and the
  **code-reviewer** now flags root-absolute URLs and hardcoded ports as findings.

## Notes

- New backend modules (`vfs_hosting.py`, `hosting_routes.py`) + routes registered
  before the SPA catch-all — restart Flight Deck to load them.
- Static SPA builds must set their bundler base to `/vfs/<name>/` at build time.
- Backward compatible with 0.7.1.
