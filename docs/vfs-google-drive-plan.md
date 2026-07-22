# Google Drive as a VFS folder

Mount a Drive folder at `vfs:<project>/` so its tree appears in the VFS panel and
agents read its files as if they were local — with **nothing mirrored** by
default, only placeholders. Plus **clonemd**: convert the remote files to
Markdown and keep them on disk as ordinary files.

Status: **plan only, not started.**

---

## Decisions taken

| | |
|---|---|
| Direction | **Read-only.** Writes into a mount are refused. clonemd output is local-only; nothing is ever pushed to Drive. |
| Search | grep searches what is materialised and **says what it skipped** — "N files in this mount are not cloned and were not searched." Silence is the failure mode we are buying our way out of. |
| clonemd covers | Google-native (Docs/Sheets/Slides), Office + PDF, plain text verbatim. **Not images/OCR.** |
| Enumeration | **Lazy per folder, with a cap.** List a directory's children when first touched; cache the listing. |

---

## The load-bearing constraint

`project_root()` returns a `pathlib.Path` and roughly twenty call sites across
`vfs_routes.py`, `code_routes.py`, `deep_memory_service.py` and the file tools do
real I/O on it. There is no storage-backend seam.

Introducing one — a virtual filesystem interface behind `project_root()` — means
touching every one of those call sites and every tool that calls `.read_text()`,
`glob.glob()` or `os.walk()`. That is a rewrite of the VFS, not a feature.

**So: a Drive mount is a real local directory tree of placeholder files,
registered as an ordinary VFS link.** Directories are real directories. Files are
real (tiny) files. `glob`, `ls`, the FD file browser, tree-walking and path
resolution all keep working with **zero changes**, because from the filesystem's
point of view nothing is unusual. Only *content* is virtual, and content has
exactly four readers to intercept.

This also means the mount reuses `.vfs-links.json` rather than inventing a
parallel registry:

```json
{
  "acme-contracts": {
    "path": "/…/fd-data/vfs/<user>/.drive/acme-contracts",
    "mode": "ro",
    "kind": "gdrive",
    "drive": { "folder_id": "1AbC…", "clonemd": false, "synced_at": 0 }
  }
}
```

`link_target_at()`, `project_root()`, `resolve_under()`, `_assert_writable()`
(via `mode: "ro"`) and the UI's link badge all work as-is. Only the new
behaviours read the `drive` block.

---

## Phase 0 — unblock

**SHIPPED (commit 34db19b)** — everything except per-user tokens, which the work
below reframed from a blocker into a deferrable swap:

- **New `captain_claw/drive_client.py`** — the structured, paginated, retrying
  client the mount consumes. Pagination (the tool discarded `nextPageToken`),
  backoff on 429/5xx honouring `Retry-After`, scope-flexible auth, query
  escaping, and Google-native export standardised on Sheet→xlsx / Slides→pptx.
  Returns `DriveFile` objects, not prose. 17 tests.
- **`google_drive` tool**: read actions accept `drive.readonly`; only writes
  need a writable scope; `folder_id` escaped in the list query.
- **`action_catalog`**: dropped the dead `drive.delete` entry.

**Per-user Google tokens — SHIPPED.** Google was one deployment-wide
connection (`system_settings` has no `user_id`), so every FD user shared one
account and any user's connect overwrote everyone's. Now:

- Tokens + user identity live per-user in `user_settings` (the OAuth *client* —
  id/secret/project/scopes — stays global; it's the deployment's app).
- `/fd/google/login` records the connecting user (the popup passes `?fd_token=`)
  into the PKCE state, so `/callback` stores the tokens against the right
  account.
- `/access_token` and `/credentials` resolve the *calling agent's* owner from
  FD's records (`X-Agent-Auth` web_auth token → registry, then port, then
  primary owner) — the same ladder deep memory uses; the shared agent secret is
  a gate, not an identity. The agent-side `GoogleOAuthManager` now sends
  `X-Agent-Auth`.
- `/status`, `/probe`, `/logout` are scoped to the logged-in user. Rotating the
  client or changing scopes clears **all** users (a refresh token is bound to
  the client that minted it); a normal logout clears just the one.
- **No migration step:** the primary owner transparently reads any legacy global
  token until they reconnect or a refresh rewrites it per-user.

This makes every Google tool — Drive mounts included — multi-tenant safe. The
`DriveClient` still uses `global_token_provider`, which under FD calls the
manager, which now resolves per-owner; no Drive code changed.

---

## Phase 1 — the mount

**SHIPPED.** `captain_claw/vfs_drive.py` + Drive routes on `vfs_routes` +
44 tests (client + mount + routes). The mount is a real placeholder tree, a
manual refresh reflects upstream adds/removes, and it lists in the panel as
`kind: "gdrive"`. Read-hydration and clonemd conversion are still Phases 2–3;
today reading a placeholder returns its marker text and the clonemd toggle only
records the flag.

Design as built:

**New module `captain_claw/vfs_drive.py`** — the only thing that knows a folder
is Drive-backed. No FD imports, mirroring `vfs.py`.

- `mount(user_id, project, folder_id)` — create the placeholder root, write the
  link entry, list the top level.
- `manifest_path(root)` → `<mount>/.drive-manifest.json`, the authority on what
  is real:
  ```json
  { "reports/q3.docx": {
      "file_id": "1X…", "mime": "application/vnd…document",
      "size": 24576, "modified": "2026-07-20T…", "state": "placeholder" } }
  ```
  `state` ∈ `placeholder | hydrated | cloned`.
- `list_dir(mount, rel)` — lazy: on first touch, page through the Drive listing,
  create real subdirectories, write placeholder files, record the manifest.
  Cached by directory with a TTL; a cap (`drive.max_files`, default 5000) that
  **logs and surfaces** what it dropped rather than truncating quietly.

**Placeholder contents matter.** Not zero bytes — a short honest line:

```
⟨Google Drive · not cloned⟩ q3.docx · 24 KB · modified 2026-07-20 · id 1X…
Enable clonemd on this folder, or read this path to fetch it on demand.
```

Anything that reads a placeholder through a path we did *not* intercept then
gets an explanation instead of an empty file. That is the difference between a
confusing bug and a legible one.

**FD routes** — extend `vfs_routes` rather than a new router, since this is a
kind of link: `POST /fd/vfs/links/gdrive` (mount), `GET /fd/vfs/links/gdrive/browse`
(folder picker — `file_tree_builder.browse_gdrive_folders()` already does this,
though it is `gws`-CLI-only and needs porting to the OAuth path), and
`POST /fd/vfs/links/gdrive/{project}/refresh`.

---

## Phase 2 — reading remote files as if local

**SHIPPED.** Hydrate-on-read, grep skip-with-count, and read-only enforcement
across the agent write/edit tools and the FD preview. 29 new tests (93 total in
the Drive/VFS suite). clonemd conversion (Phase 3) reuses the same
`bytes_to_text` converter built here.

Four interception points, all thin, all calling into `vfs_drive`.

| Reader | Behaviour |
|---|---|
| `tools/read.py` | Before opening: if the path is a placeholder, **hydrate** — download (or export) into a content cache, mark `hydrated`, return real content. Transparent. |
| `tools/grep.py` | Skip un-materialised files; **append the count to the result**. This is the chosen semantics: a search that cannot see a file must say so. |
| `tools/glob.py` | No change. Placeholders are real files, so patterns match. |
| `tools/write.py`, `edit.py` | Refuse — `mode: "ro"` already does this via `_assert_writable`; verify the tools honour it and give a Drive-specific message. |

Plus FD's own `GET /fd/vfs/read` and `/download`, so the panel shows content too.

**Content cache** at `<mount>/.drive-cache/<file_id>`, size-capped and LRU-evicted.
Hydration is keyed on `modifiedTime` so a stale entry refetches.

**Google-native docs have no bytes** — Docs/Sheets/Slides must be *exported*, so
even a "raw" read is a format decision. Use the best path already in the repo:
`_gws_docs.py:25-34,137-149` exports Sheets→xlsx and Slides→pptx and then runs
`_extract_xlsx_markdown` / `_extract_pptx_markdown`. That is markedly better than
`google_drive.py`'s map, which sends Sheets to **CSV (first tab only)** and Slides
to flat text. The two maps disagree today (`google_drive.py:38-44` vs
`_gws_drive.py:413-424`); this plan standardises on the gws-quality one, ported to
the OAuth transport.

Also port `_strip_base64_images` (`_gws_runtime.py:29-41`) — it exists only on the
gws path, and a Doc with inline images will otherwise blow up the context window.

---

## Phase 3 — clonemd

**SHIPPED.** Enabling clonemd converts the tree to real Markdown now; refresh
keeps it in delta. Cloned files are ordinary on disk, so grep searches them
with no caveat and deep memory can index the mount. 13 new tests.

Notes as built: the verbatim-vs-rename decision is on the *source* extension,
not the export format (a Google Doc has no text extension, so it renames to
`.md` rather than being copied verbatim); an unconvertible file is marked so a
refresh doesn't re-download it to fail again; and a corrupt download degrades to
a placeholder instead of raising.

A per-mount toggle. When on, a refresh materialises **real Markdown files** and
the mount stops being virtual for those paths.

**Converters**, all already in the repo:

| Source | Path |
|---|---|
| Google Doc | export `text/markdown` |
| Google Sheet | export xlsx → `_extract_xlsx_markdown` |
| Google Slides | export pptx → `_extract_pptx_markdown` |
| `.pdf` | `_extract_pdf_markdown` |
| `.docx` / `.xlsx` / `.pptx` | the matching `_extract_*_markdown` |
| `.md`/`.txt`/`.csv`/`.json`/… | copied verbatim |
| everything else | stays a placeholder |

**Naming.** `report.docx` → `report.md`; on collision (a `report.pdf` alongside),
fall back to `report.pdf.md`. A Google Doc named "Quarterly Report" becomes
`Quarterly Report.md` naturally. Recorded in the manifest either way, so the
mapping is never guessed.

**Refresh** is manual and folder-wide, per the brief. Delta by `modifiedTime`
against the manifest: unchanged files are skipped without a download, changed
files re-convert, and files that vanished upstream are **removed locally** — a
mount that accumulates ghosts is worse than one that is briefly stale.

Once cloned, everything downstream is ordinary: grep works with no caveat, and
**deep memory can index the mount** like any other folder — Drive documents
become searchable long-term memory, which is the pay-off for the whole exercise.

---

## Phase 4 — UI (SHIPPED)

In the VFS panel: a **Connect Drive** header button opening a `DriveConnect`
modal that browses Drive folders (folders only), names the mount, and offers a
clone-to-Markdown checkbox; a **GDRIVE** badge + lock on mounted projects; a
**Drive** filter chip; and per-mount hover controls — Refresh (with a
"N file(s), M cloned" note), a clonemd toggle, and Unmount (which asks about
deleting cloned Markdown only when there is any). The card's second line shows
`cloned to Markdown` or `on-demand · N not cloned · synced <time>`, so a partly
virtual folder is never a surprise. Backend enriched `/fd/vfs/projects` with
per-mount `total`/`cloned`/`uncloned` counts and stamps `synced_at` on
mount/refresh/clone.

Verified live end to end against a fake Drive: connect → browse → mount with
clone → the tree materialises (`CEE VC landscape.md` from a Google Doc,
`notes.txt` verbatim, `chart.png` a placeholder), the cloned doc previews real
content, refresh reports counts, the clonemd toggle flips card state, both
themes pass contrast, no console errors.

---

## What I would watch

- **Rate limits.** Lazy listing plus hydrate-on-read means bursty call patterns.
  Phase 0's backoff is not optional; a shared token across users (until Phase 0.1)
  makes one user's mount able to exhaust everyone's quota.
- **Placeholders in grep.** The warning is only honest if the skip list is exact.
  If a placeholder is ever searched *as text*, its marker line will match queries
  like "drive" or "cloned". The manifest, not a heuristic, must drive the skip.
- **`.docx.md` naming** is the one piece of user-visible ugliness. Worth a look
  before building.
- **Deletion semantics on unmount**: removing a mount should delete the
  placeholder tree, and should *ask* before deleting cloned Markdown, which the
  user may have come to treat as their own files.

## Deliberately out of scope

Write-back to Drive; realtime sync (`changes` API / webhooks — none exists today,
and manual refresh is the brief); Shared Drives beyond what the picker already
lists; OCR of images; and any attempt at a general virtual-filesystem backend
behind `project_root()`.
