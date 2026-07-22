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

## Phase 0 — unblock (nothing works without this)

The Drive integration is not currently in a state a filesystem can sit on.

1. **Per-user Google tokens.** `system_settings` is `(key, value, updated_at)` —
   **no `user_id`** (`flight_deck/db.py:94-98`), so one Google account is shared
   by every Flight Deck user, and `POST /fd/google/config` by any logged-in user
   overwrites everyone's connection. VFS is per-user
   (`<fd-data>/vfs/<fd-user-id>/…`). A per-user mount has no per-user token to
   hang off. Needs a `google_oauth_user_tokens(user_id, …)` table and a migration
   of the existing global row to the primary owner.
   *This is the one item that is a genuine blocker rather than a nuisance.*

2. **Pagination.** `google_drive.py:35` requests `nextPageToken` and **nothing
   ever reads it**. A folder with >100 children silently truncates — a mount
   would show a partial tree and never say so.

3. **Accept `drive.readonly`.** `google_drive.py:31,196-202` hardcodes the full
   read/write `drive` scope and refuses to run without it, so a read-only mount
   currently requires granting write access to all of Drive. Widen the check the
   way `google_mail.py:287-293` already does.

4. **Retry/backoff.** There is none anywhere in the Google tools; 429 is reported
   and dropped (`google_drive.py:239-243`). A mount doing per-directory listings
   will hit user-rate-limit 403s on its first real folder. `tenacity` is already
   a declared dependency and unused.

5. **Structured returns.** Every action returns prose for an LLM. The mount needs
   metadata, not sentences — promote `_get_file_metadata` (`:717`) and add a
   listing that returns dicts.

Also worth folding in, cheap: escape `folder_id` in the query
(`google_drive.py:266` is unescaped; `_gws_drive.py:100` shows the fix), and drop
the dead `drive.delete` catalog entry (`action_catalog.py:124` calls an action
that does not exist).

---

## Phase 1 — the mount

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

## Phase 4 — UI

In the VFS panel, beside the existing link affordances: a "Connect Google Drive"
entry in the folder picker, a cloud badge on mounted projects, a **clonemd**
toggle, a **Refresh** button with last-synced time, and — importantly — a visible
count of un-cloned files, so "this folder is partly virtual" is never a surprise.

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
