# Agent export, import and backups

Status: plan, built in five phases (one PR each, stacked: each phase's branch
starts from the one before it). Nothing is merged until all five are reviewed.

## Who it is for

- **A consultant** prepares agents for a client: onboards them (documents,
  lessons, skills, intents), lets them learn (insights, reflections), then
  exports them and imports them into the client's Flight Deck. Later they ship
  an improved version without wiping what the client added since.
- **Anyone who runs a deck** backs agents up and restores them, by hand or on a
  schedule.

Both use one machinery: an **agent bundle**, a `.captain` file.

## What an agent is made of

A process agent lives in `DATA_DIR/<slug>/`. `HC` below is its own
`~/.captain-claw`, i.e. `data/home-config-parent/.captain-claw/`.

| Category | What | Where |
|---|---|---|
| identity (always) | name, description, model (never its key), tools, cognitive mode, eco mode, vision | Flight Deck registry, `config.yaml`, `HC/cognitive_mode.txt`, `HC/eco_mode.txt` |
| | personality | `HC/personality.md` |
| | instructions | the owner's user settings in Flight Deck (`fd:process-fleet-instructions`) |
| onboarding | documents | `data/workspace/knowledge/` |
| | lessons, the setup record | `.owner-setup.json`, `HC/lessons.json` |
| | skills | `HC/skills/`, `data/skills/` |
| | practice, exams | `practice.json`, `exams/` |
| memory | insights | `HC/insights.db` |
| | intentions | `data/sessions/intentions.db` |
| | reflections | `HC/reflections/*.md` |
| | intuitions | `HC/intuitions.db` |
| data | datastore tables | `HC/datastore.db` |
| conversations | sessions | `data/sessions/sessions.db` |
| | topics, sister sessions, saved-file attribution | `HC/conversation_topics.db`, `HC/sister_sessions.db`, `data/sessions/saved_attribution.db` |
| | Flight Deck's chat log | `chat_sessions` / `chat_messages` rows for `proc-<slug>` (the owner's) |
| files | workspace files | `data/workspace/` (except `knowledge/`) |
| schedules | scheduled tasks | Flight Deck's scheduler store |

**Never in a bundle**: `.fd-owner`, `process.log`, `.env`, every key and token
in `config.yaml` (model key, web auth, BotPort secret, Telegram/Discord/Slack
tokens), caches (`.cache`, `Library`), spools (`tool_audit_spool.db`,
`turn_register_spool.db`) and indexes that are rebuilt from the rest
(`data/home-config/memory.db` — semantic memory — and Typesense deep memory).

**Stays on the deck** (listed by name in the bundle, so the importer can see
what to connect): Google/Outlook, IMAP inboxes, Telegram/WhatsApp links, MCP
servers the agent may use, external memory, custom tools.

## The bundle

A zip with the extension `.captain`:

```
manifest.json          format, versions, scope, counts, sizes and sha256 of every part
fd/agent.json          registry fields, model settings without keys, tools, instructions
fd/chats.json          Flight Deck's chat log (conversations)
fd/schedules.json      scheduled tasks (schedules)
fd/connections.json    what it was connected to, by name only
agent/<path>           the agent's own files, at their place in its folder
stores/<name>.db       each SQLite store, as a consistent snapshot
```

- **SQLite stores** are snapshotted with SQLite's backup API, never copied as
  files: agents run in WAL mode, and a plain copy of `insights.db` misses
  what is still in `insights.db-wal`. A snapshot that was filtered (phase 2)
  is `VACUUM`ed so dropped rows don't linger in free pages.
- **Into a new agent**, a snapshot becomes the store once it passes SQLite's
  integrity check and holds nothing but tables, indexes, full-text tables and
  their triggers (no views, no other virtual tables). The agent's own startup
  migrations (`ADD COLUMN`, `IF NOT EXISTS`) bring an older store up to date.
- **Into an existing agent** (phase 3), the file is never used as the store:
  its rows are read (with `trusted_schema=OFF`) and copied by column name into
  the agent's own database, whose own triggers keep its search indexes right.
  Copying by column name also survives schema differences between decks.
  (JSONL was considered; a snapshot keeps types, BLOBs and full-text tables
  exactly, and the column-name copy gives the same safety.)
- **Validation on import**: zip entries must be relative, with no `..`,
  absolute paths or symlinks; size caps per entry and in total
  (`FD_AGENT_BUNDLE_MAX_MB`, default 2048); every part's sha256 must match the
  manifest; the bundle's format version must be one this deck reads.

## Phases

### Phase 1 — bundle format, full backup, restore

- `captain_claw/flight_deck/agent_bundle/`: the parts catalogue, SQLite
  snapshot and row copy, manifest, export, inspect and restore.
- Restore makes a **new agent** through the normal spawn path
  (`_spawn_process_locked`), so the agent-count limit, ports, web auth, owner
  marker, tenant profile, instructions and the provider key resolution all
  apply as for any new agent. The bundle is unpacked inside the spawn,
  before the process starts, like an onboarding package.
- Routes: `POST /fd/agent-bundle/export/{slug}`, `GET
  /fd/agent-bundle/download/{id}`, `POST /fd/agent-bundle/upload` (returns
  what is inside), `POST /fd/agent-bundle/import/{upload_id}`.
- UI: agent ⋯ → **Back up / export**; New agent → **Import from a file**.
- Process agents only; Docker agents say so.

### Phase 2 — scopes, review, secrets, connect checklist

- Presets **Template** (identity + onboarding), **Trained** (+ memory, data,
  schedules), **Full backup** (everything), and a checklist of categories.
  Sessions are separate from memories.
- **Review before export**: insights by category. `fact`, `workflow` and
  `decision` are ticked by default; `contact`, `preference`, `deadline`,
  `project` are shown unticked (they are usually about the person who
  trained it, or other clients). The owner profile never travels.
- **Secrets**: never in an export. A Full backup may carry them, encrypted
  with a passphrase (scrypt + AES-GCM); without one they stay out.
- **Import**: model choice — keep its model, or one of the importer's tiers
  (default: the tier it was made on, when this deck has it). After import a
  **Connect these** list: model key, mail, Telegram, MCP servers, and so on.
- Schedules arrive switched off.

### Phase 3 — import into an existing agent

- Target: a new agent, or one of the importer's own agents.
- **Reset**: the agent's memory is replaced by the bundle's. An automatic
  backup of the agent is taken first and offered to restore.
- **Append**: rows go in through each store's own dedup: insights by
  `entity_key`, then by matching text; intentions by text;
  reflections merged with the existing `reflection_merge` prompt when the
  agent is up, else kept as an older reflection; lessons and skills by name.
- The agent is stopped for the write and started again.

### Phase 4 — versions and updates

- Every exported item carries the bundle's **package id** and **version**
  (insight and intention tags, lesson fields, skill and document records).
- Export asks for a package name; the version goes up by one per export of the
  same package.
- Import into an agent that already holds an earlier version of the same
  package offers **Update**: the package's earlier items are replaced by the
  new ones, everything the client added stays. Imported lessons are the base
  layer (`layer: "base"`), so the client adds on top but can't change them.

### Phase 5 — scheduled backups, destinations, signing

- **Scheduled backups** per agent or for all of a user's agents: daily or
  weekly, keep the last N, taken when the agent is asleep or idle where it can.
- **Destinations**: the deck's own folder (default), or a folder in the
  user's files (VFS), which can be a Google Drive mount.
- **Signing**: each deck has an Ed25519 key; bundles are signed, and import
  shows "Signed by <deck> (fingerprint)" or "Unsigned". Optional list of
  trusted fingerprints.

## Not in scope

- Docker agents (Phase 1 refuses them with a clear message).
- Whole-deck backup (users, settings, every agent at once).
- S3 or other remote destinations beyond the VFS.
- Members' own chats with a shared agent: they are theirs, never exported.
