# Deep memory → Flight Deck (system level)

Move Typesense-backed deep memory from an **agent-level** capability (a Python
object hanging off one agent, configured in that agent's `config.yaml`, served
by that agent's own aiohttp server) to a **system-level** one (a shared,
multi-tenant archive owned by Flight Deck, reachable by every agent and by the
dashboard) — the same shape the VFS already has.

Status: **Phases 0–3 shipped** (backend, tenancy, VFS indexing, agent proxy, FD UI,
Connections card). **Phase 4 (retire the agent-level REST surface) not started.**

## What shipped, in one picture

```
  user drops a file in VFS ──► vfs_routes write/upload/rename/delete
                                     │  (opt-in per project)
                                     ▼
                            deep_memory_service
                          hash → skip | delete+reindex
                                     │
                                     ▼
                               Typesense (loopback)
                                     ▲
   agent: typesense tool ──► /fd/deep-memory/agent/search ──┘
        (holds NO key)          FD resolves the owner
```

A hit comes back as `vfs:<project>/<path>` — the exact string the `read` tool
accepts — so the agent finds the file *and can then open it*. Search returns a
pointer, not just a snippet.

---

## Why

Deep memory today is agent-level in all seven dimensions that VFS is
system-level in:

| | VFS (system) | Deep memory (agent) |
|---|---|---|
| Core module | `captain_claw/vfs.py`, zero FD imports | `deep_memory.py`, but instantiated per-agent |
| Tenancy | `<fd-data>/vfs/<user-id>/…` path segment | **none — one flat collection** |
| Identity | `vfs_user()` from `FD_OWNER_ID` env | none |
| Run binding | `CLAW_VFS_PROJECT` env at every spawn site | Python object passed in-process |
| Config | essentially none (convention) | `DeepMemoryConfig`, per-agent `config.yaml` |
| REST | `/fd/vfs/*`, `Depends(get_current_user)` | `/api/deep-memory/*` on the **agent's** server |
| UI | `VFSPage.tsx` + `vfsStore.ts` | a static page on the agent server |

The load-bearing gap is the second row. `deep_memory.collection_name`
(`config.py`) is a single flat string, so **every** agent pointed at the same
Typesense host reads and writes the same archive. There are 49 agent configs
under `fd-data/`. That is fine for one operator on a laptop and is a
disclosure bug the moment Flight Deck has two users.

---

## Phase 0 — make it actually work (SHIPPED)

Deep memory had never once run hybrid search. `embedding_dims` defaulted to
`1536` while the default provider (model2vec `potion-base-8M`) emits `256`, so
the guard in `_embed()` discarded every vector at both index and query time.
The collection declared an `embedding` field that no document ever populated.

- `embedding_dims: 0` now means *auto-detect*; `ensure_collection()` resolves
  width as **live collection schema > probed provider > config**, and repairs a
  mismatched collection in place via `PATCH /collections` (safe precisely
  because no vectors were ever stored).
- `reembed_all()` backfills vectors from each document's stored `text` — no
  re-ingestion from sources that may no longer be reachable.
- Search moved from `GET /documents/search` to `POST /multi_search`. Typesense
  caps a query string at 4000 chars and an inlined vector costs ~21 per
  dimension, so the GET path breaks at ~190 dims — **below the 256 the default
  provider emits.** The old path only ever "worked" because vectors were off.
- Scoring rebuilt. The old `max(text_match, 1/(1+distance))` always returned
  `text_match`, a 19-digit integer, which reached the LLM as
  `score=1157451470635796601.00`. Replaced with an absolute 0..1 measure: the
  better of keyword coverage and cosine similarity. Explicitly **not**
  `rank_fusion_score` — measurement showed that decays as `0.3 × 1/rank`, i.e.
  it is positional, so filtering on it merely re-implements `max_results`.
- A relevance floor (`min_score`, default `0.12`) gates prompt injection, and
  the previously-dead `_should_search_deep_memory()` now relaxes that floor
  when the user explicitly asks for the archive.

Measured on the live 11-document collection: `"folk legends from the Balkans"`
went from **0 hits to 4**.

**Calibration caveat, carried forward:** on a labelled pair set, related and
unrelated similarities *overlap* under `potion-base-8M` (related 0.13–0.59,
unrelated −0.11–0.14). A static bag-of-embeddings model ranks acceptably but is
not calibrated for absolute similarity, so the floor is deliberately
permissive. Phase 1 should offer deep memory its own embedder setting
(`nomic-embed-text` via Ollama) rather than always inheriting semantic
memory's; that is what would make a tighter floor worth setting.

---

## The tenancy decision

**Collection per user** (`captain_claw_dm_<user-id>`) mirrors VFS's path
segment most literally, but Typesense holds every collection's index in RAM
with per-collection overhead, cross-user sharing needs `multi_search` fan-out,
and collection count becomes a scaling axis. Rejected.

**One collection + `owner_id` field**, with the tenant filter applied inside
`DeepMemoryIndex.search()` — ANDed onto any caller-supplied `filter_by`, and
parenthesised, so a caller cannot widen its own scope with an `||`.

The first draft of this plan proposed enforcing that with Typesense **scoped
API keys** — a key carrying an embedded `filter_by` the client cannot override.
That works, and was verified against Typesense 30.1 (an alice-scoped key
overriding `filter_by` to bob returns 0 hits; a tautology still returns only
alice). But scoped keys cover **search only** — Typesense has no write
equivalent — so they leave the write half needing a separate mechanism.

**Superseded by the FD-proxy design (shipped).** Agents hold no Typesense key
at all. They call Flight Deck, FD resolves the owner from its own records, and
one boundary enforces reads *and* writes. Typesense can bind to loopback and
the admin key never leaves the FD process — strictly stronger than scoped keys,
and simpler. Scoped keys remain available if a future caller ever needs to
query Typesense directly.

**Agent identity is never something the agent asserts.** `agent_secret.py` is a
*single shared secret* across all agents: it proves the caller is an FD-spawned
process, so it is a gate, not an identity. `X-Agent-Slug` is likewise supplied
by the agent itself. The owner is therefore resolved from FD's own records via
`_resolve_agent_owner_by_auth()` (the per-agent `web_auth` token, unique and
stored in the process registry and Docker label at spawn), falling back to
source-port lookup — the same ladder `code_routes._resolve_agent_caller` uses.

---

## Phases 1–2 — core, tenancy, VFS indexing, agent proxy (SHIPPED)

**Schema.** `owner_id` and `content_hash` added to
`_COLLECTION_SCHEMA_TEMPLATE`, both `optional` so they PATCH onto the existing
collection without a reindex.

**Service** — `flight_deck/deep_memory_service.py`, the only thing that talks
to Typesense. Owns the process-wide index, the opt-in registry, and the
freshness hooks.

**Opt-in per project** — `.vfs-index.json` at the user root, the same
sidecar-registry shape as `.vfs-links.json`. Off by default: a VFS project is a
working folder, and indexing it wholesale would bury the useful text under
build output. Eligibility is additionally gated on suffix, size (2 MB), hidden
files, and a `_SKIP_DIRS` set (`node_modules`, `.git`, `__pycache__`, …).

**Freshness is automatic** (hooks in `vfs_routes`, all fire-and-forget so a
Typesense outage can never fail a file write):

| event | behaviour |
|---|---|
| write / upload | re-index — but a matching `content_hash` short-circuits to a no-op *before* chunking or embedding |
| rename | drop the old reference, index the new |
| delete file | drop that reference |
| delete dir / project | drop every reference under the prefix |

Re-index is **delete-then-write**, not upsert: chunk ids derive from content,
so a file that shrinks would otherwise leave orphaned tail chunks searchable
forever.

**Summarising is per-document and off by default.** `index_document(...,
summarize=False)`. The L1/L2 summariser is one LLM call per ~1400-char chunk,
so auto-summarising would turn a 100 KB upload into ~70 calls. Without it a
chunk still carries full text and both search halves work — only the compact
prompt-injection layers are absent.

**Routes** — `flight_deck/deep_memory_routes.py`, `/fd/deep-memory/*`,
registered next to `vfs_router`. Dashboard routes use
`Depends(get_current_user)` + `_eff_owner()` against the existing
`resource_shares` table. Agent routes (`/agent/search`, `/agent/index`) gate on
the shared agent secret and resolve the owner from FD's records.

**Connection lives in Flight Deck** — Connections → *Typesense (Deep Memory)*
(`TypesenseConnection.tsx`), persisted in the FD `system_settings` table under
`deep_memory.connection` and primed into the service at server startup. The API
key is masked on read, and posting the mask back means "keep the existing key"
so an untouched form never wipes it. Tuning (chunking, embedding width,
`min_score`) stays in `config.yaml` — only the *connection* moved.

**The tool is permanently enabled** — `typesense` joined
`ToolsConfig._ALWAYS_ENABLED` beside `vfs`, so no agent can be silently unable
to see its own long-term memory. It is registered eagerly with `deep_memory=None`
and hidden from the model in exactly one case, mirroring how Google tools hide
until OAuth connects: *standalone* (no `FD_URL`) with no API key and deep memory
off, where every call could only fail. Under Flight Deck it always shows — FD
can gain a connection at any moment and its error names where to fix it.

**Tool proxy** — under Flight Deck (`FD_URL` set) the `typesense` tool routes
**every** action (index, search, delete) through FD and never constructs a
Typesense client. The agent also stops building a local `DeepMemoryIndex`
entirely under FD: a second, unscoped path to the same server opened from the
agent's own config is exactly the multi-tenant hole the proxy closes.
From the agent's side deep memory is simply always present; if FD has no
connection, FD answers 503 with *"Open Flight Deck → Connections → Typesense"*
and there is nothing for the agent to fix on its end. Standalone (no `FD_URL`),
the direct path is unchanged — that is what the single-user install uses.

**Verified end-to-end** against live Typesense: opt-in respected (a file
written while indexing was off is absent from the index entirely), binaries and
`node_modules` skipped, unchanged re-index → `unchanged`, rewrite → old content
gone and new content found, delete → gone, and **cross-tenant search returns
nothing** (alice: 0 hits on bob's file; bob: 1).

**Tests** — 37 unit tests across `test_deep_memory.py` and
`test_deep_memory_vfs.py`. Full `tests/test_flight_deck` is 9 failed / 1060
passed both with and without these changes — no regressions.

---

## Phase 3 — FD UI (SHIPPED)

`pages/DeepMemoryPage.tsx` + `components/deepmemory/DeepMemoryBrowser.tsx` +
`stores/deepMemoryStore.ts`, mirroring the `VFSPage` / `VFSBrowser` / `vfsStore`
trio (including `_authedFetch`'s 401→refresh→retry and `owner` threading).
Nav entry sits beside VFS in the Files group; `ViewMode` gained `'deep-memory'`.

- **Health panel** — collection width vs provider width side by side, green
  when they agree ("hybrid search active"), red when they don't. This is the
  dashboard that would have caught the Phase 0 bug: a collection declaring 1536
  while the provider emits 256, every vector discarded, and nothing on screen
  to say so.
- **Folder list** merges every VFS project with the opt-in registry, so a
  folder can be enabled before it has ever been indexed. Per-folder checkbox
  (auto-index), *Index now*, and a confirm-gated *Remove*.
- **Summarise** is a single explicit toggle, off by default, with the cost
  spelled out in its tooltip.
- **Score bars** are meaningful across queries because the score is absolute,
  not positional — green ≥0.5, amber ≥0.25, grey below.

**Verified in the running app** against a live Typesense: nav → page → health
green at 256/256; a purely semantic query ("a huge man who rescued people from
the water", no shared keywords) returned `reygoch.md` at 0.57 with the weak
match greyed at 0.12; the opt-in checkbox wrote through to the registry; a file
written via `POST /fd/vfs/write` was indexed **without** pressing Index now, and
a `DELETE /fd/vfs/entry` unlinked it — the automatic hooks, through the real
routes. No console errors.

**Light theme.** Non-zinc colours need explicit `dark:` variants: the zinc scale
is remapped by CSS vars, but emerald/sky/amber/red are not, so tints tuned for a
dark background measured **1.46:1 and 1.69:1** on the light one. Fixed to
**5.14:1**, passing WCAG AA, verified by sampling the rendered pixels rather
than by eye.

---

## Phase 4 — retire the agent-level surface

Delete `captain_claw/web/rest_deep_memory.py` and its wiring
(`web_server.py:1757-1783`, `:2869-2874`, `:2905`), leaving the agent with only
the `typesense` tool. This mirrors VFS's crispest signal: it has **no**
agent-level REST surface at all.

Keep `DeepMemoryIndex` importable standalone so the non-FD single-user path
still works — that is what the `"local"` owner fallback is for.

---

## Order, and what each phase is worth

Phase 1 is the only one that changes correctness (tenancy). Phases 2–4 are
surface. Phase 1 alone, shipped, closes the disclosure gap; the rest can follow
at leisure.

**Migration of live data:** additive. `PATCH` the collection to add `owner_id`
and `scope` as optional, backfill existing rows to the primary owner, then
start writing the fields. No reindex, no downtime — the same property that made
the Phase 0 repair safe.

---

## Deliberately deferred

- **Collection-per-user**, revisit only if one shared collection's RAM becomes
  the constraint. At 256 dims the vector cost is `7 × 256 × N` bytes ≈ **1.8 GB
  per million chunks** across all tenants — not a near-term concern. (At the
  old 1536 it would have been 10.75 GB, which is the real reason the width bug
  mattered beyond correctness.)
- **Per-tenant Typesense instances** — full isolation, but an ops multiplier.
- **Cross-user search** (search *everything* shared with me) — needs
  `multi_search` fan-out plus result merging; the schema above allows it later.
- **Retention / eviction.** Nothing currently ages out of deep memory. A
  multi-tenant archive with no eviction policy grows without bound.
