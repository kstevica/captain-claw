# Lupa — the agentic research desk (and Kalup, its vertical factory)

Lupa is a standalone product on top of Captain Claw's Flight Deck. It never
imports `captain_claw` — everything goes over FD's HTTP API on loopback.

```
browser ──► lupa/api (BFF :25180, serves the SPA + /api) ──► FD :25080
```

## Run it locally against your real FD

FD and the BFF must share one JWT secret — the BFF verifies access tokens
locally and mints short-lived owner tokens for scheduled briefs, factory runs
and second opinions. FD started **without** `FD_JWT_SECRET` generates a random
per-process secret the BFF cannot share.

1. **Start FD with the shared secret** (launch config `fd-for-lupa`, or):

   ```bash
   FD_JWT_SECRET=lupa-local-secret ./.venv/bin/python -m captain_claw.flight_deck.server --host 127.0.0.1 --port 25080
   ```

   Auth stays enabled by default when running the server module directly.
   (The desktop build runs with auth off — that mode won't work for Lupa.)

2. **Start the BFF** (launch config `lupa-live`, or):

   ```bash
   FD_JWT_SECRET=lupa-local-secret LUPA_FD_URL=http://127.0.0.1:25080 \
   LUPA_DATA_DIR=./lupa-data \
   ./.venv/bin/python -m uvicorn --app-dir lupa/api lupa_api.server:app --host 127.0.0.1 --port 25180
   ```

3. Open http://localhost:25180 and **register** — FD makes the first
   registered user an admin, which also makes you a Pack Studio creator.

4. Commission something. Runs use your FD's own model registry (Library
   tiers); if a run errors immediately, configure providers/models in FD
   first — Lupa adds no model config of its own.

Notes:
- The SPA bundle is committed (`lupa/api/static`) — no npm step needed.
  Rebuild with `cd lupa/web && npm run build` after UI changes.
- BFF deps live in the main venv already (`fastapi httpx uvicorn aiosqlite
  PyJWT python-multipart`); on a fresh machine: `pip install -e lupa/api`.
- Product data (streams, briefs, packs registry) lives in `LUPA_DATA_DIR`
  (default `./lupa-data`, gitignored). Repo packs under `lupa/packs/` are
  imported at startup as published system packs; runtime edits win afterward.

## Demo without Captain

Launch config `lupa-demo` runs a fake FD (`lupa/api/dev/fake_fd.py`) plus the
BFF: any email/password signs in, runs simulate in seconds, the Pack Studio
factory line works end to end.

## Tests

```bash
cd lupa/api && ../../.venv/bin/python -m pytest tests/ -q
```

## Env reference (BFF)

| var | default | meaning |
|---|---|---|
| `LUPA_FD_URL` | `http://127.0.0.1:25080` | FD base URL |
| `FD_JWT_SECRET` | — | shared with FD; required for briefs/factory/second-opinion token minting |
| `LUPA_DATA_DIR` | `./lupa-data` | product SQLite + assets |
| `LUPA_PACK` | `research-desk` | the default desk at `/` |
| `LUPA_PACKS_DIR` | `lupa/packs` | seed packs directory |
| `LUPA_PORT` / `LUPA_HOST` | `25180` / `0.0.0.0` | bind (use the `lupa-api` script) |
| `LUPA_BRIEF_TICK_SECONDS` | `60` | standing-brief scheduler tick |
| `LUPA_FACTORY_POLL_SECONDS` / `LUPA_FACTORY_TIMEOUT_SECONDS` | `2` / `900` | Studio generate/evaluate polling |

Hosted deployments additionally set on **FD**: `FD_LOCKDOWN=1`,
`FD_CORS_ORIGINS`, `FD_AGENT_SHARED_SECRET`, `FD_GLASSES_BRIDGE_TOKEN`
(see docs/research-desk-product-plan.md, Phase 0a).
