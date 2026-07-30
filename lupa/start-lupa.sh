#!/usr/bin/env bash
#
# Start the Lupa BFF (serves the SPA + /api) against a running Flight Deck.
#
# FD_JWT_SECRET must match start-fd.sh. Product data (streams, briefs, packs
# registry) persists in LUPA_DATA_DIR (./lupa-data, gitignored). The committed
# SPA bundle in lupa/api/static is served as-is — no npm step needed.
#
# Start FD first (./lupa/start-fd.sh), then this. Open http://localhost:25180
# and register — the first registered user becomes admin (and a Studio creator).
#
# Usage: ./lupa/start-lupa.sh      (override PORT/LUPA_FD_URL/etc via env)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PORT="${LUPA_PORT:-25180}"
export LUPA_FD_URL="${LUPA_FD_URL:-http://127.0.0.1:25080}"
export FD_JWT_SECRET="${FD_JWT_SECRET:-lupa-local-secret}"
export LUPA_DATA_DIR="${LUPA_DATA_DIR:-$ROOT/lupa-data}"

PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || PY="python3"

echo "Lupa BFF → http://127.0.0.1:${PORT}   (FD: ${LUPA_FD_URL}, data: ${LUPA_DATA_DIR})"
exec "$PY" -m uvicorn --app-dir lupa/api lupa_api.server:app \
  --host 127.0.0.1 --port "$PORT"
