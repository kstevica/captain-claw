#!/usr/bin/env bash
#
# Start Flight Deck for Lupa.
#
# FD and the Lupa BFF MUST share one JWT secret: the BFF verifies FD-issued
# access tokens locally and mints owner tokens for scheduled briefs, factory
# runs and second opinions. FD started without FD_JWT_SECRET generates a random
# per-process secret the BFF cannot share — so it is set here and must match
# start-lupa.sh.
#
# Auth stays enabled (the desktop build disables it; that mode won't work for
# Lupa). This uses FD's default data dir + model registry — your real local FD.
#
# Usage: ./lupa/start-fd.sh        (override PORT/FD_JWT_SECRET via env)

set -euo pipefail

# Repo root = the directory above this script's dir (lupa/..).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PORT="${FD_PORT:-25080}"
export FD_JWT_SECRET="${FD_JWT_SECRET:-lupa-local-secret}"

PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || PY="python3"

echo "Flight Deck → http://127.0.0.1:${PORT}  (auth on, shared JWT secret set)"
exec "$PY" -m captain_claw.flight_deck.server --host 127.0.0.1 --port "$PORT"
