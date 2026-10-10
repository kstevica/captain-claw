#!/usr/bin/env bash
# End-to-end check of the smart-glasses HUD (/hud/) at 600x600 with the Meta
# Ray-Ban Display user agent: boots a throwaway Flight Deck (auth on) and a
# mock agent, pairs the "glasses", then walks chat, files and data with the
# D-pad keys. See README.md. Needs: uv, node, and Playwright with Chromium.
#
#   flight-deck/e2e/hud/run.sh            # build must already be in static/
#   FD_PORT=25190 AGENT_PORT=24090 ./run.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
RUN="${E2E_RUN_DIR:-$HERE/.run}"
FD_PORT=${FD_PORT:-25190}
AGENT_PORT=${AGENT_PORT:-24090}
export E2E_RUN_DIR="$RUN"

PIDS=()
cleanup() { for p in "${PIDS[@]}"; do pkill -P "$p" 2>/dev/null || true; kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT

rm -rf "$RUN"; mkdir -p "$RUN/fd-data"
cd "$REPO"
uv run python "$HERE/mock_agent.py" "$AGENT_PORT" >"$RUN/mock.log" 2>&1 & PIDS+=($!)
FD_DATA_DIR="$RUN/fd-data" FD_AUTH_ENABLED=true FD_JWT_SECRET=e2e-only-not-a-secret \
  uv run python -m captain_claw.flight_deck.server --host 127.0.0.1 --port "$FD_PORT" >"$RUN/fd.log" 2>&1 & PIDS+=($!)

for _ in $(seq 1 120); do
  curl -fsS "http://127.0.0.1:$FD_PORT/fd/auth/status" >/dev/null 2>&1 && break
  sleep 0.5
done

# First user on a fresh deck may register; record the mock as their agent.
REG=$(curl -fsS -X POST "http://127.0.0.1:$FD_PORT/fd/auth/register" -H 'Content-Type: application/json' \
  -d '{"email":"pilot@example.com","password":"correct-horse-battery","display_name":"Pilot"}')
USER_ID=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["user"]["id"])' "$REG")
MOCK_PID=$(pgrep -f "mock_agent.py $AGENT_PORT" | tail -1)
python3 - "$RUN/fd-data/.processes.json" "$USER_ID" "$AGENT_PORT" "$MOCK_PID" <<'EOF'
import json, sys
path, owner, port, pid = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
json.dump({"scout": {"name": "Scout", "description": "Mock research agent", "web_port": port,
                     "web_auth": "mocktoken", "pid": pid, "owner": owner, "provider": "mock",
                     "model": "mock-1"}}, open(path, "w"), indent=2)
EOF

BASE="http://127.0.0.1:$FD_PORT" node "$HERE/hud_e2e.cjs"
