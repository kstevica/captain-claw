#!/usr/bin/env bash
# Agent groups in the kiosk, end to end: boots a throwaway Flight Deck (sign-in
# on) with a kiosk user, an admin, five running mock agents and six stopped ones,
# then drives the real bundle with Playwright. See README.md.
#
#   cd flight-deck && npm run build && cd .. && flight-deck/e2e/groups/run.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
RUN="${E2E_RUN_DIR:-$HERE/.run}"
FD_PORT=${FD_PORT:-25392}
BASE_AGENT=${AGENT_PORT:-24392}
PY=(uv run python)
export E2E_RUN_DIR="$RUN"
PIDS=()
cleanup() { for p in "${PIDS[@]}"; do pkill -P "$p" 2>/dev/null || true; kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT

rm -rf "$RUN"; mkdir -p "$RUN/fd-data"
cd "$REPO"
MOCK="$HERE/../onboarding/mock_agent.py"
for i in 0 1 2 3 4; do
  persona=research; [ $((i % 2)) = 1 ] && persona=inbox
  "${PY[@]}" "$MOCK" $((BASE_AGENT + i)) "$persona" >"$RUN/agent$i.log" 2>&1 & PIDS+=($!)
done
FD_DATA_DIR="$RUN/fd-data" FD_AUTH_ENABLED=true FD_JWT_SECRET=e2e-only-not-a-secret \
  "${PY[@]}" -m captain_claw.flight_deck.server --host 127.0.0.1 --port "$FD_PORT" >"$RUN/fd.log" 2>&1 & PIDS+=($!)
FD="http://127.0.0.1:$FD_PORT"
for _ in $(seq 1 120); do curl -fsS "$FD/fd/auth/status" >/dev/null 2>&1 && break; sleep 0.5; done

json() { python3 -c 'import json,sys; d=json.loads(sys.argv[1]); print(eval(sys.argv[2], {}, {"d": d}))' "$1" "$2"; }
curl -fsS -X POST "$FD/fd/auth/register" -H 'Content-Type: application/json' \
  -d '{"email":"admin@example.com","password":"correct-horse-battery","display_name":"Admin"}' >/dev/null
LOGIN="$(curl -fsS -X POST "$FD/fd/auth/login" -H 'Content-Type: application/json' \
  -d '{"email":"admin@example.com","password":"correct-horse-battery"}')"
ADMIN=$(json "$LOGIN" 'd["access_token"]')
ADMIN_ID=$(json "$LOGIN" 'd["user"]["id"]')
api() { curl -fsS -X "$1" "$FD$2" -H "Authorization: Bearer $ADMIN" -H 'Content-Type: application/json' -d "$3"; }
USER_ID=$(json "$(api POST /fd/admin/users '{"email":"maya@example.com","password":"correct-horse-battery",
  "display_name":"Maya Novak","role":"user","kiosk":true,"must_change_password":false}')" 'd["user"]["id"]')
api PUT /fd/admin/config '{"kiosk_mode":"selected"}' >/dev/null

pids=(); for i in 0 1 2 3 4; do pids+=("$(pgrep -f "mock_agent.py $((BASE_AGENT + i)) " | tail -1)"); done
python3 - "$RUN/fd-data/.processes.json" "$USER_ID" "$BASE_AGENT" "$ADMIN_ID" "${pids[@]}" <<'PY'
import json, sys
path, owner, base, admin = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
pids = [int(x) for x in sys.argv[5:]]
def entry(name, desc, i=None):
    e = {"name": name, "description": desc, "owner": owner, "provider": "anthropic", "model": "claude-sonnet-5-5",
         "web_auth": "mocktoken"}
    if i is None:
        e.update(web_port=0, pid=None, stopped=True)
    else:
        e.update(web_port=base + i, pid=pids[i])
    return e
json.dump({
    "research-assistant": entry("Research Assistant", "Market and competitor research", 0),
    "inbox-helper": entry("Inbox Helper", "Morning email summaries and reply drafts", 1),
    "market-watch": entry("Market Watch", "Daily moves in our sector", 2),
    "legal-reader": entry("Legal Reader", "Contract and term sheet review", 3),
    "travel-desk": entry("Travel Desk", "Flights, hotels and visas", 4),
    "report-writer": entry("Report Writer", "Monthly board reports"),
    "grant-scout": entry("Grant Scout", "Open calls and deadlines"),
    "home-admin": entry("Home Admin", "Bills and appointments"),
    "admin-scout": {**entry("Admin Scout", "The admin's first agent"), "owner": admin},
    "admin-clerk": {**entry("Admin Clerk", "The admin's second agent"), "owner": admin},
    "admin-third": {**entry("Admin Third", "The admin's third agent"), "owner": admin},
}, open(path, "w"), indent=2)
PY
BASE="$FD" "${PY[@]}" "$HERE/check.py" "$@"
