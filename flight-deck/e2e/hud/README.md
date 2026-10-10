# HUD end-to-end check

Drives the smart-glasses HUD (`/hud/`) the way a Meta Ray-Ban Display does:
a 600×600 viewport, the Display's user agent, and only arrow keys and Enter.
A desktop test can't open Meta's text composer, so the script simulates it:
it sets the field once, then fires `input` + `change`.

`run.sh` does the following:

1. Boots a throwaway Flight Deck with auth on (data in `.run/`).
2. Starts `mock_agent.py`, a fake captain-claw agent with the chat WebSocket,
   files and datastore APIs, and records it as the test user's process agent.
3. Runs `hud_e2e.cjs`, which covers:
   - **Pairing:** the Display user agent is redirected from `/`, the glasses
     pair by code, and the code is approved with a signed-in session.
   - **Chat:** the composer commit, the late-pinch guard, the glasses rules
     block reaching the agent, wide tables shown as cards, file chips, and
     approvals.
   - **Files:** markdown only, member files labelled, page turns, and Back
     restoring focus.
   - **Data:** tables, then rows with paging, then records with stepping.
   - **Phone approve page.**
   - **Budgets:** first-load request count, no horizontal overflow, and no
     external requests.

## Run

```bash
cd flight-deck && npm run build        # the check serves the built static/
cd .. && flight-deck/e2e/hud/run.sh    # FD on :25190, mock agent on :24090
```

Requirements:

- `uv` (Python env)
- Node 22+
- Playwright with Chromium (`npm i -g playwright` and `npx playwright install chromium`), or set `PLAYWRIGHT_MODULE=/path/to/node_modules/playwright`

Screenshots and `report.json` land in `.run/shots/` and `.run/`; `.run/` is git-ignored.
Ports: `FD_PORT` and `AGENT_PORT`.
