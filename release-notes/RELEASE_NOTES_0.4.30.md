# Captain Claw v0.4.30 Release Notes

**Release title:** Intention Tags

**Release date:** 2026-06-02

## Highlights

Captain Claw 0.4.30 adds **tags to intentions** — the assistant labels each intention with up to 5 short tags, and you can search/filter intentions by tag from both the agent and the Flight Deck UI.

Small, additive release on top of 0.4.29.

## What changed

- **Tags on intentions** — a new `tags` column on the intentions store (with an automatic migration for existing DBs). Tags are normalized (lowercased, deduped, capped at 5) and exposed as a list.
- **Agent: create / update / search by tag** — the `intentions` tool accepts `tags` on `create`/`update`, and a new `search` action finds intentions by tag with `match='any'` (default) or `match='all'`. Tag matching is exact (`vc` won't match `vcfund`). The tool prompts the model to add up to 5 short tags per intention.
- **Flight Deck panel** — the Intentions panel renders tag chips on each intention and a clickable tag-filter row (`all` + one chip per tag); click a chip to narrow the list.

## How to use

```
intentions(action="create", origin="user", title="Reconnect with founder X",
           why="warm intro pending", tags=["networking","vc","founder"])

intentions(action="search", tags=["vc"])                 # any
intentions(action="search", tags=["vc","reporting"], match="all")
```

In Flight Deck → an agent card → **Intentions**: click a tag chip (on a row or in the filter row) to filter; `all` resets.

## Backward compatibility

Fully compatible with 0.4.29. Existing intentions get an empty tag list until edited; new ones are tagged by the model. The DB migration is automatic.

## Upgrade

```bash
git pull
# rebuild the Flight Deck UI only if you build assets locally (they're committed):
npm --prefix flight-deck run build
# restart Flight Deck and the agents
```
