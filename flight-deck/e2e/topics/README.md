# Chat topic dividers

Checks the chat store's topic dividers (`src/stores/chatStore.ts`): the
"New topic" / "Back on …" lines of a lane's one continuous timeline. They come
from /new, /session switch and /clear, from the agent reporting another
session (`session_info`), and from `reconcileTopic` once the lane's Flight Deck
log has loaded.

`run.sh` runs each scenario in its own Node process against the real store,
with a fake agent socket and a fake `/fd/chat` log whose first read can be held
back, so the agent's replay can land before or after the log. It covers:

- **Signed in:** the first visit (replay before or after the log), a revisit
  (no new divider, nothing written), a reply missed while no tab was open, a
  log from before dividers, the lane moving to another session while the log
  loads, a reconnect while paged back, and switches elsewhere or here.
- **No sign-in (the desktop app):** a reload (the transcript is the replay, so
  a turn's end marks nothing), switches, and a reconnect onto another session.
- **A message sent as the socket dies** (`held…`, same store and fake socket):
  a frame the agent never took goes back on hold and is sent again once the
  chat reconnects; one the agent took (the replay shows it) is not sent twice;
  a send on a socket that wasn't open is held, not lost.

```bash
flight-deck/e2e/topics/run.sh
```

Needs Node 23.6 or later (it runs the store's TypeScript as is). Point it at
another source tree with `SRC=/path/to/flight-deck/src`.
