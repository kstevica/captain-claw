# Presenting HTML decks from Flight Deck

Present an agent-built `.html` deck on the big screen and drive it from your
phone / glasses / WhatsApp — no PDF, no PPT. All control rides the existing
glasses channel bus, so nothing new has to be paired.

## The deck file

Each slide is a `<section class="slide">…</section>`. That's the only rule —
see `deck_sample.html` for a ready-to-use template. Have the Claw agent write
the deck into its files (so it shows up in the file list).

The server injects the slide engine + remote receiver at serve time, so the
deck file itself stays trivial.

## Fully glasses-driven (recommended)

1. On the big screen open the waiting surface once — **no path**:

       /deck/view?c=deck

   It shows "Ready to present" and follows whatever you open next.
2. On the glasses, **Files → tap a `.html` deck**. It opens on the glasses
   *and* auto-mirrors onto the big screen (the waiting page navigates to it).
3. Drive with the ◀ / ▶ in the glasses header. Tap **⧉** to (re)send the
   current deck to the big screen if it opened late.

Switching to a different deck on the glasses re-points the big screen live.

## Big screen (direct)

Or open a specific deck straight away on the Python server that runs the
bridges:

    /deck/view?c=<channel>&path=<physical-path-of-the-html-file>

- `c` — any channel name you choose, e.g. `deck`.
- `path` — the file's physical path (same value the glasses file list uses;
  also what `/glasses/files` returns as `physical`).
- optional `host`/`port`/`auth` — bind the channel to a specific agent (the
  Flight Deck "Deck view" button passes these automatically).

### From the Flight Deck file list

In the React file browser, HTML files show a **Deck view** button (monitor
icon) next to the **View** (eye) button. It opens `/deck/view` on the `deck`
channel in a new tab, bound to that agent — present on the big screen, drive
from glasses / `/deck/remote` / WhatsApp.

Press **F** for fullscreen. Local keys always work: → / Space / PgDn = next,
← / PgUp = prev, Home / End = first / last. Click right 25% = next, left 25%
= prev. A dot top-left shows the bus link (green = a remote can reach it).

## Phone / glasses remote (primary)

    /deck/remote?c=<channel>      ← same channel as the deck

Big Prev / Next buttons, live `N / M` position, First / Last. This is the
reliable presenter remote — low latency, tactile.

### …or from the existing glasses file list

Open the glasses view (`/glasses/view?c=<channel>`), tab to **Files**, open the
deck's `.html`. A compact ◀ / ▶ remote appears in the header and drives the
deck on the **same channel**. (Shown only for `.html` files; both tap and the
NeuralBand down-gesture + activate work.) Same channel = the deck's channel.

## WhatsApp (demo flourish)

From an allow-listed WAID:

- `next slide` / `previous slide` / `first slide` / `last slide`
- or `/next`, `/prev`, `/slide first`, `/slide last`
- `/slide on <channel>` to point at the deck's channel
  (defaults to env `DECK_DEFAULT_CHANNEL`, else your chat channel)
- `/slide` shows current binding + usage

Routed as a slash command **before** the agent, so it's instant and never
triggers an LLM turn. Great for one on-stage "watch, I'll text my agent to
advance" moment — not as your main clicker (Meta round-trip adds latency).

## If the network dies mid-talk

The deck advances on arrow keys with zero dependencies. Press → on the
laptop and keep going.

## Plumbing (for reference)

- `POST /deck/control {channel, action}` — action ∈ next|prev|first|last|goto
- `POST /deck/state   {channel, index, total}` — deck → bus position echo
- Control events are `type:"control"` / `type:"deck_state"` on the bus;
  neither is in the WhatsApp/Messenger forward allow-list, so they never
  echo into a chat thread.
