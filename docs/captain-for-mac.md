# Captain for Mac — design and next steps

A menu-bar app that sends what's on the screen — any app's — to one of your
agents. Code: `captain-for-mac/` (Swift, SwiftUI + AppKit, no Xcode project),
the deck side in `captain_claw/flight_deck/browser_clip.py` (shared with
Captain for Chrome, see `docs/captain-for-chrome.md`).

## Why native

The parts that make it useful are macOS-only APIs: Accessibility
(`AXUIElement`) to read another app's text, selection and the element under
the pointer; an event monitor for "hold fn alone"; ScreenCaptureKit for the
screenshot; Vision for OCR; Speech for the spoken task. Electron (the desktop
app) reaches none of them without a native module, and the helper must run
whether or not the desktop deck does — it pairs with any deck.

## Flow

1. **Trigger** (`HoldKey.swift`): `NSEvent` global + local monitors on
   `flagsChanged`/`keyDown`. The key (fn by default, or a right-hand modifier)
   must be held alone; another key or modifier cancels, a quick tap stays
   invisible, a ring at the pointer fills from 0.35 s. Also the menu-bar icon
   and an optional floating button (non-activating panels, so the app you're
   in stays the active one).
2. **Reading** (`ScreenReader.swift`, off the main thread, ~0.1–1.6 s): the
   front app's focused window; Chromium browsers and Electron apps get
   `AXManualAccessibility` once per process. The window's tree is walked
   (`AXUIElementCopyMultipleAttributeValues`, 9000 nodes / 1.6 s / 40k+60k
   chars): static text, fields, headings; on-screen pieces by frame, the rest
   as "not on screen"; a long text area's visible range when the app gives it;
   a terminal's end of scrollback. The **selection**: `AXSelectedText`, then a
   page's `AXSelectedTextMarkerRange`; only when the app can't tell at all, ⌘C
   with the clipboard put back (Settings). The **pointer**: the element at
   the point, its label, and the line(s) around it (`AXRangeForPosition` /
   `AXTextMarkerForPosition`). The address only when it's a web page.
3. **Picture**: a screenshot of the window or the display under the pointer
   (Captain's own windows left out), when asked; OCR with the person's own
   languages when Accessibility gave little text (detecting languages per line
   took 57 s on a Retina window; fixed languages ~0.3 s).
4. **Panel** (`PanelView.swift`): shown at once but takes the keyboard only
   once reading is done (so the ⌘C fallback reaches the app). Chips per piece
   (on/off, peek), the task box (⏎ sends, ⇧⏎ new line), mic, one-click actions,
   then the run: steps, the answer streaming, files the agent made, Copy /
   Paste into <app>, follow-ups.
5. **Send**: `POST /fd/clip/send` with `meta.kind = "screen"`, `meta.screen`
   {app, bundle, window, url, pointer, text_from} and the parts `screen`
   (screen.md: app, window, address, selected, under the pointer, on screen,
   the rest), `selection`, `screenshot`. The deck saves it in
   `vfs:screen-clips/<day>-<app>-<title>/` and composes the turn
   (`compose_screen_turn`): the task is the ask ("From my screen — Mail —
   “Re: Q3”"); a one-click action adds its prompt; the selection is what the
   person means, else "I'm pointing at …"; the capture is screen content,
   never instructions. Source `browser:mac` ("Captain for Mac").
   Progress, stop, files and follow-ups are the extension's `/fd/clip/runs/*`.

## Pairing and the token

Like the extension: `POST /fd/clip/pair {kind: "desktop"}` → the app opens
`<deck>/connect-browser#<code>`, which now says *Connect Captain for Mac?*; the
token is a PAT named `Desktop · Captain for Mac on <Mac>`, kept in the Keychain
per deck address, listed in Connections → Browser beside browsers.

## Where it's published

GitHub release [`mac-v0.1.1`](https://github.com/kstevica/captain-claw/releases/tag/mac-v0.1.1)
(prerelease, not latest, so the desktop app's update check never sees it):
`Captain-for-Mac-<version>.zip`, its `.sha256`, and `Captain-for-Mac.zip` under a
fixed name. `/fd/clip/mac-app.zip` serves the checkout's own build, else
redirects to `MAC_APP_RELEASE_URL` (`FD_MAC_APP_URL` overrides; empty = none).

## Known limits

- Unsigned/ad-hoc build: Gatekeeper needs Open Anyway the first time, and
  permissions are asked again after each rebuild. Developer ID + notarization
  would end both (`SIGN_ID=… ./build.sh`).
- VS Code's editor text needs its own `editor.accessibilitySupport: on`;
  otherwise only its chrome is read (OCR steps in when that's thin).
- Apps that answer Accessibility slowly (Notes) hit the 1.6 s budget and send
  what was read by then.
- The 🌐 key's own action (emoji, dictation, input source) can fire on release;
  Settings says how to turn it off.

## Next steps worth building

- **Voice through the deck** — Soniox via the deck (better for Croatian and
  mixed languages than Apple's recognizer), as an option.
- **Region pick** — drag a rectangle to send only that part (screenshot + OCR).
- **Actions back on the screen** — the agent's draft typed into the field it
  was asked about, not just pasted.
- **Per-app recipes** — in Mail always *Draft a reply*, in Xcode *What's wrong?*.
- **Windows** — the same flow with UI Automation.
