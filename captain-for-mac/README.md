# Captain for Mac

What's on your screen, to one of your agents — from any app, not just the
browser. The Mac sibling of Captain for Chrome (`browser-extension/chrome/`);
it uses the same deck API (`/fd/clip/*`, `captain_claw/flight_deck/browser_clip.py`).

## Use

- **Ask**: hold **fn** alone for 3 s (key and time in Settings → Trigger), click
  the menu-bar icon, or the optional floating button. A ring by the pointer
  fills while the key counts; any other key cancels it, so fn shortcuts keep
  working.
- Captain reads the front app: its window's text through Accessibility (on
  screen first, the rest of the window after), **your selection — what you
  mean**, what's under the pointer (and the line around it), a screenshot when
  Settings → What it reads asks for one, and the screen image's text (OCR) when
  an app doesn't share its text. Each shows as a chip; switch any off before
  sending, or look at what was read (the eye).
- Type the task, or say it (mic button, or "Listen when Captain opens"; with
  "Send when I stop talking" a pause — or letting go of the held key — sends
  it). Empty = *Handle it*. One-click actions: Explain, Summarize, Draft a
  reply, Improve, Translate, What's wrong?, Make tasks, Remember, Save it.
- The agent's steps and answer show in the panel; **Copy** or **Paste into
  <app>** puts it back where you were; ask follow-ups about the same capture.
- Captures go to the **screen-clips** shared folder
  (`vfs:screen-clips/<day>-<app>-<title>/`: screen.md, selection.md,
  screenshot.png, clip.json) and into the agent's own chat, as your words.

## Build

```bash
./build.sh            # → out/Captain for Mac.app + out/Captain-for-Mac.zip (universal, ad-hoc signed)
SIGN_ID="Developer ID Application: …" ./build.sh
```

Needs Xcode's Swift (macOS 14+). A deck running from a checkout that has
`out/Captain-for-Mac.zip` hands it out in Connections → Browser
(`/fd/clip/mac-app.zip`).

## Permissions

Accessibility (read the screen, the selection, the held key) — required.
Screen Recording (screenshot, OCR), Microphone and Speech Recognition — only
for those features. Settings → Permissions shows each. macOS ties them to the
app's signature, so an ad-hoc build asks again after every rebuild.

## Checking it

- `CaptainForMac --probe 2 [--app <bundle id>] [--shot window|screen --ocr accurate|fast] [--doc]`
  reads the screen once and prints what it got (timings, tree stats).
- `CaptainForMac --selftest-hold` — the hold-key logic on synthetic events.
- `CAPTAIN_FOR_MAC_NO_BROWSER=1 CaptainForMac -deckURL <deck> --selftest "<task>" --followup "<q>" --pair --out <dir>`
  — pair, capture, send, follow the run, follow up; panel snapshots in `<dir>`.
  `-deckURL` is read for that run only; with `CAPTAIN_FOR_MAC_NO_BROWSER` the
  token isn't written to the Keychain. `CAPTAIN_FOR_MAC_TOKEN` sets one directly.
