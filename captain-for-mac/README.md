# Captain for Mac

What's on your screen, to one of your agents — from any app, not just the
browser. The Mac sibling of Captain for Chrome (`browser-extension/chrome/`);
it uses the same deck API (`/fd/clip/*`, `captain_claw/flight_deck/browser_clip.py`).

## Get it

Download it from GitHub — [Captain for Mac releases](https://github.com/kstevica/captain-claw/releases?q=mac-v&expanded=true)
(tags `mac-v*`, e.g. [`mac-v0.1.0`](https://github.com/kstevica/captain-claw/releases/tag/mac-v0.1.0)), or from your
deck: Connections → Browser → Download Captain for Mac (the deck's own build when it has one, else that release).
It needs a deck from main `8e4f99eb` on (the release after 0.8.7).

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
`out/Captain-for-Mac.zip` hands that out in Connections → Browser
(`/fd/clip/mac-app.zip`); any other deck sends people to the GitHub release
(`MAC_APP_RELEASE_URL` in `browser_clip.py`; `FD_MAC_APP_URL` overrides it, empty
turns the download off).

## Releasing a new version

A release of its own, never "latest" (the desktop app reads latest for its updates):
bump `VERSION` in `build.sh` (and `CFBundleShortVersionString` follows), build, then

```bash
gh release create mac-v<version> --target <commit> --title "Captain for Mac <version>" --prerelease --latest=false \
  Captain-for-Mac-<version>.zip Captain-for-Mac-<version>.zip.sha256 Captain-for-Mac.zip
```

and point `MAC_APP_RELEASE_URL` at the new tag. A `mac-v*` tag doesn't start
build-binaries.yml (that one runs on `v*`).

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
