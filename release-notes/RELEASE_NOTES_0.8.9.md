# Captain Claw v0.8.9 Release Notes

**Release title:** Porthole
**Release date:** 2026-11-04

A patch for the desktop app, the same day as 0.8.8. On Windows it creates and runs agents, and on
macOS a downloaded copy is no longer called damaged. **Captain for Windows** has its first download:
`Captain-Setup-0.8.9-x64.exe` on this release.

3 pull requests: #421 and #422 in the tag, and #424 in the Windows installer, merged after it.

## Fixed

- **Creating an agent on Windows.** **New agent → Create** answered "Internal Server Error". Flight
  Deck saves its list of agents with a file lock that exists only on macOS and Linux, and on Windows
  that save failed before the agent could start. It now locks where it can and skips the lock on
  Windows.
- **Agents keep running on Windows.** Flight Deck asked whether an agent was still running with a call
  that, on Windows, ends the process instead of asking. It now asks Windows directly. **Stop** ends the
  agent and everything it started, its browser included.
- **Each agent has its own settings on Windows.** Windows finds a program's home folder somewhere
  else than macOS and Linux do, so every agent would have read your own `~/.captain-claw` and not its
  own. Agents now get their own folder there as well, read their settings as UTF-8, and open no
  console window.
- **The desktop app on Windows reads and writes text as UTF-8.** Windows' default text encoding,
  cp1252, can't hold characters like "—".
- **Opening an agent's chat on Windows.** The window went blank. The Windows build left out Chromium's
  language file (`en-US.pak`), and without it the page crashed. The installer on this release has it.
  If the page ever crashes again, the app goes back to the start page and writes why to its log.
- **A downloaded Mac app isn't "damaged".** Without a Developer ID certificate, Captain.app was only
  partly signed. A copy downloaded in a browser got "Captain is damaged and can't be opened", with no
  way past it. The app is now signed ad hoc as a whole, so macOS only asks the first time: **System
  Settings → Privacy & Security → Open Anyway**.

On macOS and Linux, Flight Deck runs and stops agents as before.

## Upgrading

1. **The desktop app**: **Check for Updates…** (it also offers 0.8.9 within a minute of starting).
2. **On Windows**: install
   [`Captain-Setup-0.8.9-x64.exe`](https://github.com/kstevica/captain-claw/releases/download/v0.8.9/Captain-Setup-0.8.9-x64.exe).
   It isn't signed, so SmartScreen asks first: **More info → Run anyway**. A Windows app you built
   yourself before this needs it too, over the one you have (your agents and data stay): only the
   installer brings the chat fix and the UTF-8 setting.
3. **Each Flight Deck**: nothing changes on macOS and Linux. Pull and restart whenever suits you, or
   use **Console → Updates**.

Devices need nothing: firmware stays **0.20.4** / PaperColor B **1.0.31**. BotPort hubs need nothing:
0.8.9 keeps protocol 2. Everything new in 0.8.8 is in its
[release notes](https://github.com/kstevica/captain-claw/blob/main/release-notes/RELEASE_NOTES_0.8.8.md).

## Pull requests

#421 The desktop app is signed ad hoc as a whole, so a download isn't "damaged" · #422 Flight Deck
creates and runs agents on Windows · #424 Windows keeps its en-US locale; a crashed page is logged and
reloaded (in the Windows installer, merged after the tag)
