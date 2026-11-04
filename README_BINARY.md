# Building Captain Claw Desktop & Binary Releases

This document explains how to build Captain Claw as standalone binaries and
as a packaged desktop application (macOS, Windows, Linux).

There are two distribution modes:

1. **Server binaries** — headless executables you run from the terminal
2. **Desktop app** — an Electron shell that wraps the web UI into a native window

---

## Architecture

```
┌──────────────────────────────────────────────┐
│               Desktop App (Electron)         │
│                                              │
│   ┌──────────────────────────────────────┐   │
│   │  BrowserWindow loads localhost:23080  │   │
│   └──────────────┬───────────────────────┘   │
│                  │ HTTP + WebSocket           │
│   ┌──────────────▼───────────────────────┐   │
│   │  captain-claw-web (PyInstaller bin)  │   │
│   │  aiohttp server + Agent + SQLite     │   │
│   └──────────────────────────────────────┘   │
└──────────────────────────────────────────────┘
```

The desktop app is two processes:

- **Electron shell** — a Chromium-based window that shows the web UI.
  It has no business logic; it just spawns the backend and points a
  `BrowserWindow` at `http://127.0.0.1:23080`.
- **Python backend** — the full Captain Claw server compiled into a native
  binary with PyInstaller. No Python installation is needed on the end user's
  machine.

On launch, Electron starts the backend as a child process, polls until the
HTTP server is ready, then loads the URL. On quit, it sends `SIGTERM`
(or `taskkill` on Windows) to shut the backend down gracefully.

---

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.11+ | Backend build (activate venv if your system Python is older) |
| Node.js | 18+ | Electron packaging |
| npm | 9+ | Dependency management for Electron |
| PyInstaller | 6+ | Compiles Python into native executables |

Install Python build dependencies:

```bash
pip install -e ".[build]"
```

---

## 1. Server Binaries Only (no Electron)

This produces three standalone executables that run from the command line.

```bash
./build.sh
```

Output in `dist/captain-claw/`:

| Binary | Description |
|--------|-------------|
| `captain-claw` | Interactive CLI + web UI |
| `captain-claw-web` | Web UI server only |
| `captain-claw-orchestrate` | Headless multi-agent orchestrator |

To also create a platform archive (`.tar.gz` or `.zip`):

```bash
./build.sh --archive
```

### How it works

1. `build.sh` invokes `pyinstaller captain_claw.spec`.
2. The spec file defines three `Analysis` entry points — one per executable —
   then uses `MERGE` so they share a single set of bundled dependencies.
3. PyInstaller traces all imports, bundles `.pyc` files, C extensions, and
   data files (static assets, instruction templates) into a self-contained
   `dist/captain-claw/` folder.
4. A runtime hook (`build_support/runtime_hook.py`) runs before application
   code and patches environment variables so the frozen binary can locate
   its bundled data files (instructions, static web assets).

### What gets bundled

- All Python dependencies (litellm, aiohttp, pydantic, tiktoken, etc.)
- The web UI static files (`captain_claw/web/static/`)
- Instruction templates (`captain_claw/instructions/`)
- Native C extensions (multidict, yarl, frozenlist, pydantic-core, etc.)

### What does NOT get bundled

- Your `config.yaml` — loaded at runtime from `~/.captain-claw/`
- SQLite databases (sessions, memory, datastore) — created at runtime
- API keys — read from environment or config
- Ollama — must be installed separately if you want local models

---

## 2. Desktop App (Electron)

```bash
./build-desktop.sh            # the slim app (default): 75 MB DMG on macOS
./build-desktop.sh --full     # the self-contained app, backend inside (about 1 GB)
```

### The slim app

The default build is the Electron shell alone. The backend is not inside: on its
first launch the app installs it, and it installs every later release itself,
without a new download of the app.

What the first launch installs, from the newest release's manifest
(`desktop-manifest.json`, see [Releasing for the desktop app](#releasing-for-the-desktop-app)):

| What | Where it comes from | Size (macOS arm64) |
|------|---------------------|--------------------|
| uv | GitHub (astral-sh/uv), version and sha256 pinned by the manifest | 14 MB |
| Python 3.11 | python-build-standalone, installed by that uv (never the system's) | 20 MB |
| The dependencies | PyPI, every package pinned with its hashes (`desktop-lock-<version>.txt`) | about 180 MB |
| Captain Claw | the release's wheel, sha256 pinned by the manifest; it carries the backend and the Flight Deck frontend | 15 MB |

It takes about half a minute on a fast line, with a progress page. Chromium for
the browser tools follows in the background. Everything goes to the app's runtime
folder, not to `~/.captain-claw` (your data):

| Platform | Runtime folder |
|----------|----------------|
| macOS | `~/Library/Application Support/Captain/runtime` |
| Windows | `%LOCALAPPDATA%\Captain\runtime` |
| Linux | `~/.local/share/captain-claw-desktop/runtime` |

Inside it: `bin/uv`, `python/`, uv's `cache/`, one venv per version under
`envs/<version>/`, and `state.json` (which version runs, the one before it, and
one waiting for a restart).

### Updates

The app checks the manifest a minute after it starts and every six hours, and
**Check for Updates…** (the Captain menu on macOS, Help elsewhere) checks now.
For a new version it asks first, then:

1. **Installs it beside the running one** in `envs/<new>/`. uv's cache supplies
   every package that didn't change, so a release whose lock is the same
   downloads only its wheel (15 MB) and installs in seconds.
2. **Test-starts it**: its agent code imports, its page has all its files, and its
   Flight Deck answers on a spare port with an empty data folder and none of your
   keys. A version that fails here is removed and nothing changes.
3. **Asks to restart.** *Restart Now* stops the agents, switches and starts the
   new version; *Later* leaves it for the next time the app opens.
4. **Goes back by itself** when the new version doesn't come up: the old venv is
   still there, so it switches back and says why. A version that failed isn't
   offered again until you check yourself.

The version before stays for that, and older ones are removed with what only they
used in uv's cache. Console → Updates in the app points to this menu.

A release that needs a newer shell raises `min_shell` in its manifest
(`MIN_SHELL` in `scripts/desktop_release.py`, `SHELL_PROTOCOL` in
`desktop/runtime.js`); the app then asks its user to download the new app.

### Releasing for the desktop app

```bash
python scripts/desktop_release.py
```

builds the wheel (with the Flight Deck bundle as committed), locks its
dependencies for every platform (`uv pip compile --universal --generate-hashes`),
reads uv's checksums, and writes `dist/desktop-release/<version>/`:
`captain_claw-<version>-py3-none-any.whl`, `desktop-lock-<version>.txt` and
`desktop-manifest.json`, whose summary is the first paragraph of
`release-notes/RELEASE_NOTES_<version>.md`. It prints the `gh release upload`
command for the three files, unless the checkout has moved past the tag
`v<version>`: a wheel built from later commits would still say `<version>`, so
bump the version and tag first. uv's checksums are read first, and a GitHub
hiccup (a 502, a timeout) is tried again a few times. The app reads
`https://github.com/kstevica/captain-claw/releases/latest/download/desktop-manifest.json`,
so they have to be on the release GitHub calls *latest*.

To try a release before publishing it, write it for a local server and point an
app at it; the variables keep the test app apart from an installed one:

```bash
python scripts/desktop_release.py --base-url http://127.0.0.1:8765/ --out /tmp/rel
python3 -m http.server 8765 --directory /tmp/rel &
CAPTAIN_CLAW_MANIFEST_URL=http://127.0.0.1:8765/desktop-manifest.json \
CAPTAIN_CLAW_RUNTIME_DIR=/tmp/cc-test/runtime CAPTAIN_CLAW_HOME=/tmp/cc-test/home \
CAPTAIN_CLAW_USER_DATA=/tmp/cc-test/userdata \
  "dist/desktop/mac-arm64/Captain.app/Contents/MacOS/Captain"
```

### The full app

`--full` is the build from before the slim app: the Flight Deck frontend is built,
`./build.sh` makes the PyInstaller binaries in `dist/captain-claw/`, and
electron-builder puts them in the app (`desktop/electron-builder.full.js`). Its
first start needs no internet; it doesn't update itself.

```bash
./build-desktop.sh --backend-only    # full: only the PyInstaller backend
./build-desktop.sh --electron-only   # full: package an already built backend
```

### What it looks like

The app is **Captain** (`Captain.app`, `Captain Setup.exe`); `package.json`'s
`name` stays `captain-claw`, so the window storage and logs of earlier builds
carry over. Every start shows `desktop/setup.html` first: the Captain Bot and the
wordmark while it starts or switches versions, and on the first start the install
steps beside a short product tour. The macOS install window has its own
background. `build_support/render_desktop_art.py` makes that artwork: it copies
the tour shots from the website (`~/Dev/captain-claw.com/website/public`, or
`--website DIR`) into `desktop/assets/tour/` and renders
`desktop/icons/dmg-background.png` + `@2x` with Playwright. The display face is
Sora (`desktop/assets/fonts/`, SIL Open Font License), as on the website.

### Build steps (slim)

1. `npm install` inside `desktop/`.
2. electron-builder packages the shell for this CPU (`--arch arm64|x64` for the
   other one: the slim shell is JavaScript only).
3. On macOS the DMG is converted to LZMA (`hdiutil convert -format ULMO`):
   electron-builder writes zlib images, 110 MB for Electron alone, against 75 MB
   as LZMA.

### Output

Output goes to `dist/desktop/`:

| Platform | Slim | Full |
|----------|------|------|
| macOS | `Captain-<version>-arm64.dmg` (75 MB) | `.dmg`, `.zip` |
| Windows | `Captain Setup <version>.exe` (NSIS) | `.exe`, `.zip` |
| Linux | `.AppImage`, `.deb`, `.tar.gz` | the same |

```bash
./build-desktop.sh --platform mac|win|linux
./build-desktop.sh --clean           # wipe build artifacts first
```

---

## Project Structure

```
├── build.sh                    # Builds server binaries (PyInstaller)
├── build-desktop.sh            # Builds the desktop app (slim, or --full with PyInstaller)
├── captain_claw.spec           # PyInstaller spec (3 entry points, shared deps)
├── scripts/desktop_release.py  # Wheel + lock + manifest the slim app installs from
├── build_support/render_desktop_art.py # Tour shots + install-window background
├── build_support/
│   └── runtime_hook.py         # Patches env vars for frozen binaries
├── desktop/
│   ├── package.json            # Electron app config + electron-builder settings
│   ├── main.js                 # Electron main process (backend lifecycle, updates)
│   ├── runtime.js              # The slim app's Python: install, update, test start, go back
│   ├── setup.html              # Start, update and first-start screens
│   ├── assets/                 # Tour shots + Sora for setup.html
│   ├── electron-builder.full.js # The full build's config (backend inside)
│   ├── preload.js              # Exposes desktop detection to renderer
│   └── icons/                  # App icons (.icns, .ico, .png)
└── dist/
    ├── captain-claw/           # PyInstaller output (server binaries)
    └── desktop/                # electron-builder output (installers)
```

---

## How the Electron Shell Works

`desktop/main.js` handles the full lifecycle:

1. **Spawn** — Starts the agent server and the Flight Deck: in the slim app with
   the installed version's Python (`python -m captain_claw.flight_deck.server`,
   installing it first on the first launch), in the full app from the
   PyInstaller binaries inside it. In dev mode (`--dev`), it runs from source
   via `python3 -m …`; `npm run dev -- --runtime` uses the installed Python instead.

2. **Wait** — Polls `http://127.0.0.1:23080/api/commands` every 300ms
   until the backend responds with HTTP 200 (timeout: 30 seconds).

3. **Load** — Points the `BrowserWindow` at the backend URL. While waiting,
   a loading screen with a spinner is shown.

4. **Crash recovery** — If the backend exits unexpectedly, a dialog offers
   to restart it or quit.

5. **Shutdown** — On window close, sends `SIGTERM` to the backend. If it
   doesn't exit within 5 seconds, sends `SIGKILL`.

### Dev mode

Run the Electron shell against a source checkout (no PyInstaller build needed):

```bash
cd desktop
npm install
npm run dev
```

This requires the backend built in `dist/captain-claw/`, or a working Python
environment with `captain_claw` installed.

---

## Data Storage

The desktop app stores all user data in `~/.captain-claw/` (same as the
server binaries). This includes:

| File | Contents |
|------|----------|
| `config.yaml` | User configuration |
| `sessions.db` | Chat sessions and messages |
| `memory.db` | Semantic memory and embeddings |
| `datastore.db` | User-managed relational data |
| `workspace/` | File workspace |

This directory is independent of the app installation, so upgrading or
reinstalling the app preserves all data.

---

## App Icons

Before building for distribution, place icons in `desktop/icons/`:

| File | Platform | Notes |
|------|----------|-------|
| `icon.icns` | macOS | Use `iconutil` or an online converter |
| `icon.ico` | Windows | 256x256 multi-resolution `.ico` |
| `icon.png` | Linux | 512x512 PNG |

---

## Cross-Platform Notes

**You can only build for the platform you are on.** PyInstaller produces
native binaries, so:

- Build on macOS for `.dmg`
- Build on Windows for `.exe`
- Build on Linux for `.AppImage` / `.deb`

`.github/workflows/build-binaries.yml` does that for the command-line binaries:
macOS, Windows, Linux x64 and Linux arm64, each checked with `captain-claw
--version`. On a `v*` tag it waits for the release you publish and attaches the
four archives to it; it never creates, renames or drafts a release. To build
without a release, run it by hand (the binaries stay as the run's artifacts for
30 days):

```bash
gh workflow run build-binaries.yml --ref main -f version=test
```

### macOS

- Builds produce both `x64` (Intel) and `arm64` (Apple Silicon) targets.
- For distribution outside the App Store, you need an Apple Developer
  certificate and must notarize the app with `xcrun notarytool`.
- `hardenedRuntime` is enabled in the electron-builder config.

### Windows

- NSIS installer with optional install directory selection.
- For distribution, sign with an EV code signing certificate to avoid
  SmartScreen warnings.

### Linux

- AppImage is the most portable (single file, runs on most distros).
- `.deb` targets Debian/Ubuntu.
- `.tar.gz` is a raw archive for manual installation.

---

## Troubleshooting

**"Captain Claw couldn't finish setting up" (slim app)**

The first launch downloads Captain Claw, so it needs the internet; *Try Again*
picks up where it stopped. The log (its path is in the dialog) has each step's
output.

**"Backend binary not found" on launch (full app)**

The PyInstaller build was not included in the Electron package. Make sure
`dist/captain-claw/` exists before running `build-desktop.sh --electron-only`.

**Backend takes too long to start**

The default timeout is 30 seconds. On first launch, tiktoken downloads its
encoding files, which can be slow. Subsequent launches are faster.

**Port 23080 is already in use**

The backend automatically tries the next port (up to 10 attempts). The
Electron shell detects the actual port from the backend's stdout.

**Large binary size**

The slim desktop app is 75 MB (macOS DMG) because it carries no Python; see
[The slim app](#the-slim-app). The PyInstaller bundle includes all Python
dependencies, and everything else importable in the environment it's built from
(a development venv with torch, OpenCV or polars installed puts them in too), so
build the full app from a clean venv. Typical sizes:
- Backend alone: ~150-250 MB from a clean venv
- Full desktop app (with Electron): ~350-500 MB from a clean venv

The `excludes` list in `captain_claw.spec` strips unused packages (tkinter,
matplotlib, scipy, test suites) to reduce size.
