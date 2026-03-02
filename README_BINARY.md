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

## 2. Desktop App (Electron + Backend)

This packages everything into a native desktop application.

```bash
./build-desktop.sh
```

### Build steps

The script performs four steps:

**Step 1 — Build Python backend.**
Runs `./build.sh` to produce the PyInstaller binaries in `dist/captain-claw/`.

**Step 2 — Install Electron dependencies.**
Runs `npm install` inside `desktop/` to fetch Electron and electron-builder.

**Step 3 — Verify backend binaries.**
Checks that `dist/captain-claw/captain-claw-web` exists before packaging.

**Step 4 — Package with electron-builder.**
Bundles the Electron shell and the backend binaries into a platform installer.
The backend folder is embedded as an "extra resource" inside the app bundle.

### Output

Output goes to `dist/desktop/`:

| Platform | Artifacts |
|----------|-----------|
| macOS | `Captain Claw.dmg`, `.zip` |
| Windows | `Captain Claw Setup.exe` (NSIS), `.zip` |
| Linux | `.AppImage`, `.deb`, `.tar.gz` |

### Platform-specific builds

```bash
./build-desktop.sh --platform mac
./build-desktop.sh --platform win
./build-desktop.sh --platform linux
```

### Partial builds

```bash
# Backend only (skip Electron packaging):
./build-desktop.sh --backend-only

# Electron only (assumes backend already built):
./build-desktop.sh --electron-only

# Clean everything first:
./build-desktop.sh --clean
```

---

## Project Structure

```
├── build.sh                    # Builds server binaries (PyInstaller)
├── build-desktop.sh            # Builds desktop app (PyInstaller + Electron)
├── captain_claw.spec           # PyInstaller spec (3 entry points, shared deps)
├── build_support/
│   └── runtime_hook.py         # Patches env vars for frozen binaries
├── desktop/
│   ├── package.json            # Electron app config + electron-builder settings
│   ├── main.js                 # Electron main process (backend lifecycle)
│   ├── preload.js              # Exposes desktop detection to renderer
│   └── icons/                  # App icons (.icns, .ico, .png)
└── dist/
    ├── captain-claw/           # PyInstaller output (server binaries)
    └── desktop/                # electron-builder output (installers)
```

---

## How the Electron Shell Works

`desktop/main.js` handles the full lifecycle:

1. **Spawn** — Locates the `captain-claw-web` binary inside the app
   resources and spawns it as a child process. In dev mode (`--dev`), it
   falls back to running from source via `python3 -m captain_claw.web_server`.

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

For CI/CD, use a matrix of GitHub Actions runners (or similar) to build all
three platforms in parallel.

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

**"Backend binary not found" on launch**

The PyInstaller build was not included in the Electron package. Make sure
`dist/captain-claw/` exists before running `build-desktop.sh --electron-only`.

**Backend takes too long to start**

The default timeout is 30 seconds. On first launch, tiktoken downloads its
encoding files, which can be slow. Subsequent launches are faster.

**Port 23080 is already in use**

The backend automatically tries the next port (up to 10 attempts). The
Electron shell detects the actual port from the backend's stdout.

**Large binary size**

The PyInstaller bundle includes all Python dependencies. Typical sizes:
- Backend alone: ~150-250 MB
- Full desktop app (with Electron): ~350-500 MB

The `excludes` list in `captain_claw.spec` strips unused packages (tkinter,
matplotlib, scipy, test suites) to reduce size.
