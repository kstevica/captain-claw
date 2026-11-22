#!/usr/bin/env bash
#
# Build Captain Claw Desktop App.
#
# The default is the slim app (under 100 MB): the Electron shell alone. Its first
# launch installs the backend from the newest release's manifest
# (scripts/desktop_release.py), and later releases update it in place
# (desktop/runtime.js). --full builds the old self-contained app instead, with the
# PyInstaller backend inside (about 1 GB, no internet needed on first start).
#
# Usage:
#   ./build-desktop.sh                  # slim app for this platform and CPU
#   ./build-desktop.sh --arch x64       # slim app for another CPU (arm64|x64)
#   ./build-desktop.sh --platform win --arch x64
#                                       # another platform (mac|win|linux): --arch required
#   ./build-desktop.sh --full           # self-contained app (frontend + PyInstaller backend)
#   ./build-desktop.sh --backend-only   # full: only build the Python backend
#   ./build-desktop.sh --electron-only  # full: only package Electron (backend already built)
#   ./build-desktop.sh --clean          # wipe build artifacts first
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Parse flags ──────────────────────────────────────────────────
CLEAN=false
FULL=false
BACKEND_ONLY=false
ELECTRON_ONLY=false
PLATFORM=""
ARCH=""

for arg in "$@"; do
    case "$arg" in
        --clean)          CLEAN=true ;;
        --full)           FULL=true ;;
        --backend-only)   FULL=true; BACKEND_ONLY=true ;;
        --electron-only)  FULL=true; ELECTRON_ONLY=true ;;
        --platform|--arch) :;; # value follows
        --platform=*)     PLATFORM="${arg#--platform=}" ;;
        --arch=*)         ARCH="${arg#--arch=}" ;;
        mac|win|linux)    PLATFORM="$arg" ;;
        arm64|x64)        ARCH="$arg" ;;
        -h|--help)
            echo "Usage: $0 [--full] [--clean] [--backend-only] [--electron-only] [--platform mac|win|linux] [--arch arm64|x64]"
            exit 0
            ;;
        # A word this script doesn't know used to be skipped, so a mistyped
        # CPU (amd64, x86_64) quietly built for this machine's instead.
        *)
            echo "Error: unknown option '$arg' (see --help)"
            exit 1
            ;;
    esac
done
case "$PLATFORM" in ""|mac|win|linux) ;; *) echo "Error: unknown platform '$PLATFORM'"; exit 1 ;; esac
case "$ARCH" in ""|arm64|x64) ;; *) echo "Error: unknown CPU '$ARCH' (arm64 or x64)"; exit 1 ;; esac

case "$(uname -s)" in
    Darwin*)  HOST_PLATFORM="mac" ;;
    Linux*)   HOST_PLATFORM="linux" ;;
    MINGW*|MSYS*|CYGWIN*)  HOST_PLATFORM="win" ;;
    *)        HOST_PLATFORM="" ;;
esac

# Auto-detect platform if not specified
if [[ -z "$PLATFORM" ]]; then
    if [[ -z "$HOST_PLATFORM" ]]; then echo "Unknown platform: $(uname -s)"; exit 1; fi
    PLATFORM="$HOST_PLATFORM"
fi

# Built for this machine, its CPU is the default. Built for another platform
# there is no safe default: an Apple-silicon Mac makes an ARM64 Windows
# installer, which installs nothing on the x64 PCs most people have (0.9.1
# shipped one). (--full can't cross-build at all: PyInstaller builds for here.)
if [[ -z "$ARCH" && "$PLATFORM" != "$HOST_PLATFORM" ]]; then
    echo "Error: building for $PLATFORM on ${HOST_PLATFORM:-$(uname -s)}: give --arch x64 or --arch arm64."
    exit 1
fi

echo "========================================="
echo "  Captain Claw Desktop Build"
echo "  Platform: $PLATFORM"
if $FULL; then echo "  Kind: full (backend inside)"; else echo "  Kind: slim (backend installed on first launch)"; fi
echo "========================================="
echo ""

# ── Clean ────────────────────────────────────────────────────────
if $CLEAN; then
    echo "Cleaning build artifacts..."
    rm -rf build/ dist/captain-claw/ dist/desktop/
    rm -rf desktop/node_modules/
    echo ""
fi

if $FULL && ! $ELECTRON_ONLY; then
    # ── Step 1: Build Flight Deck frontend ───────────────────────
    # Before the backend: PyInstaller copies captain_claw/flight_deck/static
    # into the binaries, so a frontend built after it never reaches the app.
    echo "Step 1: Building Flight Deck frontend..."
    echo "─────────────────────────────────"

    cd flight-deck

    if ! command -v npm &>/dev/null; then
        echo "Error: npm is required but not found."
        echo "Install Node.js from https://nodejs.org"
        exit 1
    fi

    npm install
    npm run build

    cd "$SCRIPT_DIR"

    echo ""
    echo "Flight Deck frontend build complete."
    echo ""

    # ── Step 1b: Build Python backend with PyInstaller ───────────
    echo "Step 1b: Building Python backend..."
    echo "─────────────────────────────────"

    ./build.sh

    echo ""
    echo "Backend build complete."
    echo ""

    if $BACKEND_ONLY; then
        echo "Backend-only build done. Binaries in dist/captain-claw/"
        exit 0
    fi
fi

# ── Step 2: Install Electron dependencies ────────────────────────
echo "Step 2: Installing Electron dependencies..."
echo "─────────────────────────────────"

cd desktop

if ! command -v npm &>/dev/null; then
    echo "Error: npm is required but not found."
    echo "Install Node.js from https://nodejs.org"
    exit 1
fi

npm install

echo ""

# ── Step 3: Pick the CPU ─────────────────────────────────────────
BACKEND_DIR="$SCRIPT_DIR/dist/captain-claw"

if $FULL; then
    echo "Step 3: Verifying backend binaries..."
    echo "─────────────────────────────────"

    if [[ ! -d "$BACKEND_DIR" ]]; then
        echo "Error: Backend build not found at $BACKEND_DIR"
        echo "Run without --electron-only first, or run ./build.sh separately."
        exit 1
    fi

    EXE_NAME="captain-claw-web"
    FD_EXE_NAME="captain-claw-fd"
    if [[ "$PLATFORM" == "win" ]]; then
        EXE_NAME="captain-claw-web.exe"
        FD_EXE_NAME="captain-claw-fd.exe"
    fi

    if [[ ! -f "$BACKEND_DIR/$EXE_NAME" ]]; then
        echo "Error: $EXE_NAME not found in $BACKEND_DIR"
        exit 1
    fi

    if [[ ! -f "$BACKEND_DIR/$FD_EXE_NAME" ]]; then
        echo "Error: $FD_EXE_NAME not found in $BACKEND_DIR"
        exit 1
    fi

    # The Electron shell must match the backend's CPU: PyInstaller builds for
    # this machine only, so an x64 shell around arm64 binaries runs under
    # Rosetta on Apple silicon and can't start its backend on an Intel Mac.
    if [[ "$PLATFORM" == "mac" ]] && command -v lipo &>/dev/null; then
        BACKEND_ARCH="$(lipo -archs "$BACKEND_DIR/$FD_EXE_NAME")"
    else
        BACKEND_ARCH="$(uname -m)"
    fi
    echo "  Backend binary: $BACKEND_DIR/$EXE_NAME"
    echo "  Flight Deck binary: $BACKEND_DIR/$FD_EXE_NAME"
else
    # The slim shell is JavaScript only: any CPU, this machine's by default
    # (another platform needs --arch, checked above). runtime.js installs the
    # backend built for the CPU it runs on.
    BACKEND_ARCH="${ARCH:-$(uname -m)}"
fi

case "$BACKEND_ARCH" in
    arm64|aarch64) ARCH_FLAG="--arm64" ;;
    x86_64|amd64|x64)  ARCH_FLAG="--x64" ;;
    *)
        echo "Error: unsupported architecture: $BACKEND_ARCH"
        exit 1
        ;;
esac

echo "  Architecture: $BACKEND_ARCH"
echo ""

# ── Step 4: Package with electron-builder ────────────────────────
echo "Step 4: Packaging Electron app..."
echo "─────────────────────────────────"

# electron-builder never empties its output folder: installers from an
# earlier build (another arch, an older backend) would sit next to the new ones.
# An open Finder window can re-create .DS_Store mid-delete; the folder staying
# behind with only that in it is harmless.
rm -rf "$SCRIPT_DIR/dist/desktop" 2>/dev/null || true

CONFIG_ARGS=()
if $FULL; then CONFIG_ARGS=(--config electron-builder.full.js); fi

case "$PLATFORM" in
    mac)    npx electron-builder --mac "$ARCH_FLAG" ${CONFIG_ARGS[@]+"${CONFIG_ARGS[@]}"} ;;
    win)    npx electron-builder --win "$ARCH_FLAG" ${CONFIG_ARGS[@]+"${CONFIG_ARGS[@]}"} ;;
    linux)  npx electron-builder --linux "$ARCH_FLAG" ${CONFIG_ARGS[@]+"${CONFIG_ARGS[@]}"} ;;
esac

cd "$SCRIPT_DIR"

# ── Step 4b: Check the CPU the app was built for ─────────────────
# Read it from the packaged executable, not from the flags: 0.9.1's "x64"
# Windows installer carried an ARM64 app, and on x64 PCs it unpacked nothing
# and still exited 0. Every build host is little-endian, which od relies on.
first_match() { for f in "$@"; do if [[ -e "$f" ]]; then echo "$f"; return 0; fi; done; }

WANT_ARCH="${ARCH_FLAG#--}"
FOUND_ARCH=""
case "$PLATFORM" in
    mac)    # Mach-O: the CPU type follows the 4-byte magic.
        APP_EXE="$(first_match dist/desktop/mac*/Captain.app/Contents/MacOS/Captain)"
        [[ -n "$APP_EXE" ]] && FOUND_ARCH="$(od -An -tx4 -j4 -N4 "$APP_EXE" | tr -d ' ')"
        ;;
    win)    # PE: 0x3c holds the offset of "PE\0\0"; the machine type follows it.
        APP_EXE="$(first_match dist/desktop/win*-unpacked/Captain.exe)"
        if [[ -n "$APP_EXE" ]]; then
            PE_AT="$(od -An -tu4 -j60 -N4 "$APP_EXE" | tr -d ' ')"
            FOUND_ARCH="$(od -An -tx2 -j$((PE_AT + 4)) -N2 "$APP_EXE" | tr -d ' ')"
        fi
        ;;
    linux)  # ELF: e_machine at 18.
        APP_EXE="$(first_match dist/desktop/linux*-unpacked/captain-claw)"
        [[ -n "$APP_EXE" ]] && FOUND_ARCH="$(od -An -tx2 -j18 -N2 "$APP_EXE" | tr -d ' ')"
        ;;
esac
case "$FOUND_ARCH" in
    01000007|8664|003e) FOUND_ARCH="x64" ;;
    0100000c|aa64|00b7) FOUND_ARCH="arm64" ;;
esac

if [[ -z "$APP_EXE" ]]; then
    echo "Error: no packaged app in dist/desktop/ to read its CPU from"
    exit 1
fi
if [[ "$FOUND_ARCH" != "$WANT_ARCH" ]]; then
    echo "Error: $APP_EXE is built for ${FOUND_ARCH:-an unknown CPU}, not $WANT_ARCH"
    exit 1
fi
echo "  Checked: $APP_EXE is $FOUND_ARCH"

# ── Step 5: Recompress the disk image (macOS) ────────────────────
# electron-builder writes zlib (UDZO) images: Electron alone is 110 MB that way,
# 75 MB as LZMA (ULMO, macOS 10.15+, which Electron needs anyway). Its blockmap
# describes the old image, and nothing reads it (no electron-updater), so it goes.
if [[ "$PLATFORM" == "mac" ]] && command -v hdiutil &>/dev/null; then
    for dmg in dist/desktop/*.dmg; do
        [[ -f "$dmg" ]] || continue
        echo "Recompressing $(basename "$dmg") as LZMA..."
        hdiutil convert "$dmg" -format ULMO -ov -quiet -o "${dmg%.dmg}.ulmo.dmg"
        mv -f "${dmg%.dmg}.ulmo.dmg" "$dmg"
        rm -f "$dmg.blockmap"
    done
fi

echo ""
echo "========================================="
echo "  Build Complete!"
echo "========================================="
echo ""
echo "  Output: dist/desktop/"
echo ""

# List output files
if [[ -d "dist/desktop" ]]; then
    echo "  Artifacts:"
    find dist/desktop -maxdepth 1 \( -name "*.dmg" -o -name "*.zip" -o -name "*.exe" -o -name "*.AppImage" -o -name "*.deb" -o -name "*.tar.gz" \) -exec du -h {} \; | sed 's/^/    /'
fi
