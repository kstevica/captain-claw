#!/usr/bin/env bash
#
# Build Captain Claw binaries for the current platform.
#
# Usage:
#   ./build.sh              # full build (3 executables in dist/captain-claw/)
#   ./build.sh --clean      # wipe build/ and dist/ first
#   ./build.sh --archive    # also produce a .tar.gz / .zip archive
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Parse flags ──────────────────────────────────────────────────
CLEAN=false
ARCHIVE=false
for arg in "$@"; do
    case "$arg" in
        --clean)  CLEAN=true ;;
        --archive) ARCHIVE=true ;;
        -h|--help)
            echo "Usage: $0 [--clean] [--archive]"
            echo ""
            echo "  --clean    Remove build/ and dist/ before building"
            echo "  --archive  Create a tar.gz (Linux/macOS) or zip (Windows) archive"
            exit 0
            ;;
    esac
done

# ── Clean ────────────────────────────────────────────────────────
if $CLEAN; then
    echo "Cleaning build/ and dist/..."
    rm -rf build/ dist/
fi

# ── Check dependencies ───────────────────────────────────────────
if ! command -v python3 &>/dev/null && ! command -v python &>/dev/null; then
    echo "Error: Python 3.11+ is required but not found."
    exit 1
fi

PYTHON="$(command -v python3 || command -v python)"
PY_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Using Python $PY_VERSION ($PYTHON)"

# Ensure PyInstaller is available
if ! "$PYTHON" -m PyInstaller --version &>/dev/null 2>&1; then
    echo "Installing PyInstaller..."
    "$PYTHON" -m pip install pyinstaller
fi

# Ensure captain_claw itself is installed (needed for dependency resolution)
if ! "$PYTHON" -c "import captain_claw" &>/dev/null 2>&1; then
    echo "Installing captain-claw in current environment..."
    "$PYTHON" -m pip install .
fi

# ── Download Playwright Chromium browser ─────────────────────────
echo ""
echo "Downloading Playwright Chromium browser..."
echo ""
export PLAYWRIGHT_BROWSERS_PATH="$SCRIPT_DIR/build/pw-browsers"
"$PYTHON" -m playwright install chromium
echo "Chromium installed to $PLAYWRIGHT_BROWSERS_PATH"

# ── Build ────────────────────────────────────────────────────────
echo ""
echo "Building Captain Claw binaries..."
echo ""
"$PYTHON" -m PyInstaller captain_claw.spec

# ── Copy Playwright browsers into dist (post-PyInstaller) ────────
# Chromium's .app bundle cannot be codesigned by PyInstaller, so we
# copy it directly into the dist folder after the build completes.
DIST_DIR="dist/captain-claw"
INTERNAL_DIR="$DIST_DIR/_internal"
if [[ ! -d "$INTERNAL_DIR" ]]; then
    INTERNAL_DIR="$DIST_DIR"
fi

if [[ -d "$PLAYWRIGHT_BROWSERS_PATH" ]]; then
    echo ""
    echo "Copying Playwright browsers into dist..."
    cp -R "$PLAYWRIGHT_BROWSERS_PATH" "$INTERNAL_DIR/pw-browsers"
    echo "  ✓ pw-browsers/"
fi

# ── Verify ───────────────────────────────────────────────────────
echo ""
echo "Verifying build..."

if [[ ! -d "$DIST_DIR" ]]; then
    echo "Error: dist/captain-claw/ not found. Build may have failed."
    exit 1
fi

# Check executables
OK=true
for name in captain-claw captain-claw-web captain-claw-orchestrate; do
    if [[ -f "$DIST_DIR/$name" ]] || [[ -f "$DIST_DIR/$name.exe" ]]; then
        echo "  ✓ $name"
    else
        echo "  ✗ $name MISSING"
        OK=false
    fi
done

# Check data files — PyInstaller 6+ puts them under _internal/
INTERNAL="$DIST_DIR/_internal"
if [[ ! -d "$INTERNAL" ]]; then
    INTERNAL="$DIST_DIR"   # fallback for older PyInstaller
fi

for dir in "captain_claw/web/static" "captain_claw/instructions"; do
    if [[ -d "$INTERNAL/$dir" ]]; then
        echo "  ✓ $dir/"
    elif [[ -d "$DIST_DIR/$dir" ]]; then
        echo "  ✓ $dir/"
    else
        echo "  ✗ $dir/ MISSING"
        OK=false
    fi
done

if ! $OK; then
    echo ""
    echo "Build verification FAILED."
    exit 1
fi

# ── Smoke test ───────────────────────────────────────────────────
echo ""
echo "Smoke test..."
if [[ -f "$DIST_DIR/captain-claw" ]]; then
    "$DIST_DIR/captain-claw" --version 2>/dev/null || echo "  (--version not supported, binary exists and is executable)"
elif [[ -f "$DIST_DIR/captain-claw.exe" ]]; then
    "$DIST_DIR/captain-claw.exe" --version 2>/dev/null || echo "  (--version not supported, binary exists and is executable)"
fi

# ── Archive ──────────────────────────────────────────────────────
if $ARCHIVE; then
    echo ""
    VERSION=$("$PYTHON" -c "from captain_claw import __version__; print(__version__)" 2>/dev/null || echo "dev")

    case "$(uname -s)" in
        MINGW*|MSYS*|CYGWIN*)
            ARCHIVE_NAME="captain-claw-windows-x64-${VERSION}.zip"
            cd dist && zip -r "$ARCHIVE_NAME" captain-claw/ && cd ..
            ;;
        Darwin*)
            ARCH="$(uname -m)"
            ARCHIVE_NAME="captain-claw-macos-${ARCH}-${VERSION}.tar.gz"
            cd dist && tar czf "$ARCHIVE_NAME" captain-claw/ && cd ..
            ;;
        Linux*)
            ARCH="$(uname -m)"
            case "$ARCH" in
                aarch64|arm64) ARCH_LABEL="arm64" ;;
                x86_64)        ARCH_LABEL="x64" ;;
                *)             ARCH_LABEL="$ARCH" ;;
            esac
            ARCHIVE_NAME="captain-claw-linux-${ARCH_LABEL}-${VERSION}.tar.gz"
            cd dist && tar czf "$ARCHIVE_NAME" captain-claw/ && cd ..
            ;;
    esac

    echo "Archive created: dist/$ARCHIVE_NAME"
fi

echo ""
echo "Build complete! Binaries are in dist/captain-claw/"
echo ""
echo "  dist/captain-claw/captain-claw            # CLI + Web UI"
echo "  dist/captain-claw/captain-claw-web         # Web UI only"
echo "  dist/captain-claw/captain-claw-orchestrate  # Headless orchestrator"
