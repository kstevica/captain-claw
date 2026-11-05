#!/bin/bash
# Build "Captain for Mac.app" (and a zip of it the deck hands out) into out/.
#   ./build.sh            release build, ad-hoc signed
#   SIGN_ID="Developer ID Application: …" ./build.sh   signed with that identity
set -euo pipefail
cd "$(dirname "$0")"
VERSION="${VERSION:-0.1.1}"
BUILD="$(date +%Y%m%d%H%M)"
APP="out/Captain for Mac.app"

LOG="$(mktemp)"
swift build -c release --arch arm64 --arch x86_64 >"$LOG" 2>&1 || { grep -E "error|warning: unre" "$LOG" | head -40; echo "swift build failed (log: $LOG)" >&2; exit 1; }
BIN="$(swift build -c release --arch arm64 --arch x86_64 --show-bin-path)/CaptainForMac"
[ -x "$BIN" ] || { echo "build failed: no $BIN" >&2; exit 1; }

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BIN" "$APP/Contents/MacOS/CaptainForMac"
sed -e "s/__VERSION__/$VERSION/" -e "s/__BUILD__/$BUILD/" Support/Info.plist > "$APP/Contents/Info.plist"
cp ../desktop/icons/icon.icns "$APP/Contents/Resources/AppIcon.icns"
codesign --force --deep --options runtime --timestamp=none --sign "${SIGN_ID:--}" "$APP" >/dev/null
codesign --verify --deep --strict "$APP"

( cd out && rm -f Captain-for-Mac.zip && ditto -c -k --keepParent "Captain for Mac.app" Captain-for-Mac.zip )
echo "Built $APP ($VERSION, $BUILD) and out/Captain-for-Mac.zip"
