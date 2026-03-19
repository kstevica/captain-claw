#!/bin/sh
set -e

# Ensure HOME is set (some container runtimes may not set it).
if [ -z "$HOME" ]; then
  export HOME="/home/claw"
fi

CONFIG="$HOME/.captain-claw/config.yaml"
mkdir -p "$HOME/.captain-claw"

# Seed the home config with Docker-required overrides on first run.
# On subsequent runs the file already exists (persisted via bind mount)
# and may contain settings saved from the web UI — don't overwrite it.
if [ ! -f "$CONFIG" ]; then
  cat > "$CONFIG" <<EOF
model:
  provider: openai
  model: gpt-5-mini
  allowed:
    - id: gpt-5-mini
      provider: openai
      model: gpt-5-mini
      reasoning_level: high
      description: "Good for everyday tasks, light coding, reasoning"
    - id: gemini-flash-lite
      provider: gemini
      model: gemini-3.1-flash-lite-preview
      temperature: 0
      description: "simple and fast model"
web:
  host: "0.0.0.0"
session:
  path: /data/sessions/sessions.db
workspace:
  path: /data/workspace
skills:
  managed_dir: /data/skills
EOF
fi

# Always ensure web.host is 0.0.0.0 regardless of what the user saved,
# otherwise the container becomes unreachable.
if grep -q 'host:.*127\.0\.0\.1' "$CONFIG" 2>/dev/null; then
  TMP=$(mktemp /tmp/config.XXXXXX)
  sed 's/host:.*127\.0\.0\.1/host: "0.0.0.0"/' "$CONFIG" > "$TMP" && mv "$TMP" "$CONFIG"
fi

# Ensure Docker-required paths point to persistent /data volumes.
# Without these, data ends up in the ephemeral container layer and is
# lost on restart.
_ensure_path() {
  # Usage: _ensure_path <section> <key> <required_value>
  local section="$1" key="$2" required="$3"
  if grep -q "^${section}:" "$CONFIG" 2>/dev/null; then
    if ! grep -A5 "^${section}:" "$CONFIG" | grep -q "path:.*${required}"; then
      TMP=$(mktemp /tmp/config.XXXXXX)
      awk -v sec="$section" -v val="$required" '
        $0 ~ "^"sec":" { in_sec=1; print; next }
        in_sec && /^  path:/ { print "  path: " val; next }
        /^[^ ]/ { in_sec=0 }
        { print }
      ' "$CONFIG" > "$TMP" && mv "$TMP" "$CONFIG"
    fi
  fi
}
_ensure_path workspace path /data/workspace
_ensure_path session path /data/sessions/sessions.db

# If the first argument looks like a flag (e.g. --tui, --port),
# prepend the default command so `docker run image --tui` works.
if [ "${1#-}" != "$1" ]; then
  set -- captain-claw "$@"
fi

exec "$@"
