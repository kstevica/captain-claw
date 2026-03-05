#!/bin/sh
set -e

CONFIG="/root/.captain-claw/config.yaml"
mkdir -p /root/.captain-claw

# Seed the home config with Docker-required overrides on first run.
# On subsequent runs the file already exists (persisted via bind mount)
# and may contain settings saved from the web UI — don't overwrite it.
if [ ! -f "$CONFIG" ]; then
  cat > "$CONFIG" <<EOF
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
  sed -i 's/host:.*127\.0\.0\.1/host: "0.0.0.0"/' "$CONFIG"
fi

# If the first argument looks like a flag (e.g. --tui, --port),
# prepend the default command so `docker run image --tui` works.
if [ "${1#-}" != "$1" ]; then
  set -- captain-claw "$@"
fi

exec "$@"
