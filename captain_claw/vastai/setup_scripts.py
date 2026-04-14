"""Shell script templates for provisioning Ollama on vast.ai instances.

These scripts are passed as the ``onstart`` command when creating
a vast.ai instance.  They run once after the container boots.
"""

from __future__ import annotations


def ollama_setup_script(
    pre_pull_model: str = "",
    secure: bool = True,
    bearer_token: str = "",
) -> str:
    """Return a bash on-start script that installs and starts Ollama.

    The script:
    1. Installs Ollama via the official installer (skips if already present).
    2. Starts ``ollama serve`` in the background.
    3. Waits for the Ollama API to become healthy.
    4. Optionally installs an nginx reverse proxy with Bearer token auth.
    5. Optionally pulls a model if *pre_pull_model* is set.

    Parameters
    ----------
    pre_pull_model:
        Model tag to automatically pull after Ollama starts
        (e.g. ``"llama3.2"``).  Empty string = skip.
    secure:
        When True, install nginx as a reverse proxy on port 11434 that
        validates the Bearer token.  Ollama then only listens on
        localhost:11435 (inaccessible from outside).
    bearer_token:
        Required when *secure* is True.  The expected Bearer token.
    """
    pull_block = ""
    if pre_pull_model:
        # Sanitize the model tag — only allow safe characters.
        safe_model = "".join(
            c for c in pre_pull_model if c.isalnum() or c in ".-_:/"
        )
        pull_block = f"""
echo "[captain-claw] Pulling model: {safe_model}"
ollama pull "{safe_model}" 2>&1 | tail -5
echo "[captain-claw] Model pull complete: {safe_model}"
"""

    if secure:
        # Secure mode: Ollama on localhost:11435, nginx proxy on 0.0.0.0:11434
        ollama_host = "127.0.0.1:11435"
        ollama_health_url = "http://127.0.0.1:11435/api/version"
        nginx_block = _nginx_proxy_block(bearer_token)
    else:
        # Open mode: Ollama directly on 0.0.0.0:11434
        ollama_host = "0.0.0.0:11434"
        ollama_health_url = "http://localhost:11434/api/version"
        nginx_block = ""

    return f"""#!/bin/bash
set -e

echo "[captain-claw] Starting Ollama setup (secure={'yes' if secure else 'no'})..."

# Install Ollama if not already present (ollama/ollama image has it pre-installed).
if ! command -v ollama &>/dev/null; then
    echo "[captain-claw] Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "[captain-claw] Ollama already installed: $(ollama --version 2>/dev/null || echo unknown)"
fi

# Start Ollama server in the background if not already running.
if ! curl -sf {ollama_health_url} > /dev/null 2>&1; then
    echo "[captain-claw] Starting Ollama server on {ollama_host}..."
    OLLAMA_HOST="{ollama_host}" nohup ollama serve > /var/log/ollama.log 2>&1 &
    echo "[captain-claw] Ollama PID: $!"

    # Wait for Ollama to become healthy (up to 120 seconds).
    echo "[captain-claw] Waiting for Ollama health check..."
    for i in $(seq 1 60); do
        if curl -sf {ollama_health_url} > /dev/null 2>&1; then
            echo "[captain-claw] Ollama is healthy after $((i*2)) seconds."
            break
        fi
        if [ "$i" -eq 60 ]; then
            echo "[captain-claw] ERROR: Ollama failed to start within 120 seconds."
            tail -20 /var/log/ollama.log 2>/dev/null
            exit 1
        fi
        sleep 2
    done
else
    echo "[captain-claw] Ollama already running."
fi
{nginx_block}
# Verify GPU is visible.
echo "[captain-claw] GPU status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi not available)"
{pull_block}
echo "[captain-claw] Setup complete."
"""


def _nginx_proxy_block(bearer_token: str) -> str:
    """Return a bash snippet that installs and configures nginx as a
    reverse proxy in front of Ollama with Bearer token authentication.

    nginx listens on ``0.0.0.0:11434`` (the externally-mapped port)
    and proxies to Ollama on ``127.0.0.1:11435``.
    """
    # Escape any characters that might break the nginx config.
    safe_token = bearer_token.replace("\\", "\\\\").replace('"', '\\"')

    return f"""
# ── Secure proxy: nginx with Bearer token auth ──
echo "[captain-claw] Installing nginx for secure proxy..."
apt-get update -qq && apt-get install -y -qq nginx > /dev/null 2>&1

cat > /etc/nginx/sites-available/ollama-proxy <<'NGINX_EOF'
server {{
    listen 0.0.0.0:11434;

    # Generous timeouts for large model pulls / long inference.
    proxy_read_timeout 600s;
    proxy_send_timeout 600s;
    client_max_body_size 0;

    location / {{
        # Validate Bearer token.
        set $expected "Bearer {safe_token}";
        if ($http_authorization != $expected) {{
            return 401 '{{"error": "unauthorized"}}';
        }}

        proxy_pass http://127.0.0.1:11435;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_buffering off;
    }}
}}
NGINX_EOF

# Disable default site, enable ours.
rm -f /etc/nginx/sites-enabled/default
ln -sf /etc/nginx/sites-available/ollama-proxy /etc/nginx/sites-enabled/
nginx -t 2>&1 && nginx
echo "[captain-claw] nginx proxy running on :11434 -> Ollama :11435 (auth required)"
"""


def env_vars_for_instance(bearer_token: str, *, secure: bool = True) -> dict[str, str]:
    """Return environment variables for a vast.ai instance.

    Parameters
    ----------
    bearer_token:
        The token that clients will use as ``Authorization: Bearer <token>``
        to access services on this instance.
    secure:
        When True, Ollama binds to localhost only (nginx handles external
        traffic).  When False, Ollama binds to all interfaces directly.
    """
    return {
        # Ollama bind address depends on security mode.
        "OLLAMA_HOST": "127.0.0.1:11435" if secure else "0.0.0.0:11434",
    }
