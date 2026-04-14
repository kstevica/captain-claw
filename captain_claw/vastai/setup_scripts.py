"""Shell script templates for provisioning Ollama on vast.ai instances.

These scripts are passed as the ``onstart`` command when creating
a vast.ai instance.  They run once after the container boots.
"""

from __future__ import annotations


def ollama_setup_script(
    pre_pull_model: str = "",
    ollama_host: str = "0.0.0.0:11434",
) -> str:
    """Return a bash on-start script that installs and starts Ollama.

    The script:
    1. Installs Ollama via the official installer (skips if already present).
    2. Starts ``ollama serve`` in the background bound to all interfaces.
    3. Waits for the Ollama API to become healthy.
    4. Optionally pulls a model if *pre_pull_model* is set.

    Parameters
    ----------
    pre_pull_model:
        Model tag to automatically pull after Ollama starts
        (e.g. ``"llama3.2"``).  Empty string = skip.
    ollama_host:
        Bind address for ``ollama serve``.
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

    return f"""#!/bin/bash
set -e

echo "[captain-claw] Starting Ollama setup..."

# Install Ollama if not already present (vastai/ollama image has it pre-installed).
if ! command -v ollama &>/dev/null; then
    echo "[captain-claw] Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "[captain-claw] Ollama already installed: $(ollama --version 2>/dev/null || echo unknown)"
fi

# Start Ollama server in the background if not already running.
if ! curl -sf http://localhost:11434/api/version > /dev/null 2>&1; then
    echo "[captain-claw] Starting Ollama server on {ollama_host}..."
    OLLAMA_HOST="{ollama_host}" nohup ollama serve > /var/log/ollama.log 2>&1 &
    echo "[captain-claw] Ollama PID: $!"

    # Wait for Ollama to become healthy (up to 120 seconds).
    echo "[captain-claw] Waiting for Ollama health check..."
    for i in $(seq 1 60); do
        if curl -sf http://localhost:11434/api/version > /dev/null 2>&1; then
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

# Verify GPU is visible.
echo "[captain-claw] GPU status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi not available)"
{pull_block}
echo "[captain-claw] Setup complete."
"""


def env_vars_for_instance(bearer_token: str) -> dict[str, str]:
    """Return environment variables for a vast.ai instance.

    Configures the vast.ai base image's built-in auth and HTTPS.

    Parameters
    ----------
    bearer_token:
        The token that clients will use as ``Authorization: Bearer <token>``
        to access services on this instance.
    """
    return {
        "ENABLE_AUTH": "true",
        "OPEN_BUTTON_TOKEN": bearer_token,
        "ENABLE_HTTPS": "true",
        # Ollama binds to all interfaces inside the container.
        "OLLAMA_HOST": "0.0.0.0:11434",
    }
