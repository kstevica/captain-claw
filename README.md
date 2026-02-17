# Captain Claw

A powerful console-based AI agent built with Python.

## Features

- **Multi-provider LLM support** via LiteLLM (OpenAI, Anthropic, Ollama, Google, etc.)
- **Tool system** with shell, file operations, glob, and web fetch
- **Session management** with SQLite persistence
- **Interactive CLI** with streaming responses
- **Token tracking** and automatic context compaction

## Quick Start

```bash
# Install
cd captain-claw-dev
pip install -e .

# Set API key
export OPENAI_API_KEY="your-key-here"

# Run
captain-claw
```

## Configuration

Edit `config.yaml` or use environment variables.
By default Captain Claw loads `./config.yaml` (current working directory) first, then falls back to `~/.captain-claw/config.yaml`.

```yaml
model:
  provider: "openai"
  model: "gpt-4o-mini"
  temperature: 0.7
  allowed:
    - id: "chatgpt-fast"
      provider: "openai"
      model: "gpt-4o-mini"
    - id: "claude-sonnet"
      provider: "anthropic"
      model: "claude-3-5-sonnet-latest"
    - id: "gemini-flash"
      provider: "gemini"
      model: "gemini-2.0-flash"
    - id: "ollama-cloud"
      provider: "ollama"
      model: "minimax-m2.5:cloud"
```

Provider examples:

- ChatGPT: `provider: "openai"` (or `"chatgpt"`), `model: "gpt-4o-mini"`
- Claude: `provider: "anthropic"` (or `"claude"`), `model: "claude-3-5-sonnet-latest"`
- Gemini: `provider: "gemini"` (or `"google"`), `model: "gemini-2.0-flash"`
- Ollama: `provider: "ollama"`, `model: "llama3.2"` (or any local/remote Ollama model)

Per-session live selection is available in the CLI:

- `/models` to list allowlisted models
- `/session model <id>` to switch model for the active session
- `/session model default` to revert the session to config defaults

## Commands

- `captain-claw` - Start interactive session
- `captain-claw --model gpt-4o` - Override model
- `captain-claw --no-stream` - Disable streaming
- `captain-claw --verbose` - Enable debug logging
- `captain-claw version` - Show version

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check captain_claw/
```

## Architecture

- `agent.py` - Main agent orchestrator
- `llm/` - LLM provider abstraction
- `tools/` - Tool registry and implementations
- `session/` - SQLite session storage
- `cli.py` - Terminal UI
- `config.py` - Configuration management
