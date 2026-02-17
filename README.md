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

Edit `config.yaml` or use environment variables:

```yaml
model:
  provider: "openai"
  model: "gpt-4o-mini"
  temperature: 0.7
```

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
