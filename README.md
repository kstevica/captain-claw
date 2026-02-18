# Captain Claw

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Interface](https://img.shields.io/badge/interface-terminal%20CLI-black)](#quick-start)
[![Models](https://img.shields.io/badge/models-OpenAI%20%7C%20Claude%20%7C%20Gemini%20%7C%20Ollama-orange)](#multi-model-ai-agent)
[![Guardrails](https://img.shields.io/badge/guardrails-input%20%7C%20output%20%7C%20script%2Ftool-red)](#built-in-guardrails)

Captain Claw is a terminal-first agentic system for everyday work. It helps developers, operators, and technical teams automate tasks with strong control using multi-model LLM support, persistent multi-session workflows, built-in safety guards, and tool execution in one CLI.

If you are looking for a powerful open-source agentic system for day-to-day terminal workflows, Captain Claw is built for that exact use case.

## First 5 Minutes

1. Install Captain Claw in a virtual environment.
2. Set one API key (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`) or use Ollama.
3. Start with `captain-claw`.
4. Run `/models` and pick a model with `/session model <id>`.
5. Create a second session (`/new incident-hotfix`) and switch models per session.
6. Enable guards in `config.yaml` when you want stricter safety behavior.

In under five minutes, you can have a multi-model terminal agentic system with persistent sessions and guardrails.

## Why Captain Claw

- Multi-model AI in one CLI: ChatGPT/OpenAI, Claude/Anthropic, Gemini/Google, and Ollama.
- Parallel work across sessions: keep separate short-term contexts per task while preserving long-term session history.
- Live per-session model switching: use different models for different threads without restarting.
- Built-in safety guards: input, output, and script/tool checks with configurable enforcement.
- Real tool execution: shell, file read/write, glob, and web fetch with persistent session traces.
- Practical for production-like workflows: monitor view, selectable pipeline modes, context compaction, and resumable sessions.

## Feature Snapshot

| Capability | Why it matters |
|---|---|
| Multi-model routing | Pick the best model for each task (quality, speed, cost). |
| Per-session model selection | Keep one session on GPT, another on Claude, another on Ollama. |
| Persistent multi-session state | Resume work exactly where you left off. |
| Built-in guard system | Reduce risky prompts, outputs, and command execution. |
| Built-in tools | Move from chat to action: shell, file ops, web fetch. |
| Planning + monitor modes | Better visibility for longer, multi-step agent runs. |
| Context compaction | Keep long threads usable without losing continuity. |

## Why Captain Claw vs Alternatives

| Criteria | Captain Claw | Typical Single-Model CLI | Chat-Only Web UI | Script-Only Automation Tool |
|---|---|---|---|---|
| Multi-provider support | Yes (OpenAI, Claude, Gemini, Ollama) | Usually limited | Usually single provider | Not applicable |
| Per-session model routing | Yes | Rare | No | No |
| Persistent multi-session memory | Yes | Limited | Usually tab-based, shallow | No conversational memory |
| Built-in input/output/script guards | Yes | Rare | Partial/moderation-only | No LLM guard layer |
| Tool execution in same loop | Yes | Varies | Usually no local execution | Yes, but no LLM orchestration |
| Terminal-first workflow | Yes | Yes | No | Yes |
| Designed for iterative human+agent ops | Yes | Partial | Partial | Partial |

## Core Strengths

### Multi-Model AI Agent

Captain Claw lets you route work to the right model for the job:

- `openai` / `chatgpt`
- `anthropic` / `claude`
- `gemini` / `google`
- `ollama`

This makes it easy to compare quality, cost, latency, and tool-calling behavior across providers from the same interface.

### Built for Everyday Work

Captain Claw is designed for practical everyday workflows, not just toy chat interactions.

- Code investigation and patching across multiple repositories.
- Incident response sessions with isolated contexts.
- Automated script generation with controlled execution.
- Fast web research and local file updates in one loop.
- Session handoff via persisted context and descriptions.

### Multi-Session Workflow

Sessions are first-class:

- Create named sessions for separate projects or incidents.
- Switch instantly between sessions.
- Rename sessions and set descriptions (manual or auto-generated from context).
- Run a prompt in another session and return to your current one.
- Persist model selection per session so each session can use a different model.
- Protect session memory from accidental reset with `/session protect on`.
- Procreate a new session from two parent sessions with compacted merged memory while keeping parents unchanged.
- Schedule recurring tasks with Captain Claw pseudo-cron (`/cron ...`) managed inside the app runtime.

### Built-In Guardrails

Captain Claw includes three built-in guard types:

- `input` guard: checks content before requests go to an LLM.
- `output` guard: checks model output before it is used or displayed.
- `script_tool` guard: checks scripts, commands, and tool payloads before execution.

Each guard supports:

- `enabled: false|true` (default is `false`)
- `level: stop_suspicious|ask_for_approval`

This allows strict blocking for sensitive environments or interactive approval for high-velocity usage.

## Common Use Cases

- Agentic assistant in terminal for coding, operations, research, and automation tasks.
- Multi-model evaluation workspace for OpenAI vs Claude vs Gemini vs Ollama.
- DevOps and SRE runbook execution with command guardrails.
- Security-conscious automation where suspicious actions require approval.
- Long-running project work split into dedicated sessions by feature or incident.

## Installation

### Requirements

- Python `>=3.11`

### Install From Source

```bash
git clone https://github.com/kstevica/captain-claw
cd captain-claw
python -m venv venv
source venv/bin/activate
pip install -e .
```

### Install Development Dependencies

```bash
pip install -e ".[dev]"
```

## Quick Start

### 1) Configure API keys

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"
```

Use only the keys you need for your active providers.

### 2) Start Captain Claw

```bash
captain-claw
```

### 3) Try model selection

```text
/models
/session model chatgpt-fast
/session model claude-sonnet
/session model default
```

### 4) Create multiple live sessions

```text
/new feature-a
/new bugfix-b
/sessions
/session switch #1
/session run #2 summarize current blockers
/session protect on
/session procreate #1 #2 "release merged context"
```

## Configuration

Captain Claw configuration is YAML-driven with env override support.

Load precedence:

1. `./config.yaml` (current working directory)
2. `~/.captain-claw/config.yaml` (fallback)

### Example `config.yaml`

```yaml
model:
  provider: "openai" # ollama, openai/chatgpt, anthropic/claude, gemini/google
  model: "gpt-4o-mini"
  temperature: 0.7
  max_tokens: 32000
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

guards:
  input:
    enabled: false
    level: "stop_suspicious" # or ask_for_approval
  output:
    enabled: false
    level: "stop_suspicious" # or ask_for_approval
  script_tool:
    enabled: false
    level: "stop_suspicious" # or ask_for_approval

tools:
  enabled: ["shell", "read", "write", "glob", "web_fetch", "web_search"]
  web_search:
    provider: "brave"
    api_key: "" # or BRAVE_API_KEY env var
    max_results: 5
    safesearch: "moderate"

workspace:
  path: "./workspace" # local artifact root; tool outputs go under ./workspace/saved
```

### Useful Environment Overrides

```bash
CLAW_MODEL__PROVIDER="openai"
CLAW_MODEL__MODEL="gpt-4o-mini"
CLAW_GUARDS__INPUT__ENABLED="true"
CLAW_GUARDS__INPUT__LEVEL="ask_for_approval"
CLAW_TOOLS__WEB_SEARCH__API_KEY="your_brave_api_key"
# or use BRAVE_API_KEY="your_brave_api_key"
```

## CLI Commands

### Global Commands

| Command | Description |
|---|---|
| `/help` | Show command help |
| `/config` | Show active configuration |
| `/history` | Show conversation history |
| `/compact` | Manually compact older session history |
| `/pipeline loop\|contracts` | Select pipeline mode (`loop` for fast/simple, `contracts` for planner+completion gate) |
| `/planning on\|off` | Legacy alias for `/pipeline contracts\|loop` |
| `/monitor on\|off` | Enable/disable monitor split view |
| `/monitor trace on\|off` | Enable/disable full intermediate LLM response trace logging into monitor/session history |
| `/monitor pipeline on\|off` | Enable/disable compact pipeline-only trace logging into session history |
| `/monitor full on\|off` | Enable/disable raw full tool output rendering in monitor pane |
| `/scroll <chat\|monitor> <up\|down\|pageup\|pagedown\|top\|bottom> [n]` | Scroll one monitor pane independently |
| `/scroll status` | Show current monitor/chat pane scroll positions |
| `/clear` | Clear current session messages (blocked when `/session protect on`) |
| `/exit` or `/quit` | Exit Captain Claw |

### Session Commands

| Command | Description |
|---|---|
| `/session` | Show active session info |
| `/sessions` or `/session list` | List recent sessions |
| `/new [name]` | Create and switch to a new session |
| `/session new [name]` | Create and switch to a new session |
| `/session switch <id\|name\|#index>` | Switch to another session |
| `/session rename <new-name>` | Rename active session |
| `/session description <text>` | Set session description |
| `/session description auto` | Auto-generate description from session context |
| `/session export [chat\|monitor\|pipeline\|pipeline-summary\|all]` | Export active session chat/monitor history and compact pipeline trace files |
| `/session protect on\|off` | Enable/disable active-session memory reset protection |
| `/session procreate <id\|name\|#index> <id\|name\|#index> <new-name>` | Create a new session by merging compacted memory from two parent sessions |
| `/session run <id\|name\|#index> <prompt>` | Run one prompt in another session, then return |
| `/runin <id\|name\|#index> <prompt>` | Alias for `/session run` |

### Model Commands

| Command | Description |
|---|---|
| `/models` | List allowed models |
| `/session model` | Show active session model |
| `/session model list` | List allowed models |
| `/session model <id\|#index\|provider:model\|default>` | Set model for active session |

### Captain Claw Cron Commands

| Command | Description |
|---|---|
| `/cron "<task>"` | Run one-off task through Captain Claw (same runtime + guards as manual input) |
| `/cron run script <path>` | Run an existing script from the active session folder (validated before run) |
| `/cron run tool <path>` | Run an existing tool helper from the active session folder (validated before run) |
| `/cron add every <Nm\|Nh> <task\|script\|tool ...>` | Add interval pseudo-cron job |
| `/cron add daily <HH:MM> <task\|script\|tool ...>` | Add daily pseudo-cron job |
| `/cron add weekly <day> <HH:MM> <task\|script\|tool ...>` | Add weekly pseudo-cron job |
| `/cron list` | List active pseudo-cron jobs |
| `/cron history <job-id\|#index>` | Show persisted per-job cron chat history and monitor history |
| `/cron run <job-id\|#index>` | Run a saved cron job immediately |
| `/cron pause <job-id\|#index>` | Pause a cron job |
| `/cron resume <job-id\|#index>` | Resume a paused cron job |
| `/cron remove <job-id\|#index>` | Delete a cron job |

## Example Workflow

```text
> Investigate failing integration tests and propose a fix.
/session description auto
/models
/session model claude-sonnet
> Apply the patch and run tests.
/new release-notes
/session model chatgpt-fast
> Draft release notes from the previous session updates.
```

This workflow shows why Captain Claw works well as an agentic system for everyday engineering work: each thread can use its own model and maintain its own short-term context while staying persistent.

### Session Protection and Procreate Notes

- `/session protect on` prevents `/clear` from resetting the current session memory.
- `/session protect off` re-enables normal `/clear` behavior.
- `/session procreate ...` compacts each parent session memory snapshot before merging into the child session.
- Parent sessions are not compacted or modified during procreation.
- Session procreate writes progress steps to monitor output while resolving, compacting, merging, and creating the child session.

### Session Export Notes

- `/session export chat` writes a chat-only history export.
- `/session export monitor` writes a tool/monitor history export.
- `/session export pipeline` writes a compact pipeline trace export (`.jsonl`) without fetched/tool content bodies.
- `/session export pipeline-summary` writes a markdown digest of pipeline trace events (counts + compact timeline).
- `/session export` (or `/session export all`) writes all exports.
- With `/monitor trace on`, monitor exports include full intermediate `llm_trace` entries for planner/critic/main LLM calls.
- With `/monitor pipeline on`, session history stores compact `pipeline_trace` events intended for debugging instruction/pipeline behavior with minimal noise.
- Exports are session-scoped and saved under `saved/showcase/<session-id>/exports/`.

### Captain Claw Cron Notes

- Cron here is Captain Claw managed pseudo-cron, not system `cron`.
- Guardrails remain active before every cron-triggered prompt/script/tool run.
- Scheduled jobs are persisted and executed inside the Captain Claw runtime loop.
- Cron output is visually tagged (`[CRON ...]`) in chat and monitor panes for quick differentiation.
- Cron monitor output logs each execution step (`job_start`, `run_script_tool_start`, `job_done`, failures, and status updates).
- Each cron job stores its own chat history and monitor history; inspect with `/cron history <job-id|#index>`.

## Tooling and Execution Model

Captain Claw can use:

- `shell`: run commands
- `read`: read files
- `write`: write files
- `glob`: search files by pattern
- `web_fetch`: fetch and parse web content
- `web_search`: search the web (Brave API-backed) for current sources and links

### File Output Policy

- Tool-generated files are written under `<workspace-root>/saved` (default: `./workspace/saved`).
- Relative paths are resolved within that saved root.
- Writes are always session-scoped: category paths are normalized to `saved/<category>/<session-id>/...`; uncategorized paths are normalized to `saved/tmp/<session-id>/...`.
- Unsafe absolute/traversal paths are remapped for safety.
- Session-scoped paths always use stable session IDs, not mutable session names.

### Script Workflow

- Explicit script requests trigger script generation and execution workflow.
- Generated scripts are saved under `saved/scripts/<session-id>/`.
- Reusable tool helpers are saved under `saved/tools/<session-id>/`.
- For list-heavy requests (for example `for each`, `top N`, `all <items>`), Captain Claw first extracts list members from request/context, then chooses strategy:
  - direct loop mode: keeps member list in task memory and loops through pending members in completion guidance
  - script mode: auto-generates a temporary Python worker under `saved/tools/<session-id>/`, executes it with the active Python interpreter, then continues response completion
- This prevents premature stop after the first member and improves consistency for per-member tasks.

### Web Fetch Modes

- `extract_mode="text"` (default): parsed readable content with preserved links.
- `extract_mode="html"`: raw HTML response.

## Monitoring, Planning, and Context Management

- Monitor mode provides split output for chat and tool/system traces.
- `/monitor trace on` records full intermediate LLM responses (`llm_trace`) in monitor/session history for export and analysis.
- `/monitor pipeline on` records compact pipeline-only events (`pipeline_trace`) for low-noise diagnostics and smaller exports.
- `/monitor full on` disables compact monitor summarization (for example `web_fetch`) and renders raw tool output.
- `/scroll <chat|monitor> ...` scrolls chat and monitor panes independently while monitor mode is enabled.
- `/scroll status` prints current scroll offsets and limits.
- Planning mode adds task pipeline orchestration per turn, including nested task trees with leaf-based progress tracking.
- Long sessions auto-compact based on context thresholds and preserve continuity summary.

## Security and Guarding Behavior

When guards are enabled, checks run across interactions:

- Before LLM requests (`input`)
- After LLM responses (`output`)
- Before script/tool execution (`script_tool`)

Enforcement options:

- `stop_suspicious`: block immediately.
- `ask_for_approval`: ask for explicit user approval before continuing.

This makes Captain Claw suitable for teams that need a terminal agentic system with stronger operational control.

## FAQ

### Is Captain Claw only for coding tasks?

No. It is strong for coding, and equally useful for ops automation, web research, scripting, and multi-session task orchestration in everyday terminal workflows.

### Can I use local models only?

Yes. Set provider/model to Ollama and run fully local where your model setup allows it.

### Can I run different models at the same time?

Yes. Model selection is per session. Different sessions can run different providers/models in parallel workflow.

### Do I need guards enabled?

No. Guards are off by default. Enable them when you want stronger safety behavior for prompts, model outputs, and command/script execution.

## Development

```bash
# Run tests
pytest

# Lint
ruff check captain_claw/
```

## Architecture

- `captain_claw/agent.py`: main orchestration logic
- `captain_claw/llm/`: provider abstraction and adapters
- `captain_claw/tools/`: tool registry and tool implementations
- `captain_claw/session/`: SQLite-backed session persistence
- `captain_claw/cli.py`: terminal UI
- `captain_claw/config.py`: configuration and env overrides
- `instructions/`: externalized prompt/instruction templates
