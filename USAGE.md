# Captain Claw - User Guide

## Quick Start

```bash
cd /home/kstevica/.openclaw/workspace/captain-claw-dev
source venv/bin/activate
captain-claw
```

## Commands

| Command | Description |
|---------|-------------|
| `/help` | Show help |
| `/config` | Show current configuration |
| `/history` | Show conversation history |
| `/compact` | Manually compact old session history |
| `/planning on|off` | Toggle planning mode and task pipeline orchestration |
| `/clear` | Clear current session |
| `/new [name]` | Start a new session (optionally named) |
| `/sessions` | List recent sessions |
| `/session` | Show active session info |
| `/session list` | List recent sessions |
| `/session new [name]` | Create and switch to a new session |
| `/session switch <id|name|#index>` | Switch to another session |
| `/session rename <new-name>` | Rename the active session |
| `/session description <text>` | Set active session description (stored in metadata) |
| `/session description auto` | Auto-generate description from session context/tasks |
| `/session model` | Show active model for current session |
| `/session model list` | List allowed models from config |
| `/session model <id|#index|provider:model|default>` | Select per-session model live |
| `/session run <id|name|#index> <prompt>` | Run one prompt in another session, then return |
| `/models` | List allowed models from config |
| `/runin <id|name|#index> <prompt>` | Alias for `/session run` |
| `/exit` or `/quit` | Exit |

## Interacting

Just type your message and press Enter. Captain Claw will:
1. Process your request
2. Use tools if needed (shell, read, write, glob, web_fetch)
3. Return the response

### File Output Policy

- Tool-generated files are saved under `<captain-claw launch directory>/saved`.
- If `saved/` does not exist, Captain Claw creates it automatically.
- Relative paths are written under `saved/`.
- Absolute or traversal paths are remapped under `saved/` for safety.

### Script Workflow

- Captain Claw can decide to generate and run scripts for automation-style tasks.
- If you explicitly ask to generate/create/build a script or tool, script workflow is mandatory.
- Generated scripts are saved under `saved/scripts/<session>/` and executed from that directory.
- Reusable helper tools are saved under `saved/tools/<session>/`.

### Web Fetch Modes

`web_fetch` supports two extraction modes:
- Default (`extract_mode="text"`): parses page HTML with BeautifulSoup and returns human-readable text for the LLM, preserving links as `label (url)` for follow-up research.
- Raw HTML (`extract_mode="html"`): returns the original HTML response.
- Output length defaults to `tools.web_fetch.max_chars` in `config.yaml` (default `100000` chars) and can be overridden per call with `max_chars`.

Examples:

```
Use web_fetch on https://example.com
Use web_fetch on https://example.com with extract_mode="html"
```

### Examples

```
> What files are in the current directory?
> Read the file /home/kstevica/.openclaw/workspace/README.md
> Run the command "ls -la"
> What's the weather in Zagreb?
```

## Configuration

Edit `config.yaml` to change:
- Model (currently: `minimax-m2.5:cloud`)
- Temperature
- Enabled tools
- Shell timeout and safety settings

## Testing the Agent

```bash
# Test Ollama connection
curl http://127.0.0.1:11434/api/tags

# Run with debug logging
captain-claw --verbose
```

## Troubleshooting

**No response?**
- Check Ollama is running: `curl http://127.0.0.1:11434/api/tags`
- Check model is available: `captain-claw --verbose`

**Errors?**
- Run with `--verbose` flag for debug output
- Check `config.yaml` is valid YAML
