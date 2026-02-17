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
| `/clear` | Clear current session |
| `/new [name]` | Start a new session (optionally named) |
| `/sessions` | List recent sessions |
| `/session <id|name>` | Load session by ID or latest by name |
| `/exit` or `/quit` | Exit |

## Interacting

Just type your message and press Enter. Captain Claw will:
1. Process your request
2. Use tools if needed (shell, read, write, glob, web_fetch)
3. Return the response

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
