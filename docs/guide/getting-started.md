# Getting started

[← README](../../README.md) · [All guides](README.md)

From `pip install` to a running fleet: the launchers, a deck for a team, a voice device on your desk, Docker, and the configuration file.

## Install and run

```bash
pip install captain-claw
export OPENAI_API_KEY="sk-..."          # or ANTHROPIC_API_KEY, GEMINI_API_KEY, etc.
captain-claw-web                         # http://127.0.0.1:23080
```

```bash
flight-deck               # Flight Deck multi-agent dashboard  (alias of captain-claw-fd)
captain-claw-fd           # Flight Deck multi-agent dashboard
captain-claw-agent        # Agent web server                   (alias of captain-claw-web)
captain-claw-web          # Agent web server
captain-claw              # Interactive terminal
captain-claw --tui        # Terminal UI
captain-claw-mcp          # MCP server for Claude Desktop
botport                   # Agent-to-agent routing hub
```

`flight-deck` and `captain-claw-agent` are friendly aliases added in 0.5.3 — `flight-deck` launches the dashboard, `captain-claw-agent` runs a single agent's web server. (Editable installs: run `pip install -e .` once so the new entry points are generated.)

First run starts onboarding automatically. For Ollama, no key needed — set `provider: ollama` in `config.yaml`.

**Optional extras:** `pip install 'captain-claw[cv]'` for local OpenCV vision, `pip install 'captain-claw[pdf]'` for PDFs read with their headings and tables (PyMuPDF, AGPL-3.0), `pip install 'captain-claw[whatsapp-web]'` to let users link their own WhatsApp (experimental; see [WhatsApp Web](devices-and-channels.md#whatsapp-web--your-own-whatsapp)).

## A deck for a team

Before the second person logs in, turn on auth, fix the JWT secret and tell the deck its public address:

```bash
FD_AUTH_ENABLED=true FD_JWT_SECRET=$(openssl rand -hex 32) FD_PUBLIC_URL=https://deck.example.com flight-deck
```

The first account becomes the admin. The full checklist (sharing, team keys, Gmail sending, inbound MCP for Claude) is in [docs/team-deployment.md](../team-deployment.md).

With sign-in on, agents stay in their own files and non-admins' agents don't run code (0.8.4); see the [checklist](../team-deployment.md#what-the-launch-hardening-changed-08x).

## A voice device on your desk

Flash an M5Stack StickS3, StopWatch or PaperColor, or a Waveshare ePaper 1.54 (the deck needs a Soniox key, in Connections → Soniox or as the admin's team key):

```bash
uv tool install platformio
cd firmware/device
pio run -e stopwatch -t upload     # or: -e sticks3, -e papercolor_b, -e epaper154, -e epaper154_touch
```

Then join the device's hotspot, open `http://192.168.4.1`, enter your Wi-Fi, the deck address and an access token from Connections → *Agent access for Claude (MCP)*. See [firmware/device/README.md](../../firmware/device/README.md).

More in [Devices and channels](devices-and-channels.md).

## Docker

```bash
docker pull kstevica/captain-claw:latest
docker run -d -p 23080:23080 \
  -v $(pwd)/config.yaml:/app/config.yaml:ro \
  -v $(pwd)/.env:/app/.env:ro \
  -v $(pwd)/docker-data/home-config:/root/.captain-claw \
  -v $(pwd)/docker-data/workspace:/data/workspace \
  kstevica/captain-claw:latest
```

See [README_DETAILED.md](../../README_DETAILED.md#docker) for Docker Compose and persistent data setup.

## One-click deploy

[![Deploy on RepoCloud](https://d16t0pc4846x52.cloudfront.net/deploylobe.svg)](https://repocloud.io/details/Captain%20Claw/)

## Configuration

YAML-driven with environment variable overrides (`CLAW_` prefix).

```yaml
model:
  provider: gemini
  model: gemini-2.5-flash
  allowed:
    - id: claude-sonnet
      provider: anthropic
      model: claude-sonnet-4-20250514
    - id: gpt-4o
      provider: openai
      model: gpt-4o

web:
  enabled: true
  port: 23080
```

**Load precedence:** `./config.yaml` > `~/.captain-claw/config.yaml` > env vars > `.env` > defaults.

Full reference: [USAGE.md](../../USAGE.md#configuration-reference) (23 config sections).

## Next

- [Flight Deck](flight-deck.md) — what to do once the dashboard is open
- [BotPort and MCP](botport-and-mcp.md) — connect your decks to each other and to Claude
- [Inside an agent](agents.md) — every supported model and the 65 built-in tools
