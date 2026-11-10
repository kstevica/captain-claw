# Getting started

[← README](../../README.md) · [All guides](README.md)

From `pip install` to a running fleet: the launchers, a deck for a team, a voice device on your desk, Docker, and the configuration file.

Prefer a video? Watch **[the complete Captain guide](https://www.youtube.com/watch?v=VehoeF9ewdQ)**, or **[the admin tutorial](https://www.youtube.com/watch?v=IXkO3AbID38)** for running a deck for a team. More on the **[Captain YouTube channel](https://www.youtube.com/@captain--ai-agents)**.

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

## The desktop app

**Captain** for macOS and Windows (Linux untested so far) is a 75 MB download from [captain-claw.com/download](https://captain-claw.com/download): the window, without Python inside. Its first start installs everything else from the newest release, about 200 MB, in half a minute on a fast line, with a page that shows each step. After that it starts in seconds, and **Check for Updates…** (the Captain menu on a Mac, Help elsewhere) installs each new release beside the running one, test-starts it and switches on a restart, going back by itself if the new version doesn't come up (0.8.6). It runs its own Flight Deck with sign-in off, for one person. To add your models, **Console → Models → New tier set → Quick start**: pick a provider and paste its key, and every tier gets its current models (0.8.10). **The first time you open it:** Captain isn't notarized yet, so macOS says it can't verify the app. Press **Done**, then **Open Anyway** in System Settings → Privacy & Security, and confirm. A download from 0.8.8's first hour that macOS calls *damaged* had no full signature; download it again. Building it and publishing a release for it: [README_BINARY.md](../../README_BINARY.md#2-desktop-app-electron).

## A deck for a team

Before the second person logs in, turn on auth, fix the JWT secret and tell the deck its public address:

```bash
FD_AUTH_ENABLED=true FD_JWT_SECRET=$(openssl rand -hex 32) FD_PUBLIC_URL=https://deck.example.com flight-deck
```

The first account becomes the admin. The full checklist (sharing, team keys, Gmail sending, inbound MCP for Claude) is in [docs/team-deployment.md](../team-deployment.md).

With sign-in on, agents stay in their own files and non-admins' agents don't run code (0.8.4); see the [checklist](../team-deployment.md#what-the-launch-hardening-changed-08x).

Once it's running from a git checkout, **Console → Updates** updates the deck itself, with a test start, a warning to everyone and a way back (0.8.5). See [Updating the deck](flight-deck.md#updating-the-deck).

## Coding agents as your crew

To have Claude Code, Codex, OpenCode, Crush (0.8.6) or Pi work for you, install the CLI on the deck's host and open **Crew**, which lists what's there and the command for what isn't (0.8.5). OpenCode's free models need no key. An admin can always use Crew; everyone else needs an admin's tick, and that tick trusts them with the server. See [Crew](flight-deck.md#crew).

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
- [Inside an agent](agents.md) — every supported model and the 79 built-in tools
