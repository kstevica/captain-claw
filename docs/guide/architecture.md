# Architecture

[← README](../../README.md) · [All guides](README.md)

Where each part of Captain Claw lives in the code.

| Component | Path |
|---|---|
| Agent (14-mixin composition) | `captain_claw/agent.py` |
| LLM providers | `captain_claw/llm/` |
| 65 tools + registry | `captain_claw/tools/` |
| Flight Deck (FastAPI + React) | `captain_claw/flight_deck/` |
| Device bridge (StickS3 / StopWatch / PaperColor / Waveshare, `/device/ws`) + OTA store + backgrounds | `captain_claw/flight_deck/device_bridge.py`, `device_firmware.py`, `device_backgrounds.py` |
| Device firmware (PlatformIO, every device) | `firmware/device/` |
| Kiosk File and Data libraries | `captain_claw/flight_deck/file_library.py`, `data_library.py` |
| Scheduled tasks | `captain_claw/flight_deck/fd_scheduler.py` |
| Admin console | `flight-deck/src/components/console/` |
| Reasoning effort per vendor | `captain_claw/llm/reasoning.py` |
| Payments (opt-in) | `captain_claw/payments/` |
| HUD for smart glasses (`/hud/`) | `flight-deck/src/hud/` |
| Agent links (inbound `/link/ws` + HTTP + webhooks, outbound connector) | `captain_claw/flight_deck/agent_link.py`, `agent_link_http.py`, `agent_link_out.py` |
| WhatsApp Web (linked-device service, supervisor, routes, auto-replies) | `captain_claw/flight_deck/whatsapp_web/`, `whatsapp_web_supervisor.py`, `whatsapp_web_routes.py`, `whatsapp_web_autoreply.py` |
| Owner notifications (WhatsApp, Telegram, devices, the bell) | `captain_claw/flight_deck/owner_notify.py` |
| Idle-agent sleep and pre-wake | `captain_claw/flight_deck/agent_idle.py` |
| Tool-call register (store, routes; agent-side recorder and digest) | `captain_claw/flight_deck/tool_audit_store.py`, `tool_audit_routes.py`; `captain_claw/tool_audit.py`, `tool_audit_digest.py` |
| Onboarding (setup and reading, knowledge browser, lessons, exams, practice) | `captain_claw/flight_deck/agent_preset*.py`, `agent_knowledge.py`, `agent_lessons.py`, `agent_onboarding.py`, `agent_exam.py`, `agent_practice.py` |
| Knowledge and lessons on the agent | `captain_claw/tools/knowledge.py`, `captain_claw/lessons.py` |
| Taskboard (sweeps, turn ingest, triage, next steps, threads, hand-over, proof; agent-side turn register) | `captain_claw/flight_deck/taskboard*.py`, `flight-deck/src/components/taskboard/`; `captain_claw/turn_register.py` |
| Confinement on sign-in decks | `captain_claw/deck_confine.py` |
| Your own AI keys | `captain_claw/flight_deck/byok.py` |
| Profile researcher | `captain_claw/flight_deck/profile_sources.py`, `profile_research.py`, `profile_routes.py` |
| Pictures in replies | `captain_claw/reply_images.py` |
| HTML to page and PNG (`visualize` tool) | `captain_claw/tools/visualize.py` |
| Word tracked changes (`docx` tool) | `captain_claw/docx_ops.py` |
| OneDrive VFS mounts | `captain_claw/onedrive_client.py` |
| Bat (plan, loop, judge, store) | `captain_claw/flight_deck/bat_*.py` |
| Fabrication guard | `captain_claw/fabrication_guard.py` |
| Lupa research desk (on the Flight Deck API) | `lupa/` |
| DAG orchestrator | `captain_claw/session_orchestrator.py` |
| Semantic memory (vector + BM25) | `captain_claw/semantic_memory.py` |
| Deep memory (Typesense) | `captain_claw/deep_memory.py` |
| Insights (fact extraction) | `captain_claw/insights.py` |
| Nervous system (dreaming) | `captain_claw/nervous_system.py` |
| Cognitive tempo | `captain_claw/cognitive_tempo.py` |
| MCP server | `captain_claw/mcp_serve.py` |
| BotPort client (agent, legacy) | `captain_claw/botport_client.py` |
| BotPort link (Flight Deck ⇄ other decks) | `captain_claw/flight_deck/botport_link.py`, `botport_routes.py` |
| BotPort hub | `botport/botport/` |
| Web UI + REST API | `captain_claw/web/` |
| Prompt templates (~100 files) | `captain_claw/instructions/` |
| Config (Pydantic) | `captain_claw/config.py` |

For every command, tool and config key, see [USAGE.md](../../USAGE.md).
