# Captain Claw v0.4.29 Release Notes

**Release title:** Multi-Agent Vision & Reliable Hand-offs

**Release date:** 2026-06-02

## Highlights

Captain Claw 0.4.29 makes the fleet genuinely collaborative around **images** and hardens the **delivery path** between agents. Agents can now hand files to each other, multimodal models see images natively, a text-only agent can delegate vision to a multimodal peer — and a set of integrity guards stop agents from racing, leaking internal context, or *claiming* they did something they didn't.

Everything is additive and backward compatible with 0.4.28.

## What changed

### 1. Agent-to-agent file transfer (`flight_deck` tool)

The `flight_deck` tool gained a `file` parameter on `consult`/`delegate`. The sender resolves a local file (its `saved/` workspace or an absolute path); **Flight Deck** uploads it to the target agent — using the target's resolved auth token, so it works even when peers require auth — and forwards the resulting path into the target's chat as `image_paths`/`file_paths`. So *"send this image to MiniMax and ask what's in it"* actually delivers the file to the peer.

### 2. Image / vision pipeline

A text-only chat model can't see images; this release makes images work end-to-end regardless of the agent's model:

- **Inline images for multimodal Ollama models** — attached images are resized (~1568px, to bound tokens) and sent inline via Ollama's per-message `images[]` array. So minimax-m3, llava, qwen-vl, etc. **see images directly** — no separate vision model needed.
- **`image_vision` always available** (Eco-core) and now **falls back to the agent's own chat model** (resized base64) when no `model_type: "vision"` model is configured.
- **`read` refuses images/binaries** with an actionable "use image_vision" message instead of the old "file too large" / "utf-8 can't decode" errors.
- **Manual image upload fixed** — the chat composer's upload endpoint now accepts image extensions, and images are tagged `[Attached image: …]` so the agent routes them to vision.
- **Delegate-to-see** — guidance now tells a blind agent to delegate the image to a multimodal peer via `flight_deck(file=…)` and relay the result.

### 3. Delivery integrity (multi-agent hand-offs)

Fixes the duplicate "I sent it, waiting" replies, re-planning, and leaks seen with small models over WhatsApp:

- **Serialized inbound queue** — delegate/consult results and triggered peer notifications now drain through a single `_inbound_queue` consumer, one-at-a-time, only when the agent is free. Replaces the previous racy "route-when-free / append-to-a-list" split, so peer results never interleave with the current turn or each other.
- **Delegate-result framing** — the returned result is explicitly framed as *"this is the ANSWER — relay it, don't delegate again, don't say you're waiting"*, so the model stops re-planning.
- **False-action-claim gate** — if a reply *claims* it delegated/sent to a peer ("Poslao sam…", "I delegated it", "čekam odgovor") but no `flight_deck`/`consult_peer` tool was actually called that turn, a corrective silent retry fires (bounded). Catches the model lying about hand-offs; truthful acks after a real delegate pass through.
- **Internal-context strip** — models that echo the injected `[INTERNAL CONTEXT … [END INTERNAL CONTEXT]` block (todos, fleet info) into their reply now have it stripped before it reaches the user.
- **Bounded inline-image cost** — reflection/insight prompts that concatenate many `[Attached image:]` markers no longer re-encode a dozen images per call (capped to the last 2).

## How to use

**Send an image to a peer:** *"send this photo to MiniMax and ask what's in it"* — the file is transferred to the peer and described.

**See an image (multimodal Ollama model):** just attach it and ask — the model sees it inline. No vision model required.

**See an image (text-only model):** configure a vision model (`model_type: "vision"` — can be a local Ollama vision model, no API key), or let the agent delegate to a multimodal peer.

## Backward compatibility

Fully compatible with 0.4.28. The image-inline path only triggers when an image is attached; the inbound queue and integrity gates are transparent; agent-to-agent file transfer is opt-in via the tool's `file` parameter.

## Upgrade

```bash
git pull
# rebuild the Flight Deck UI only if you build assets locally (they're committed):
npm --prefix flight-deck run build
# restart Flight Deck and the agents
```
