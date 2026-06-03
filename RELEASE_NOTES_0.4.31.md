# Captain Claw v0.4.31 Release Notes

**Release title:** Video Understanding

**Release date:** 2026-06-03

## Highlights

Captain Claw 0.4.31 adds **`video_vision`** — the fleet can now watch and describe videos, end to end. Attach a clip in Flight Deck/glasses or send one over WhatsApp and the server deterministically samples frames, transcribes the audio (with timestamps), describes each frame with a vision model, and synthesizes one coherent description — streaming progress to you while it works.

The whole pipeline runs **server-side and deterministically**: the agent never decides how many frames to grab or writes its own extraction script — it just consumes the constructed analysis.

## What changed

### New: `video_vision` tool
- **Frames + audio + synthesis.** Samples frames at a fixed cadence (**first frame ~1s in, then every 6s**, up to 20; longer clips widen the interval so the tail isn't dropped), transcribes the audio via Soniox **with timestamps**, describes each frame with vision, and synthesizes a single description that pairs visuals with what's said over time.
- **Deterministic frame extraction** — one precise `ffmpeg` seek per timestamp, so the count is exactly what's scheduled (no fps-sampler dropping the last frame).
- **Segment support** — optional `start`/`end` (`'90'`, `'1:30'`, `'1:02:03'`) and an optional `interval` override.
- **Works on text-only agents** — if the calling agent has no vision model, frame description is delegated to a multimodal peer (e.g. an Ollama vision agent) over the same Flight Deck consult path used for images, one frame at a time.
- **Compressed audio** — 16 kHz mono **Opus** for the Soniox upload (≈5–7 MB/hour vs ≈115 MB for WAV), with a WAV fallback.

### Deterministic auto-analysis on attach
- Attaching a video (Flight Deck/glasses upload **or** WhatsApp) runs `video_vision` **server-side before the agent turn**, then feeds the analysis into the message. The agent only writes the answer.
- During a video turn the **`scripts` and `shell` tools are blocked**, so the agent can't burn time/tokens writing its own cv2/ffmpeg or "save-to-file" scripts.

### WhatsApp + glasses
- New inbound **WhatsApp `video`** handler: downloads the clip, hands it to the agent, and analyzes it automatically.
- **Progressive updates** while it works — "🎙 Transcribing…", the transcript (or a summary + key excerpts for long ones), then a live "🎞 Analyzing N frames…" — mirrored to WhatsApp and the web/glasses UI.
- Clear, disambiguated audio status: *no audio track* vs *transcription not configured* vs *no speech detected*.

### Infrastructure
- **Upload gate raised to 800 MB** and video extensions (`.mp4 .mov .webm .mkv .avi .m4v`) allowed, so videos can actually be attached.
- **Flight Deck shares secrets with agents** at spawn — `SONIOX_API_KEY` and the WhatsApp Cloud API creds are propagated from FD's env into each agent (when the agent didn't set them), so the keys live in one place.
- The FD upload proxy now surfaces the agent's real error (e.g. *"Unsupported file type"*) instead of a bare status code.

## How to use

Attach a video and ask — that's it:

```
describe this video            (Flight Deck / glasses upload, or a WhatsApp video)
```

Or call the tool directly for finer control:

```
video_vision(path="saved/clip.mp4")                      # whole video
video_vision(path="saved/clip.mp4", start="0:10", end="0:30")
video_vision(path="saved/clip.mp4", interval=3)          # a frame every 3s
```

## Requirements

- **`ffmpeg`** (provides `ffmpeg` + `ffprobe`) must be installed on the agent host: `apt install ffmpeg` / `brew install ffmpeg`.
- **`SONIOX_API_KEY`** for audio transcription (otherwise the video is analyzed frames-only). With FD secret-sharing, setting it once in Flight Deck's env is enough.
- A **vision model** on the agent, or a multimodal **peer** in the fleet, for the per-frame descriptions.

## Backward compatibility

Fully compatible with 0.4.30. `video_vision` is additive; everything else is unchanged. If `ffmpeg` isn't installed the tool returns a clear error rather than failing silently.

## Upgrade

```bash
git pull
sudo apt install -y ffmpeg          # if not already installed on the agent host
# rebuild the Flight Deck UI only if you build assets locally (they're committed):
npm --prefix flight-deck run build
# restart Flight Deck, then (re)spawn the agents so they pick up the shared env
```
