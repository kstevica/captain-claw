# Captain Claw / Vatra — run hardening for long-form deliverables

**From:** Captain Spark (research/fiction front-end)
**Re:** Multi-agent run that produced a broken deliverable ("story run 3", intent = a fiction/mystery brief)
**Model in use:** DeepSeek v4.1 Flash for every agent (Lead, specialists, Reporter). This matters — most fixes below exist to make a *fast, weak* model reliable. Assume each agent will: mis-handle multi-step tool calls, truncate arguments, lose long context, and describe work instead of doing it. The orchestration has to catch that, not trust it.

**What Spark already did on its side (so we don't duplicate):** the composed intent now carries (a) a hard **delivery contract** — "the deliverable is the complete assembled work itself, one document, no summary, no file pointer, no placeholder"; (b) a **silent continuity pass** requirement (canon of ages/dates/whereabouts/sums/tallies, no future-dated facts); (c) fiction craft + fair-play rigor. These are *prompt-level*. A Flash-class model does not reliably obey prompt-level self-checks over 18k words. **Everything below is the structural backstop those prompts need.** Prompt + structure together, not either alone.

---

## 1. What happened (evidence from the run log)

The run spawned 5 specialists in two groups (A: Long-Horizon Planner, Deep Researcher, Fact Checker → build the bible/pack/audit; B: two "Editor & Long-form Writer" agents splitting the story into Part One and Part Two), then a Reporter synthesis step. Timeline of the failures, with log timestamps:

1. **VFS write persisted a placeholder instead of content — twice.**
   - `18:52:29` Deep Researcher writes `vfs:…/eppo-authenticity-pac` (note the **truncated path** — no `k`, no `.md`) with content `# THE EPPO AUTHENTICITY PACK ### Groundi…`. `18:53:21`: *"I need to correct that — the write captured a placeholder, not the pack. Writing the real content now."*
   - `19:08:16` Editor Part Two: *"Read tool is blocked; I have what I need."* → *"saving work"*. The resulting file `s5-story-bridge-ch6-7.md` shipped to Spark as **88 bytes**, its entire content the literal string `[written to disk: vfs:…/s5-story-bridge-ch6-7.md — use read tool to view]`. The real Chapters 6–7 were never persisted.

2. **Board waits are far too short for the production cadence, so agents give up on live producers.**
   - `18:50:51` Fact Checker waits on `'Story Bible clue register sealed space culprit' (≤90s)`; `18:52:21` **timed out** — *"posts … not ready"*.
   - `18:54:45` Fact Checker: *"No teammate artifacts are on the board yet (s1's bible and s2's pack haven't landed), so I can't audit them directly. I'll produce the audit … on a **reconstructed reference architecture**."* → **the QA agent audited a hallucinated reconstruction, never the real artifacts.**
   - `18:57:17` Editor waits `'Part One manuscript chapter one' (≤45s)` → timed out. `19:00:58` waits `'handoff chapter seven part one complete' (≤75s)` → `19:02:13` timed out.
   - Meanwhile the producers were *actively working*: Part One's own dispatch ran **274.9s**; the Planner's **395.7s**. The waiters' 45–90s windows expire long before a producer that takes ~5 minutes ever posts.

3. **Two writers raced on one artifact with no handshake.**
   - `19:00:57` Editor Part Two: *"Part One currently ends at Chapter Three … s4 may still be appending."* `19:04:47`: *"Part One has grown since I read it."* `19:04:55`: *"s4 … now has Chapters One–Five. My bridge would duplicate s4's work, so I'll drop it and wait."* `19:08:00`: *"s4 has stalled at Chapter Five … I'll supply the missing Chapters Six and Seven."*
   - Result: overlapping Ch 4–7 written twice, plus the placeholder stub for 6–7, plus a gap Part One never filled.

4. **The Lead denied the one request that would have closed the gap.**
   - `19:09:01` Editor Part Two files a clarify: *"supply Part One Chapters Six and Seven (the archive look and the Friday handoff) so the reviewer can use the canonical…"* `19:09:06`: **"Lead denied … request."** The coordination channel worked; the Lead refused the fix.

5. **The Reporter described the deliverable instead of emitting it.**
   - `19:09:31` Synthesizing → `19:15:06` Reporter ✓. The `truth` returned to Spark was a **~3 KB hand-off note** ("*Fair Measure is assembled and complete. File: `vfs:…/fair-measure.md`*") that **pointed at a file that was never written** to the VFS. There was no `fair-measure.md` and no assembled manuscript anywhere — only the scattered part files. Spark faithfully exported the note as the deliverable, because the note *was* the run's `truth`.

**Net:** a fast model, an over-eager board timeout, a raced artifact, a denied fix, a synthesis step that summarized instead of assembling, and a write tool that dropped content. Any one of these degrades the result; together they produced a deliverable that didn't contain the story.

---

## 2. Fixes, in priority order

### P0 — VFS write must never persist a placeholder, and must verify

This is the most damaging bug and the clearest. The write tool (or the agent scaffold around it) is letting the tool's *acknowledgement string* (`[written to disk: … — use read tool to view]`) become the file's *content*, and is truncating the path argument (`eppo-authenticity-pac`, `s5-story-bridge-ch4-7` with no `.md`).

- **Reject placeholder content at the tool boundary.** If a write's content is empty, or matches the tool's own readback/acknowledgement pattern (e.g. begins with `[written to disk` / `use read tool`), fail the write loudly and make the agent re-issue it. Never let a tool-result string round-trip into a file body.
- **Write-then-readback verify.** After every VFS write, read the first and last N bytes back and compare length/hash to what was sent. On mismatch, retry (up to k times) before surfacing success. A Flash model won't do this itself — do it in the tool wrapper.
- **Stop truncating the path.** `eppo-authenticity-pac` and the missing `.md` suffixes indicate the path param is being cut (token limit on the arg, or a bad split). Validate that a write path has an extension and round-trips intact; reject/repair otherwise.
- **Surface write failures to the agent as errors it must act on**, not as silent successes. Right now the agent only noticed by luck ("the write captured a placeholder").

### P0 — Synthesis must EMIT the assembled artifact, deterministically; never a pointer or a summary

The Reporter returning a description with a `vfs:` path is the failure Spark sees as "the deliverable is a 3 KB note."

- **Assemble by concatenation of named files, not by asking the model to "assemble."** For a piece-based deliverable, the synthesis step should be a deterministic merge of the declared part files (in declared order) into one artifact, then a light model pass only to smooth seams — not a from-memory rewrite. A Flash model asked to "assemble 5 pieces" will summarize them.
- **The run's returned `truth` must be the artifact's bytes, or a verified handle that actually resolves.** Before completing, assert the referenced file exists in the VFS and is non-trivial (size floor, chapter/section count for the declared structure). If synthesis names `fair-measure.md`, the run must not complete unless that file exists and contains the whole work.
- **Reject "description-as-deliverable."** If the synthesis output is short relative to the sum of its inputs, or reads as a note *about* the work (contains "assembled and complete", "see the file", a `vfs:`/`saved/` path as the payload), treat it as a failed synthesis and retry, don't ship it.

### P1 — Board timing: make waits adaptive and gate synthesis on real artifacts

The fixed 45/75/90s waits are calibrated for a model that posts in seconds; Flash agents post in minutes.

- **Key the wait to the producer, not a constant.** If a required upstream artifact has a known/assigned producer that is still `running`, waiters should extend automatically (heartbeat-based) rather than expire on a flat timeout. "Producer alive and working" ≠ "give up."
- **Emit progress heartbeats** ("s4: Chapter Five, still writing") so a waiter can distinguish *slow* from *stalled*. Run 3 shows the waiter guessing this by hand ("no change in ~4 minutes").
- **Hard gate: synthesis and QA must not start until their declared inputs have landed.** The Fact Checker auditing a *reconstruction* because the bible "hadn't landed" is the single most expensive failure — it's why the continuity errors slipped through. A QA/synthesis agent with missing inputs should **block or fail**, never fabricate a substitute to work on.
- **Raise the ceilings for long-form.** A run that produces ~18k words across 5 agents needs minute-scale, not second-scale, board windows. Make these configurable per run class (short answer vs. long-form manuscript) rather than one global default.

### P1 — Don't race two writers on one continuous artifact

Splitting a single manuscript into "Part One" and "Part Two" written in parallel, with no lock and no handshake on the seam, guarantees the overlap/gap Spark had to repair by hand.

- **Prefer sequential hand-off for a single continuous artifact:** writer B starts from a *frozen, complete* Part One, not a still-growing one. Or:
- **Give each writer a hard, non-overlapping range and a single owner of the seam.** If B needs A's Chapters 6–7 to bridge, that's a dependency the scheduler should satisfy (block B on A's completion of that range), not something B resolves by writing a duplicate and later dropping it.
- **The Lead should honor a gap-closing clarify request** (or the scheduler should make it unnecessary). Denying "supply the missing Chapters Six and Seven" left the hole that reached synthesis.

### P1 — Run a real continuity/fact pass over the ASSEMBLED draft, and gate on it

Spark now sends a canon-continuity instruction, but a Flash model won't self-audit 18k words reliably, and in run 3 the Fact Checker never even saw the real draft.

- **Add an explicit QA stage after assembly** whose input is the *final merged artifact* (not intermediate pieces, not a reconstruction), charged to check a concrete canon: each character's age/fixed details, day-hour-place-and-whereabouts per scene, dates, sums (paid ≤ attempted), tallies ("four things" lists four), single alibi per suspect, and **no fact dated later than the scene that cites it**.
- **Make it a gate.** The run should not complete `done` while the QA stage reports unresolved canonical contradictions; loop a bounded fix pass.
- **Keep the QA output OUT of the deliverable.** It corrects the prose; it must not append a ledger/table to the story (Spark's readers complained the four-role recap read like compliance evidence).

### P2 — Model-specific hardening for DeepSeek v4.1 Flash

Flash is the right cost/speed choice, but treat it as unreliable at protocol and long context:

- **Retry tool calls on malformed output** (truncated path, empty content, placeholder body) automatically; don't rely on the agent to notice.
- **Keep per-agent context small and explicit.** Hand each agent exactly the inputs it needs as attached content, not "go read the board" — the "Read tool is blocked; I have what I need" moment shows an agent proceeding blind when a fetch failed.
- **Deterministic where possible.** Concatenation, seam-detection, canon tables, and existence checks should be code, not model judgment. Reserve the model for prose, not bookkeeping.
- **Prefer a slightly stronger model for the two roles that carry the whole run: the Lead and the Reporter/synthesis.** If tiers allow, running just those on a stronger tier while keeping the specialists on Flash would likely fix the "denied the fix" and "described instead of assembled" failures at low marginal cost. (Spark already forwards a per-slot tier map, so this is a config choice, not a code change.)

---

## 3. Suggested config knobs (so Spark/operators can tune per run class)

- `board.wait.default_ms`, and a `long_form` profile that multiplies it (target: minutes, not seconds).
- `board.wait.mode = heartbeat` — extend while the producer is alive; expire only on stall.
- `synthesis.require_inputs_present = true` — block synthesis/QA until declared inputs land.
- `synthesis.emit = concat_then_smooth` for piece-based deliverables (vs. `rewrite`).
- `vfs.write.verify_readback = true`, `vfs.write.reject_placeholder = true`.
- `run.complete.require_deliverable_file = true` — fail completion if the named deliverable file is absent/empty.
- `qa.canon_pass = true` and `qa.gate_on_contradictions = true`.

---

## 4. Acceptance test (reproduce + verify the fix)

Re-run the same fiction brief on Flash and assert, without any hand-repair:

1. Every agent write round-trips (readback == sent); no file is a placeholder stub; no path is truncated.
2. The declared parts assemble into **one** deliverable file that actually exists in the VFS; the run's `truth` is that file's content (or a handle that resolves to it), not a note about it.
3. No duplicated scene and no missing chapter across the seam.
4. The QA/canon pass ran against the final merged draft and the run did not complete with an unresolved age/date/sum/tally/alibi/future-fact contradiction.
5. Board waits did not expire on any producer that was still alive and working.

If all five hold on Flash, the pipeline is robust to the weakest model we run on — which is the bar that matters.

---

*Spark-side companion changes already shipped: delivery contract on every template + freeform + continuation intent; silent continuity-pass instruction; mystery-thriller victim-motivation / single-alibi / on-page-source-for-precise-figures / economical role-resolution. These reduce the load on CC's structural fixes but do not replace them.*
