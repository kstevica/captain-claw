"""In-turn context governor for long single-turn workers (Vatra).

A Vatra worker does its whole piece inside ONE turn of up to hundreds of tool
calls; session compaction only runs between turns, so the per-call packer in
``_build_messages`` is the only thing keeping the request within the tier's
window. Measured on the glasses deck it under-counted 1.4-2x — replayed
reasoning was never counted for an OpenAI-compatible thinking model, and
cl100k reads number-heavy text low — and once over budget it dropped one
message per call, so the request prefix moved on every call and the server's
prompt cache missed every time.

With the governor on (see :func:`captain_claw.context_spill.governor_enabled`):

* **Real tokens.** Replayed reasoning is counted, and the budget is divided by
  a ratio calibrated from what the provider reports it actually received.
* **Reasoning echo.** Only the last few assistant steps carry their reasoning
  back to the model (providers that demand it on every step keep it all).
* **Cut once and hold.** When the turn's chain no longer fits, its oldest steps
  are cut in one block down to ~60% of the budget, on an assistant-step
  boundary, and the cut point is kept so the following calls only append —
  one cache miss per cut instead of one per call.
* **Learned window / overflow recovery.** A context-overflow error teaches the
  agent the server's real window and forces a re-cut instead of ending the turn.
* **Fold (phase 2).** Before a call whose real input would pass ~80% of the
  window, older tool results are folded — in one batch, oldest and already
  noted first — into stable stubs (a short head + the file holding the full
  text), down to ~50%. At ~70% a live note asks the worker to put what it still
  needs from those results into its notes file first; a notes write or a
  `vatra post` that cites a source (its URL or file) makes it safe to fold
  first. The newest step's results — not yet seen by the model — fold only
  when the request would not fit otherwise.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from typing import Any

from captain_claw.logging import get_logger

log = get_logger(__name__)


class AgentContextGovernorMixin:
    """Calibration, reasoning echo and cut-and-hold packing (see module doc)."""

    _REASONING_ECHO_STEPS = 3
    _REASONING_ECHO_SHARE = 0.25   # echoed reasoning past this share of history folds to markers
    _GOVERNOR_LOW = 0.60       # a cut keeps the history at ~60% of its budget
    _PRESEND_LIMIT = 0.97      # re-cut before sending past 97% of the real window
    _CALIBRATION_MIN = 1.0
    _CALIBRATION_MAX = 2.0
    _CALIBRATION_ALPHA = 0.5
    _CALIBRATION_STEP = 0.05   # the ratio moves the budget only in 5% steps
    _CALIBRATION_IMAGE_SHARE = 0.25  # calls with more image tokens than this don't calibrate
    _FOLD_HIGH = 0.80          # fold when the real input would pass 80% of the window …
    _FOLD_LOW = 0.50           # … down to 50%
    _NUDGE_AT = 0.70           # ask for notes before the fold comes
    _FOLD_KEEP_RESULTS = 2     # the newest results are never folded
    _FOLD_MIN_CHARS = 1_500
    _FOLD_HEAD_CHARS = 300
    _FOLD_MIN_GAIN = 0.05      # a fold must free ≥5% of the window, or it waits
    _NOTE_INLINE_CHARS = 4_000  # a notes write this short stays readable in context

    # ── switches ─────────────────────────────────────────────────────

    def _governor_on(self) -> bool:
        from captain_claw.context_spill import governor_enabled

        return governor_enabled()

    def _reasoning_replay_required(self) -> bool:
        """Providers that need every step's reasoning back (DeepSeek's thinking
        mode, Gemini's thought parts) — the same set ``_replays_reasoning``
        already counts in full."""
        return bool(self._replays_reasoning())

    # ── calibration ──────────────────────────────────────────────────

    def _ctx_model_key(self) -> str:
        provider = getattr(self, "provider", None)
        return f"{getattr(provider, 'provider', '')}/{getattr(provider, 'model', '')}"

    def _stored_calibration(self) -> float | None:
        """The ratio measured for the CURRENT model, if any (a ratio or a learned
        window from another model would mis-size this one)."""
        key = self._ctx_model_key()
        if getattr(self, "_ctx_calibration_model", None) == key:
            value = getattr(self, "_ctx_calibration", None)
            if isinstance(value, (int, float)):
                return float(value)
        meta = getattr(getattr(self, "session", None), "metadata", None)
        stored = meta.get("ctx_calibration") if isinstance(meta, dict) else None
        if isinstance(stored, dict) and stored.get("model") == key and isinstance(stored.get("ratio"), (int, float)):
            return float(stored["ratio"])
        return None

    def _ctx_ratio(self) -> float:
        """Real/estimated token ratio for this agent's model, stepped (>= 1.0)."""
        value = self._stored_calibration()
        value = 1.0 if value is None else value
        value = min(self._CALIBRATION_MAX, max(self._CALIBRATION_MIN, value))
        step = self._CALIBRATION_STEP
        return round(math.ceil(value / step - 1e-9) * step, 2)

    def _set_calibration(self, value: float) -> None:
        key = self._ctx_model_key()
        self._ctx_calibration = round(min(self._CALIBRATION_MAX, max(self._CALIBRATION_MIN, value)), 4)
        self._ctx_calibration_model = key
        meta = getattr(getattr(self, "session", None), "metadata", None)
        if isinstance(meta, dict):
            meta["ctx_calibration"] = {"ratio": self._ctx_calibration, "model": key}

    def _observe_ctx_calibration(self, estimated: int, provider_input: int) -> None:
        """Fold one main-turn call's (estimated, provider-reported) input into
        the ratio. Samples a server can't have meant (reported input far below
        the estimate — cache not reported, or a different request) are skipped."""
        if estimated < 2_000 or provider_input <= 0:
            return
        # Image tokens are an estimate that differs per model: a call they
        # dominate says little about the text ratio.
        _window = getattr(self, "last_context_window", None) or {}
        if int(_window.get("image_tokens", 0) or 0) > self._CALIBRATION_IMAGE_SHARE * estimated:
            return
        sample = provider_input / estimated
        if not (0.5 <= sample <= 3.0):
            return
        sample = min(self._CALIBRATION_MAX, max(self._CALIBRATION_MIN, sample))
        prev = self._stored_calibration()
        prev = sample if prev is None else prev
        old_step = self._ctx_ratio()
        self._set_calibration(prev + self._CALIBRATION_ALPHA * (sample - prev))
        new_step = self._ctx_ratio()
        if new_step != old_step:
            log.info("Context calibration changed", ratio=new_step, sample=round(sample, 3),
                     estimated=estimated, provider_input=provider_input)
            self._emit_governor_event("calibrated", ratio=new_step, sample=round(sample, 3),
                                      estimated=estimated, provider_input=provider_input)

    def _bump_ctx_calibration(self, factor: float = 1.15) -> None:
        """An overflow the estimate didn't see coming, with no numbers to learn
        from: assume the count runs that much low."""
        self._set_calibration(max(self._ctx_ratio(), 1.0) * factor)
        self._emit_governor_event("calibration_bumped", ratio=self._ctx_ratio())

    def _learned_window(self) -> int:
        """A server window learned from an overflow — for the current model only."""
        if getattr(self, "_learned_ctx_window_model", None) != self._ctx_model_key():
            return 0
        return int(getattr(self, "_learned_ctx_window", 0) or 0)

    def _learn_ctx_window(self, capacity: int, *, requested: int | None = None,
                          prompt_tokens: int | None = None, prompt_floor: int = 0) -> None:
        """Remember the server's real window from a context-overflow error.

        Only a modest output reserve comes off it — the output-budget heal
        already shrinks the output side per request — and the window never drops
        below half the capacity, so one overflow can't starve the worker of its
        own history. The error's prompt count (or its lower bound) calibrates."""
        if capacity <= 0:
            return
        try:
            out = int(getattr(getattr(self, "provider", None), "max_tokens", 0) or 0)
        except (TypeError, ValueError):
            out = 0
        reserve = min(x for x in (out or capacity // 4, requested or capacity // 4, capacity // 4) if x > 0)
        window = max(int(capacity) // 2, int(capacity) - reserve - 1_024)
        current = self._learned_window()
        if not current or window < current:
            self._learned_ctx_window = window
            self._learned_ctx_window_model = self._ctx_model_key()
        est = int((getattr(self, "last_context_window", {}) or {}).get("estimated_input_tokens", 0) or 0)
        seen = int(prompt_tokens or prompt_floor or 0)
        if est and seen > est:
            self._observe_ctx_calibration(est, seen)
        log.warning("Learned context window from an overflow", capacity=capacity,
                    window=self._learned_window(), prompt_tokens=prompt_tokens or prompt_floor)
        self._emit_governor_event("window_learned", capacity=capacity, window=self._learned_window())

    # ── per-turn window state ────────────────────────────────────────

    def _reset_turn_window(self) -> None:
        """Forget the turn's cut point (new turn, or the session was rewritten)."""
        self._turn_window = None
        self._turn_force_recut = False
        self._fold_nudged = False
        self._context_nudge_text = ""
        self._nudge_one_shot = False
        self._last_build_msg_count = None

    def _emit_governor_event(self, step: str, **fields: Any) -> None:
        """One monitor row (hidden from the model) so the run log shows it."""
        emit = getattr(self, "_emit_tool_output", None)
        if not callable(emit):
            return
        try:
            text = "\n".join([f"step={step}"] + [f"{k}={v}" for k, v in fields.items()])
            emit("context_governor", {"step": step, **fields}, text)
        except Exception:  # noqa: BLE001 — tracing never breaks a call
            pass

    # ── cut and hold ─────────────────────────────────────────────────

    @staticmethod
    def _chain_groups(candidate_messages: list[dict[str, Any]], chain: list[int]) -> list[list[int]]:
        """Split the chain into steps: an assistant message and everything after
        it up to the next assistant message (its tool results, any notices).
        A leading run with no assistant (results whose call is gone) is its own
        step. Cutting whole steps never orphans a tool result."""
        groups: list[list[int]] = []
        for pos in chain:
            role = str(candidate_messages[pos].get("role", "")).strip().lower()
            if role == "assistant" or not groups:
                groups.append([pos])
            else:
                groups[-1].append(pos)
        return groups

    def _governed_fill(
        self,
        candidate_messages: list[dict[str, Any]],
        candidate_sidx: list[int | None],
        chain_start_pos: int,
        selected: set[int],
        used_tokens: int,
        history_budget: int,
        background_pos: int | None,
        cost: Callable[[int], int],
    ) -> tuple[int, int]:
        """Admit the turn's chain and earlier history for one call, cut-and-hold
        style; adds to *selected* and returns ``(used_tokens, dropped)``."""
        n = len(candidate_messages)
        dropped = 0

        def _free(pos: int) -> bool:
            return pos not in selected and pos != background_pos

        # Live notes (newest positions) first, as the plain packer does.
        for pos in range(n - 1, -1, -1):
            if _free(pos) and "_live_notes" in candidate_messages[pos]:
                c = cost(pos)
                if used_tokens + c <= history_budget:
                    selected.add(pos)
                    used_tokens += c
                else:
                    dropped += 1
        chain = [p for p in range(chain_start_pos, n)
                 if _free(p) and "_live_notes" not in candidate_messages[p]]
        prior = [p for p in range(0, chain_start_pos)
                 if _free(p) and "_live_notes" not in candidate_messages[p]]
        avail = max(0, history_budget - used_tokens)
        low = max(0, int(self._GOVERNOR_LOW * history_budget) - used_tokens)
        window = getattr(self, "_turn_window", None)
        if not isinstance(window, dict):
            window = {"floor": None, "prior": None}
        force = bool(getattr(self, "_turn_force_recut", False))
        self._turn_force_recut = False

        def _sidx_set(positions: list[int]) -> frozenset:
            return frozenset(candidate_sidx[p] for p in positions if candidate_sidx[p] is not None)

        def _prior_fit(room: int) -> list[int]:
            # Newest-first, stopping at the first that doesn't fit: the kept
            # earlier history is one contiguous tail, so it stays a stable
            # prefix rather than a size-dependent sample.
            keep: list[int] = []
            for p in reversed(prior):
                c = cost(p)
                if c > room:
                    break
                keep.append(p)
                room -= c
            return keep

        def _held_prior() -> list[int]:
            held = window.get("prior")
            return [p for p in prior if candidate_sidx[p] in held] if held is not None else []

        def _sum(ps: list[int]) -> int:
            return sum(cost(p) for p in ps)

        if not force:
            start_i = 0
            if window.get("floor") is not None:
                start_i = next(
                    (i for i, p in enumerate(chain)
                     if candidate_sidx[p] is not None and candidate_sidx[p] >= window["floor"]),
                    len(chain),
                )
            kept_chain = chain[start_i:]
            chain_cost = _sum(kept_chain)
            if window.get("prior") is None:
                # Earlier turns sit BEFORE the chain in the request, so they are
                # decided once per turn (and kept across cuts when they fit) —
                # never re-sampled per call. Within the low-water room only, so
                # the chain keeps its headroom.
                prior_kept = _prior_fit(max(0, low - chain_cost))
            else:
                prior_kept = _held_prior()
            if chain_cost + _sum(prior_kept) <= avail:
                window["prior"] = _sidx_set(prior_kept)
                self._turn_window = window
                selected.update(kept_chain)
                selected.update(prior_kept)
                return (used_tokens + chain_cost + _sum(prior_kept),
                        dropped + start_i + len(prior) - len(prior_kept))

        # Cut: keep the newest whole steps up to the low-water mark, in one block.
        groups = self._chain_groups(candidate_messages, chain)
        prior_kept = _held_prior()
        if _sum(prior_kept) > low // 2:
            prior_kept = []          # earlier history gives way before the chain does
        target = max(0, low - _sum(prior_kept))
        room = max(0, avail - _sum(prior_kept))
        kept_groups: list[list[int]] = []
        acc = 0
        for group in reversed(groups):
            c = _sum(group)
            if acc + c > room:
                if not kept_groups:
                    continue         # a step bigger than the whole room: skip it, keep older ones
                break
            if kept_groups and acc + c > target:
                break
            kept_groups.insert(0, group)
            acc += c
        kept_chain = [p for g in kept_groups for p in g]
        first_kept = kept_chain[0] if kept_chain else None
        start_i = chain.index(first_kept) if first_kept is not None else len(chain)
        if first_kept is not None and candidate_sidx[first_kept] is not None:
            floor = candidate_sidx[first_kept]
        else:
            last = [candidate_sidx[p] for p in chain if candidate_sidx[p] is not None]
            floor = (max(last) + 1) if last else None
        echo_floor = self._echo_floor_after_cut(candidate_messages, candidate_sidx, kept_chain)
        self._turn_window = {"floor": floor, "prior": _sidx_set(prior_kept), "echo_floor": echo_floor}
        self._turn_window_cuts = int(getattr(self, "_turn_window_cuts", 0) or 0) + 1
        before = used_tokens + _sum(chain) + _sum(prior)
        after = used_tokens + acc + _sum(prior_kept)
        log.info("In-turn context cut", dropped_messages=len(chain) - len(kept_chain),
                 kept_messages=len(kept_chain), before=before, after=after,
                 budget=history_budget, forced=force)
        self._emit_governor_event("cut", dropped_messages=len(chain) - len(kept_chain),
                                  kept_messages=len(kept_chain), history_before=before,
                                  history_after=after, history_budget=history_budget,
                                  ratio=self._ctx_ratio())
        selected.update(kept_chain)
        selected.update(prior_kept)
        return after, dropped + (len(chain) - len(kept_chain)) + len(prior) - len(prior_kept)

    # ── reasoning echo ───────────────────────────────────────────────

    def _echo_floor_after_cut(self, candidate_messages: list[dict[str, Any]],
                              candidate_sidx: list[int | None], kept_chain: list[int]) -> int | None:
        """Session index from which kept steps still carry their full reasoning
        after a cut: the last few reasoning steps."""
        steps = [p for p in kept_chain
                 if str(candidate_messages[p].get("role", "")).strip().lower() == "assistant"
                 and candidate_messages[p].get("reasoning_content")]
        if not steps:
            return None
        keep = steps[-self._REASONING_ECHO_STEPS:]
        return candidate_sidx[keep[0]]

    def _echo_floor(self) -> int | None:
        window = getattr(self, "_turn_window", None)
        return window.get("echo_floor") if isinstance(window, dict) else None

    # ── fold (phase 2) ───────────────────────────────────────────────

    def _notes_path(self) -> str:
        """Where this worker keeps its running notes."""
        from captain_claw.context_spill import notes_path

        return notes_path()

    def _is_notes_path(self, path: Any) -> bool:
        """This worker's notes file (outside Vatra: a plain ``notes.md``), or any
        Vatra notes file of the governed shape."""
        from captain_claw.context_spill import is_governor_notes_path

        p = str(path or "").strip().replace("\\", "/").lower()
        if not p:
            return False
        own = self._notes_path().lower()
        return (p == own or is_governor_notes_path(p)
                or (own == "notes.md" and p.rsplit("/", 1)[-1] == "notes.md"))

    def _is_receipt(self, msg: dict[str, Any]) -> bool:
        """A successful notes write/edit or `vatra post` — what the sources it
        cites were written down in, so they fold first. Never folded itself."""
        if str(msg.get("role", "")) != "tool":
            return False
        if str(msg.get("content", "") or "").lstrip().lower().startswith("error"):
            return False
        name = str(msg.get("tool_name", "") or "").lower()
        args = msg.get("tool_arguments") or {}
        if name in ("write", "edit"):
            return self._is_notes_path(args.get("path"))
        return name == "vatra" and str(args.get("action", "")).lower() == "post"

    def _turn_real_projection(self) -> tuple[int, int]:
        """(projected real input of the next call, real window): the last build's
        full estimate plus what was appended since, times the ratio."""
        from captain_claw.context_spill import real_context_budget

        budget = real_context_budget(self)
        window = getattr(self, "last_context_window", None) or {}
        last_count = getattr(self, "_last_build_msg_count", None)
        if (not budget or not isinstance(last_count, int) or not self.session
                or last_count > len(self.session.messages)):
            return 0, budget
        est = int(window.get("estimated_input_tokens", 0) or 0)
        replays = self._replays_reasoning()
        added = 0
        from captain_claw.config import get_config
        from captain_claw.llm.vision import tool_image_count, typical_image_tokens

        _image_cost = 0
        for msg in self.session.messages[last_count:]:
            added += self._wire_token_count(msg)   # counts reasoning when it is replayed
            if str(msg.get("role", "")) == "tool" and "[Tool image:" in str(msg.get("content", "")):
                if not _image_cost:
                    _image_cost = typical_image_tokens(
                        get_config().context.inline_images.max_pixels,
                        str(getattr(self.provider, "provider", "") or ""),
                        str(getattr(self.provider, "model", "") or ""),
                    )
                added += _image_cost * tool_image_count(str(msg.get("content", "")))
            if (not replays and str(msg.get("role", "")) == "assistant"
                    and msg.get("reasoning_content")):
                added += self._ensure_reasoning_token_count(msg)
        return int((est + added) * self._ctx_ratio()), budget

    @staticmethod
    def _fold_key(tool: str, args: dict[str, Any]) -> str:
        """What a stub names: the URL / path / query, else a short command."""
        for field in ("url", "path", "query", "file_id"):
            value = args.get(field)
            if isinstance(value, str) and value.strip():
                return value.strip()[:300]
        urls = args.get("urls")
        if isinstance(urls, list) and urls:
            return f"batch of {len(urls)} URLs"
        command = args.get("command")
        if isinstance(command, str) and command.strip():
            return " ".join(command.split())[:80]
        return tool

    def _fold_stub(self, msg: dict[str, Any]) -> str | None:
        """Stable stub for a folded tool result: tool, key, size, the file with the
        full text (saved now if the result wasn't spilled already), a short head.
        None when the full text could not be put on disk — then it isn't folded."""
        from captain_claw import context_spill

        content = str(msg.get("content", "") or "")
        tool = str(msg.get("tool_name", "") or "tool")
        args = msg.get("tool_arguments") or {}
        key = self._fold_key(tool, args if isinstance(args, dict) else {})
        first, _, rest = content.partition("\n")
        found = re.search(re.escape(context_spill.POINTER_MARK) + r"(\S+)", first)
        if found and context_spill.is_governed_output(content):
            pointer = found.group(1).rstrip("]).,")
            body = rest.lstrip("\n")
        else:
            # One file per result: two shell outputs, two pages of one file or
            # two batches must not overwrite each other's saved text.
            # message_id first — some providers reuse call ids (embedded_0, call_0).
            uniq = str(msg.get("message_id") or msg.get("tool_call_id") or "")
            pointer = context_spill.save_source(
                self, f"folded-{tool}", f"{key}#{uniq}" if uniq else key, content, shared=False)
            body = content
        if not pointer:
            return None
        head = " ".join(body[: self._FOLD_HEAD_CHARS * 2].split())[: self._FOLD_HEAD_CHARS]
        return (f"[folded {tool} {key} · {len(content)} chars · {context_spill.POINTER_MARK}"
                f"{pointer} — read it with `read` (offset/limit) or `grep`]\n{head} …")

    def _nudge_text(self, *, folded: bool) -> str:
        import os

        in_vatra = bool(os.environ.get("CLAW_VATRA_SESSION", "").strip())
        share = " or `vatra post` it" if in_vatra else ""
        final = " Write your final part from your notes." if in_vatra else ""
        if folded:
            return ("[Context folded] Some tool results were just folded into short file "
                    "pointers (the full text stays on disk — `read`/`grep` it if needed). Keep "
                    f"appending what you use to {self._notes_path()}{share} (one short cited block "
                    f"per source, numbers verbatim).{final}")
        return ("[Context is filling up] In a few steps your older tool results will be folded "
                "into short file pointers (the full text stays on disk — `read`/`grep` it if "
                f"needed). Before that, append what you still need from them to {self._notes_path()}"
                f"{share} (one short cited block per source, numbers verbatim).{final}")

    def _maybe_fold_turn_context(self, turn_start_idx: int) -> int:
        """Fold older tool results of this turn when the next call would pass the
        high-water mark; return how many were folded."""
        if not self.session:
            return 0
        if getattr(self, "_nudge_one_shot", False):
            self._nudge_one_shot = False
            self._context_nudge_text = ""
        real, budget = self._turn_real_projection()
        if not real or not budget:
            return 0
        was_nudged = bool(getattr(self, "_fold_nudged", False))
        if real > self._NUDGE_AT * budget and not was_nudged:
            self._fold_nudged = True
            self._context_nudge_text = self._nudge_text(folded=False)
            self._emit_governor_event("nudge", real_tokens=real, budget=budget)
        if real <= self._FOLD_HIGH * budget:
            return 0
        from captain_claw.msg_origin import is_model_hidden_tool

        messages = self.session.messages
        window = getattr(self, "_turn_window", None) or {}
        floor = window.get("floor") if isinstance(window, dict) else None
        start = max(turn_start_idx, floor if isinstance(floor, int) else turn_start_idx)

        def _is_step(i: int) -> bool:
            return str(messages[i].get("role", "")) == "assistant"

        results = [i for i in range(start, len(messages))
                   if str(messages[i].get("role", "")) == "tool" and not is_model_hidden_tool(messages[i])]
        # The newest step's results reach the model with this call: not seen yet.
        last_step = max((i for i in range(start, len(messages)) if _is_step(i)), default=-1)
        keep = set(results[-self._FOLD_KEEP_RESULTS:])
        # A source is noted when a later step's notes write / post cites it
        # (the notes directive asks for the URL or file in each block).
        receipt_texts: list[tuple[int, str]] = []
        for r in results:
            if self._is_receipt(messages[r]):
                rargs = messages[r].get("tool_arguments") or {}
                text = " ".join(str(rargs.get(k) or "") for k in ("content", "text", "new_string", "title"))
                step = max((j for j in range(start, r) if _is_step(j)), default=-1)
                receipt_texts.append((step, text.lower()))

        def _noted(i: int) -> bool:
            args = messages[i].get("tool_arguments") or {}
            if not isinstance(args, dict) or not receipt_texts:
                return False
            keys = [str(args.get(k) or "").strip().lower() for k in ("url", "path", "file_id")]
            urls = args.get("urls")
            if isinstance(urls, list):
                keys += [str(u).strip().lower() for u in urls]
            keys = [k.rstrip("/") for k in keys if len(k.rstrip("/")) >= 4]
            # Whole citation only: https://ex.com must not match a note citing
            # https://ex.com/pricing; a trailing slash or sentence period is fine.
            return any(step > i and re.search(re.escape(key) + r"/?(?![\w/%?#=&-]|\.\w)", text)
                       for step, text in receipt_texts for key in keys)

        def _eligible(i: int) -> bool:
            content = str(messages[i].get("content", "") or "")
            return (i not in keep and not messages[i].get("_folded")
                    and not content.startswith("[folded ")
                    and not self._is_receipt(messages[i])
                    and len(content) >= self._FOLD_MIN_CHARS)

        seen = sorted((i for i in results if i < last_step and _eligible(i)),
                      key=lambda i: (not _noted(i), i))   # noted first, then oldest
        unseen = [i for i in results if i > last_step and _eligible(i)]
        ratio = self._ctx_ratio()
        # A fold rewrites history (one cache miss): only when it frees enough —
        # unless the request would not fit at all.
        potential = sum(max(0, self._ensure_message_token_count(messages[i]) - 120) for i in seen)
        if (potential * ratio < self._FOLD_MIN_GAIN * budget
                and real <= self._PRESEND_LIMIT * budget):
            seen = []
        saved = 0
        folded = 0
        failed = False
        newly: list[dict[str, Any]] = []

        def _fold(i: int) -> bool:
            nonlocal saved, folded, failed
            msg = messages[i]
            stub = self._fold_stub(msg)
            if stub is None:
                failed = True       # disk trouble: keep the text, leave it to the cut
                return False
            before = self._ensure_message_token_count(msg)
            msg["content"] = stub
            msg["_folded"] = True
            saved += max(0, before - self._recount_message_tokens(msg))
            folded += 1
            newly.append(msg)
            return True

        need = (real - self._FOLD_LOW * budget) / ratio
        for i in seen:
            if saved >= need or not _fold(i):
                break
        seen_folded = folded
        # Results the model hasn't read are folded only when the request would
        # not fit otherwise.
        if not failed:
            for i in unseen:
                if real - saved * ratio <= self._PRESEND_LIMIT * budget or not _fold(i):
                    break
        if not folded:
            return 0
        # A fold of read results already changes earlier messages: move the
        # reasoning echo floor in the same batch (unread ones sit past the
        # cached prefix — don't break it for them).
        steps = [i for i in range(start, len(messages))
                 if _is_step(i) and messages[i].get("reasoning_content")]
        if steps and seen_folded:
            win = getattr(self, "_turn_window", None)
            if not isinstance(win, dict):
                win = {"floor": None, "prior": None}
            win["echo_floor"] = steps[-self._REASONING_ECHO_STEPS:][0]
            self._turn_window = win
        self._drop_folded_read_guards(newly)
        after_real = max(0, real - int(saved * ratio))
        self._fold_nudged = False
        # The model never saw the pre-fold note: tell it once, now.
        self._context_nudge_text = "" if was_nudged else self._nudge_text(folded=True)
        self._nudge_one_shot = not was_nudged
        self._fold_count = int(getattr(self, "_fold_count", 0) or 0) + 1
        log.info("In-turn context fold", folded=folded, saved_tokens=saved,
                 real_before=real, real_after=after_real, budget=budget)
        self._emit_governor_event("fold", folded_results=folded, saved_tokens=saved,
                                  real_before=real, real_after=after_real, budget=budget)
        return folded

    def _drop_folded_read_guards(self, folded: list[dict[str, Any]]) -> None:
        """A stub tells the model to `read` the saved copy: the per-turn duplicate
        guard must not refuse that read as "content has not changed"."""
        import os

        counts = getattr(self, "_turn_tool_call_counts", None)
        if not isinstance(counts, dict) or not counts:
            return
        from captain_claw import context_spill

        for msg in folded:
            first = str(msg.get("content", "") or "").partition("\n")[0]
            found = re.search(re.escape(context_spill.POINTER_MARK) + r"(\S+)", first)
            paths = [found.group(1).rstrip("]).,")] if found else []
            args = msg.get("tool_arguments") or {}
            if isinstance(args, dict) and isinstance(args.get("path"), str):
                paths.append(args["path"].strip())
            for path in paths:
                if not path:
                    continue
                base = f"read|path={os.path.abspath(path)}"
                # Any section of it: `|offset=…`, `|limit=…` (loop_guards.range_signature).
                for sig in [k for k in counts if k == base or k.startswith(base + "|")]:
                    counts.pop(sig, None)

    def _context_pressure_note(self) -> str:
        """The pending pre-fold notes nudge (a live note), if any."""
        return str(getattr(self, "_context_nudge_text", "") or "")
