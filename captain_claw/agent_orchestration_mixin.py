"""Main request orchestration (complete/stream) for Agent.

This mixin contains the core ``complete()`` and ``stream()`` methods that
drive the agent's turn-level loop.  Scale detection, advisory injection,
deferred init, and micro-loop summary building have been moved to
``AgentScaleDetectionMixin``.  Completion gate, finalization, and coverage
validation have been moved to ``AgentCompletionMixin``.
"""

import asyncio
import json
from typing import Any, AsyncIterator

from captain_claw.agent_scale_detection_mixin import _build_scale_advisory
from captain_claw.agent_stuck import (
    MSG_BUDGET_EXHAUSTED,
    MSG_RETRIES_EXHAUSTED,
    MSG_STUCK,
)
from captain_claw.config import get_config
from captain_claw.exceptions import GuardBlockedError, LLMAPIError, LLMError
from captain_claw.llm import Message
from captain_claw.logging import get_logger


log = get_logger(__name__)


# Phrases signalling the model emitted its REASONING / decision *about*
# answering instead of the answer itself — a "preamble" it should continue
# past. Real answers rarely open by narrating "the user is asking…" or stack
# multiple of these decision statements, so detection stays high-precision.
_PREAMBLE_META_STARTS = (
    "the user is asking", "the user wants", "the user is requesting",
    "the user asked", "the user's question", "the user would like",
    "the user is looking", "the user needs",
)
_PREAMBLE_DECISION_MARKERS = (
    "i can answer this", "i can answer that", "i'll answer", "i will answer",
    "let me answer", "i can provide this from", "from my existing knowledge",
    "without needing to do additional", "without additional research",
    "no need to search", "don't need to search", "do not need to search",
    "this is a straightforward knowledge question", "this is a simple question",
    "i can help with this directly", "i can answer from my", "answer this directly",
)


def _looks_like_planning_preamble(text: str) -> bool:
    """True if a short text-only response is the model's reasoning/decision
    *about* answering rather than the answer itself (e.g. "The user is asking…
    I can answer this from my knowledge without searching."). Conservative:
    only a meta opener, or ≥2 decision markers in a short message."""
    t = (text or "").strip()
    if not t or len(t) > 700:
        return False
    low = t.lower()
    if low.startswith(_PREAMBLE_META_STARTS):
        return True
    n = sum(1 for m in _PREAMBLE_DECISION_MARKERS if m in low)
    return n >= 2 and len(t) < 400


def _strip_planning_preamble(text: str) -> str:
    """Drop a leading reasoning/meta preamble, returning the substantive
    remainder (or '' if the whole message was preamble). Lets us ship the
    real answer hiding behind a "The user is asking… I can answer this…"
    opener instead of nudging the model and burning an iteration."""
    t = (text or "").strip()
    if not t:
        return ""
    parts = _re.split(r"(?<=[.!?])\s+", t)
    kept: list[str] = []
    dropping = True
    for p in parts:
        low = p.strip().lower()
        if dropping and (
            low.startswith(_PREAMBLE_META_STARTS)
            or any(m in low for m in _PREAMBLE_DECISION_MARKERS)
        ):
            continue
        dropping = False
        kept.append(p)
    return " ".join(kept).strip()


def _looks_like_leaked_plan_json(text: str) -> bool:
    """True if the final response is raw internal planning/contract JSON that
    leaked as the user-facing answer. Weak models sometimes echo the task
    contract ({"summary":…, "tasks":[…], "requirements":[…]}) instead of
    actually answering. We catch it and nudge for a real reply."""
    t = (text or "").strip()
    if not t:
        return False
    # Tolerate a ```json fence.
    if t.startswith("```"):
        t = t.strip("`").lstrip()
        if t[:4].lower() == "json":
            t = t[4:].lstrip()
    if not t.startswith("{"):
        return False
    try:
        obj = json.loads(t)
    except Exception:
        return False
    if not isinstance(obj, dict):
        return False
    keys = {str(k).lower() for k in obj.keys()}
    # Contract/plan shape: a task list plus requirements/summary/etc.
    return "tasks" in keys and bool(
        keys & {"requirements", "summary", "prefetch_urls", "checks"}
    )

# Tools that bring in new external / document content and may warrant
# deferred scale re-extraction.  Lightweight tools like datastore, glob,
# shell, todo do NOT qualify — they return structured local data that
# should never trigger the research micro-loop.
_CONTENT_FETCH_TOOLS: frozenset[str] = frozenset({
    "web_fetch", "web_get", "read",
    "pdf_extract", "docx_extract", "pptx_extract",
    "xlsx_extract", "pocket_tts",
})

# ── Eco-mode: core tools always sent with full definitions ──
# Google tools are listed here so they're always offered to the LLM when
# present, but they only reach ``tool_defs`` after passing the
# ``requires_google`` gate in the registry — so when Google OAuth is
# disconnected they're filtered out upstream and never kept regardless.
_ECO_CORE_TOOLS: frozenset[str] = frozenset({
    "shell", "read", "write", "edit", "glob", "grep",
    "web_fetch", "web_search",
    "google_mail", "google_drive", "google_calendar",
    "whatsapp_send_file", "intentions",
    # Scheduling: cron self-describes as a frozen/always-available tool, and
    # scheduling requests arrive in any language ("za 5 minuta", "in 5 min").
    # The English-only intent patterns can't catch those, so keep cron in core
    # — otherwise eco strips its schema and the agent can't defer, it just acts
    # immediately. (See cron_tool.py "Frozen tool (always available)".)
    "cron",
    # Vision: always offer image_vision/video_vision so attached media can be
    # analysed regardless of the caption's language (eco patterns are English).
    "image_vision", "video_vision",
    # Fleet awareness: always let an agent see/consult/delegate to peers.
    # ``flight_deck`` self-discovers peers via /fd/fleet, so it works
    # whenever FD_URL is set and a peer is running. Without this, eco mode
    # hides it and the agent wrongly reports it "can't reach other agents".
    "flight_deck",
})

# ── Nano-mode: barebone tool allowlist for tiny local models. ──
# The agent should solve complex tasks by writing scripts and running
# them via shell rather than driving an LLM-side reasoning chain over
# many specialized tools.  Anything outside this set is hidden from
# the LLM (definitions stripped) when nano mode is active.
_NANO_TOOLS: frozenset[str] = frozenset({
    "shell", "read", "write", "edit", "glob", "grep",
    "web_fetch", "web_search",
    "pdf_extract", "docx_extract", "xlsx_extract", "pptx_extract",
    "datastore", "insights", "personality", "clipboard",
})

# ── Eco-mode: intent-based keyword → tool mapping ──
# When eco mode is active, only core tools get full definitions by default.
# These patterns add extra tools based on keywords detected in the user message.
import re as _re

_ECO_INTENT_PATTERNS: list[tuple[_re.Pattern[str], frozenset[str]]] = [
    # Web & URLs
    (_re.compile(r"https?://|fetch\b|scrape|crawl|web_get", _re.I),
     frozenset({"web_get", "web_fetch", "web_search", "browser"})),
    # Search
    (_re.compile(r"\bsearch\b|\blook\s?up\b|\bfind\s+(?:info|out|about)\b|\bgoogle\b", _re.I),
     frozenset({"web_search", "web_fetch"})),
    # Documents
    (_re.compile(r"\.pdf\b|pdf\b", _re.I),
     frozenset({"pdf_extract", "summarize_files"})),
    (_re.compile(r"\.docx?\b|word\s+doc", _re.I),
     frozenset({"docx_extract", "summarize_files"})),
    (_re.compile(r"\.xlsx?\b|spreadsheet|excel", _re.I),
     frozenset({"xlsx_extract", "summarize_files"})),
    (_re.compile(r"\.pptx?\b|presentation|powerpoint|slides", _re.I),
     frozenset({"pptx_extract", "summarize_files"})),
    (_re.compile(r"summar\w+\s+files|review\s+(?:all|the|these)\s+(?:files|documents|docs|folder)", _re.I),
     frozenset({"summarize_files", "pdf_extract", "docx_extract", "xlsx_extract", "pptx_extract"})),
    # Images
    (_re.compile(r"\bimage\b|\bphoto\b|\bpicture\b|\bscreenshot\b|\bdraw\b|\bgenerate\s+(?:an?\s+)?image", _re.I),
     frozenset({"image_gen", "image_ocr", "image_vision", "screen_capture"})),
    # Email
    (_re.compile(r"\bemail\b|\bmail\b|\bgmail\b|\bdraft\b|\bsend\s+(?:a\s+)?message\b", _re.I),
     frozenset({"send_mail", "google_mail", "gws"})),
    # Calendar
    (_re.compile(r"\bcalendar\b|\bschedule\b|\bmeeting\b|\bagenda\b|\bevent\b", _re.I),
     frozenset({"google_calendar", "gws"})),
    # NOTE: cron is intentionally NOT matched here — it lives in _ECO_CORE_TOOLS
    # so scheduling works in any language, not just the English cues above.
    # Google Drive
    (_re.compile(r"\bdrive\b|\bgoogle\s+d(oc|rive)\b|\bshared\s+folder\b", _re.I),
     frozenset({"google_drive", "gws"})),
    # Data & database
    (_re.compile(r"\bdatastore\b|\bdatabase\b|\btable\b|\bsql\b|\binsert\b|\bquery\b", _re.I),
     frozenset({"datastore", "direct_api"})),
    # Browser automation
    (_re.compile(r"\bbrowser\b|\blogin\b|\bclick\b|\bnavigate\b|\bweb\s+app\b", _re.I),
     frozenset({"browser"})),
    # Audio / TTS
    (_re.compile(r"\bspeak\b|\bsay\b|\baudio\b|\btts\b|\bvoice\b|\bread\s+aloud\b", _re.I),
     frozenset({"pocket_tts"})),
    # API
    (_re.compile(r"\bapi\b|\bendpoint\b|\brest\b|\bhttp\s+(?:get|post|put)\b", _re.I),
     frozenset({"direct_api"})),
    # Desktop
    (_re.compile(r"\bdesktop\b|\bscreen\b|\bclick\b.*\bbutton\b|\bopen\s+app\b", _re.I),
     frozenset({"desktop_action", "screen_capture"})),
    # Twitter / social
    (_re.compile(r"\btweet\b|\btwitter\b|\bpost\b.*\bsocial\b", _re.I),
     frozenset({"twitter"})),
    # Clipboard
    (_re.compile(r"\bclipboard\b|\bcopy\b|\bpaste\b", _re.I),
     frozenset({"clipboard"})),
    # Termux / Android
    (_re.compile(r"\btermux\b|\bandroid\b|\bphone\b|\btorch\b|\bbattery\b|\bgps\b", _re.I),
     frozenset({"termux"})),
]


def _eco_select_tools_by_intent(user_input: str) -> frozenset[str]:
    """Return tool names matched by intent patterns in the user message."""
    matched: set[str] = set()
    for pattern, tools in _ECO_INTENT_PATTERNS:
        if pattern.search(user_input):
            matched |= tools
    return frozenset(matched)


# ── Stall-detection heuristic ───────────────────────────────────────────
# Some smaller / weaker models reply with intent-only announcements
# ("Let me look that up…", "I'll fetch the file now.") and stop, instead
# of actually calling a tool or producing the deliverable. We detect that
# pattern on a no-tool-calls turn and silently retry with a corrective
# instruction (and force ``tool_choice="required"`` when tools are
# available so the retry cannot stall the same way).
_STALL_FIRST_LINE_RE = _re.compile(
    r"^\s*(?:"
    r"let me\b"
    r"|let's\b"
    r"|i'?ll\b"
    r"|i will\b"
    r"|i'?m going to\b"
    r"|i am going to\b"
    r"|i'?m about to\b"
    r"|proceeding\b"
    r"|starting\b"
    r"|working on\b"
    r"|one (?:moment|sec)\b"
    r"|(?:just )?a (?:moment|sec(?:ond)?)\b"
    r"|on it\b"
    r"|sure[,!.]?\s+(?:let me|i'?ll|i will|one|just)"
    r")",
    _re.IGNORECASE,
)
# Maximum number of silent stall retries per turn. Two is enough to
# convert most weak-model stalls into a useful turn without driving an
# infinite re-roll loop when the model genuinely has nothing to add.
MAX_STALL_RETRIES = 2
# Length budget for the stall check. Real stalls are terse single
# sentences ("Let me look that up.", ~25 chars). Anything longer is
# only treated as a stall if it has no substantive follow-through
# beyond the opening intent phrase (see ``_looks_like_stall``).
_STALL_MAX_LEN = 60
# Minimum substantive content (chars) AFTER the opening intent sentence
# required to disqualify a message from being a stall. Used for medium-
# length messages that open with a stall phrase but actually continue
# with a real answer (e.g. "Let me explain why X. First, …").
_STALL_FOLLOWTHROUGH_MIN_CHARS = 40


def _looks_like_stall(text: str) -> bool:
    """Return True when the assistant text reads as an intent-only stall.

    Mirrors the FD-side ``isLikelyStall`` heuristic so the server can
    pre-empt stalls before they reach the chat UI:

    * Empty / whitespace-only output → stall.
    * Very short output (≤ ``_STALL_MAX_LEN`` chars) whose first
      non-empty line opens with an action-announcing phrase → stall.
    * Longer messages that *open* with an action-announcing phrase but
      go on to deliver substantive content (≥
      ``_STALL_FOLLOWTHROUGH_MIN_CHARS`` chars after the first
      sentence) are NOT stalls — that's a "Let me explain…" answer,
      not a "Let me look that up" non-answer.
    """
    if text is None:
        return True
    stripped = str(text).strip()
    if not stripped:
        return True
    first_line = ""
    for line in stripped.splitlines():
        ln = line.strip()
        if ln:
            first_line = ln
            break
    if not first_line:
        return True
    if not _STALL_FIRST_LINE_RE.match(first_line):
        return False
    if len(stripped) <= _STALL_MAX_LEN:
        return True
    # Medium-length message that opens with a stall phrase. Look past
    # the first sentence and see whether the rest of the message has
    # substantive content. If yes, it's a real answer.
    after_first_sentence = _re.split(r"(?<=[.!?])\s+", stripped, maxsplit=1)
    tail = after_first_sentence[1].strip() if len(after_first_sentence) > 1 else ""
    return len(tail) < _STALL_FOLLOWTHROUGH_MIN_CHARS


# Phrases that CLAIM a peer hand-off / send was performed. Used to catch the
# model lying ("Poslao sam MiniMax-u", "I delegated it", "čekam odgovor")
# without actually calling flight_deck/consult_peer this turn. EN + HR.
_DELEGATION_CLAIM_RE = _re.compile(
    r"(?i)("
    r"poslao sam|poslala sam|šaljem|saljem|proslije|delegira|predao sam|"
    r"\bi\s+(sent|delegated|forwarded|passed|asked)\b|sent it to|"
    r"asked\s+\w+\s+(to|for)|čekam\s+(odgovor|opis)|cekam\s+(odgovor|opis)|"
    r"waiting\s+for\s+(the\s+)?(peer|agent|minimax|response|reply)"
    r")"
)


def _claims_delegation(text: str) -> bool:
    """True if the reply claims a peer hand-off/send was done."""
    return bool(text and _DELEGATION_CLAIM_RE.search(str(text)))


# Reply claims it searched/fetched the web (used to catch a model that
# fabricates research instead of calling a web tool). EN + HR phrasing.
# Matches a claim of COMPLETED web research — the thing this detector exists to
# catch (a model that says "I searched the web and found…" without calling a web
# tool). It must NOT match a model that merely *describes* its web tools
# ("web_search to search the web", "web_fetch_batch fetches URLs"), *proposes* to
# search ("I'll search the web for…"), or *asks* what to search ("what would you
# like me to search the web for?"). Those are present/future/capability phrasings,
# not false completion claims, and matching them hijacks ordinary replies (e.g.
# answering "what tools do you have?") into a forced web_search loop. Hence:
# require past-tense / completion signals only.
_WEB_RESEARCH_CLAIM_RE = _re.compile(
    r"(?i)("
    r"(searched|fetched|pulled|scraped|gathered|retrieved|browsed|crawled|queried)\s+"
    r"(the\s+|a\s+|\d+\s+|through\s+)?(web|online|internet|sources?|links?|urls?|sites?)|"
    r"(from|across)\s+\d+\s+(sources?|sites?|links?|izvora)|"
    r"based\s+on\s+(my\s+)?(latest\s+|recent\s+)?(web\s+)?(search|research)|"
    r"according\s+to\s+(the\s+)?(sources?|search\s+results?)|"
    r"paralelno\s+(dohvat|dohvać|dohvac)\w*|dohvatio\s+\d+\s+izvora|pretražio\s+(sam\s+)?web"
    r")"
)

# User explicitly wants fresh/web data and NOT memory — disables the automatic
# memory-context injection for that turn. EN + HR phrasing.
_REFRESH_INTENT_RE = _re.compile(
    r"(?i)("
    r"don'?t\s+use\s+(your\s+|stuff\s+from\s+your\s+)?memory|without\s+(using\s+)?memory|"
    r"not\s+from\s+(your\s+)?memory|ignore\s+(your\s+)?memory|"
    r"ne\s+koristi\s+memoriju|bez\s+memorije|"
    r"refresh\s+(the\s+)?(data|info)|fresh\s+(search|data|results)|"
    r"search\s+the\s+web|from\s+the\s+web|svjež\w*\s+(pretrag|podatk)"
    r")"
)


# Explicit "run a Basna" command — when the user asks for a Basna ensemble run,
# we relay the task to the `basna` tool deterministically instead of leaving the
# choice to the model (weak models otherwise research it themselves). EN + HR.
_BASNA_RUN_RE = _re.compile(
    r"(?i)("
    r"\b(run|execute|start|launch|kick\s*off|spin\s*up|fire\s*up|begin|do|create)\b"
    r"[^.!?\n]{0,40}?\bbasn[aeu]\b"
    r"|\bbasn[aeu]\b[^.!?\n]{0,20}?\b(run|execute|start|launch)\b"
    # Croatian: match the verb *stem* so infinitive / 1pl forms count too
    # (pokreni / pokrenuti / pokrenimo, izvrši / izvršiti, …) — the bare imperative
    # was missing "pokrenuti", which is why "pokrenuti basnu" never relayed.
    r"|\b(pokren\w*|izvr[šs]\w*|kreni)\b[^.!?\n]{0,40}?\bbasn[aeu]\b"
    r")"
)


def _detect_basna_run(user_input: str) -> str | None:
    """If the message is a 'run a Basna' command, return the task to run, else None."""
    if not user_input:
        return None
    # Never relay on injected system envelopes — the Basna/delegate COMPLETION
    # callbacks ("[Basna run '…' finished] … Relay it to the user …") mention
    # "Basna run" and would otherwise re-trigger an endless chain of new runs.
    stripped_lead = user_input.lstrip()
    if stripped_lead.startswith("[") or "Basna run you started" in user_input:
        return None
    if not _BASNA_RUN_RE.search(user_input):
        return None
    msg = user_input.strip()
    # 1) Explicit "task is/task:" marker wins.
    m = _re.search(r"(?i)\btask\s*(?:is|:)\s*(.+)", msg, _re.S)
    if m and m.group(1).strip():
        return m.group(1).strip()
    # 2) Strip a leading "…run/execute a basna [on|about|:]" command prefix.
    stripped = _re.sub(
        r"(?i)^\s*(let'?s|please|can you|could you|go|hey|ok|okay)?\s*[,:]?\s*"
        r"(run|execute|start|launch|kick\s*off|spin\s*up|fire\s*up|do|begin|create)\s+"
        r"(a\s+|an\s+|new\s+|the\s+|another\s+)?basn[aeu]\b\s*(run\s+)?"
        r"(on|about|for|to|regarding|:|—|-)?\s*",
        "", msg, count=1)
    if stripped and stripped != msg and len(stripped.strip()) >= 8:
        return stripped.strip()
    # 3) Fall back to the whole message; the router/auto-title handles it.
    return msg


def _claims_web_research(text: str) -> bool:
    """True if the reply claims it searched/fetched the web."""
    return bool(text and _WEB_RESEARCH_CLAIM_RE.search(str(text)))


def _wants_fresh_data(text: str) -> bool:
    """True if the user asked for fresh/web data and to skip memory."""
    return bool(text and _REFRESH_INTENT_RE.search(str(text)))


class AgentOrchestrationMixin:
    """Core request orchestration: complete() and stream()."""

    # ------------------------------------------------------------------
    # complete() — main entry point
    # ------------------------------------------------------------------

    async def complete(self, user_input: str) -> str:
        """Process user input and return response.

        Args:
            user_input: User's message

        Returns:
            Agent's response
        """
        if not self._initialized:
            await self.initialize()

        # Deterministic Basna relay: when the user explicitly asks to run/execute
        # a Basna, hand the task straight to the `basna` tool rather than relying
        # on the model to pick the tool — weak models otherwise do the research
        # themselves. Covers every channel (web/whatsapp/glasses/api) since they
        # all funnel through complete().
        _tools = getattr(self, "tools", None)
        if _tools is not None and _tools.has_tool("basna"):
            _basna_task = _detect_basna_run(user_input)
            if _basna_task:
                self._add_session_message(role="user", content=user_input)
                try:
                    _res = await self._execute_tool_with_guard(
                        "basna", {"action": "start", "task": _basna_task}, "basna-relay",
                    )
                    _reply = (getattr(_res, "content", "") or "").strip() or "Basna run started."
                except Exception as _e:  # noqa: BLE001 — surface a clean message
                    _reply = f"I couldn't start the Basna run: {_e}"
                self._add_session_message(role="assistant", content=_reply)
                return _reply

        self._last_memory_debug_signature = None
        self._last_semantic_memory_debug_signature = None
        # When the user explicitly asks for fresh/web data (e.g. "refresh from
        # the web, don't use memory"), skip the automatic memory-context
        # injection for this whole turn so the agent can't lean on stale memory.
        self._skip_memory_injection = _wants_fresh_data(user_input)
        restore_skill_env = self._apply_skill_env_overrides_for_run()
        skill_env_restored = False

        turn_usage = self._empty_usage()
        self.last_usage = self._empty_usage()
        # Clear any stale cancel signal from a previous turn so it doesn't
        # immediately abort this new turn.
        cancel_ev = getattr(self, "cancel_event", None)
        if cancel_ev is not None:
            cancel_ev.clear()
        # Reset per-turn duplicate tool call tracker.  This dict maps
        # (tool_name, canonical_args_json) → execution count so that we can
        # detect the LLM re-requesting the exact same tool call and stop it
        # before wasting resources on an infinite re-fetch loop.
        self._turn_tool_call_counts: dict[str, int] = {}
        # Reset blind-rewrite tracker — a full write in a PREVIOUS turn must
        # not block a rewrite the user explicitly asked for in this turn.
        self._blind_write_paths: set[str] = set()
        # Reset per-turn all-blocked streak counter.
        self._all_blocked_streak: int = 0
        # Reset tool-avoidance nudge flag (one nudge per turn max).
        self._tool_avoidance_nudged: bool = False
        # Reset per-turn stall retry counter. Bumped each time we silently
        # re-roll a no-tool-call response that reads as an intent-only
        # stall ("Let me…", "I'll fetch…").  Capped by MAX_STALL_RETRIES
        # so a genuinely-stuck turn still terminates.
        self._stall_retry_count: int = 0
        # Reset per-turn coverage gate streak.
        self._coverage_gate_streak: int = 0
        self._coverage_gate_prev_missing: int = -1
        self._pw_enforcement_streak: int = 0
        # Reset per-turn success flag (updated by finish()).
        self._last_complete_success = True
        # Scale-progress tracker: populated when the scale advisory fires.
        # The tool loop uses this to emit "3 of 27 (11%)" progress.
        self._scale_progress: dict[str, Any] | None = None
        self._deferred_scale_attempts: int = 0
        planning_pipeline: dict[str, Any] | None = None
        recent_source_urls: list[str] = []
        # Pre-populate playbook context cache for the sync _build_messages path.
        if hasattr(self, "_build_playbook_context_note"):
            try:
                _pb_note = await self._build_playbook_context_note(user_input)
                self._playbook_context_cache = {"query": user_input, "note": _pb_note}
            except Exception:
                self._playbook_context_cache = None

        # Refresh project context cache each turn so newly-linked sessions
        # pick up project context without requiring agent restart.
        if hasattr(self, "_refresh_project_context_cache"):
            try:
                await self._refresh_project_context_cache()
            except Exception:
                pass

        effective_user_input = user_input
        effective_user_input, clarification_context_applied = self._resolve_effective_user_input(user_input)

        _force_script = getattr(self, "_force_script_mode", False)

        # Pre-populate cross-session context cache for _build_messages.
        self._cross_session_context_cache = None
        if hasattr(self, "_resolve_cross_session_context"):
            try:
                _cs_note = await self._resolve_cross_session_context(effective_user_input)
                if _cs_note:
                    selectors = self._extract_session_references(effective_user_input)
                    self._cross_session_context_cache = {
                        "note": _cs_note,
                        "selectors": selectors,
                    }
                    log.info(
                        "Cross-session context cached for _build_messages",
                        selectors=selectors,
                        note_length=len(_cs_note),
                    )
            except Exception as exc:
                log.debug("Cross-session context resolution failed", error=str(exc))

        require_all_sources = self._request_references_all_sources(effective_user_input)
        is_worker = getattr(self, "_is_worker", False)
        use_contract_pipeline = self._should_use_contract_pipeline(
            effective_user_input,
            self.planning_enabled,
            pipeline_mode=self.pipeline_mode,
        )
        if clarification_context_applied and not self._is_simple_request(user_input):
            # Clarification follow-ups usually represent partially-specified
            # continuations of a larger request; keep strict completion gating.
            # But a SIMPLE follow-up (a short read-only ask or trivial edit)
            # doesn't justify the heavyweight pipeline — forcing it there is
            # what made trivial follow-ups loop until the stuck detector fired.
            # Check the raw user message, not the merged input (the merge always
            # looks "complex" because it splices in prior context).
            use_contract_pipeline = True
        # Workers should never use contract pipelines — they execute a
        # single focused task and should return as soon as it's done.
        if is_worker:
            use_contract_pipeline = False
        explicit_script_request = self._is_explicit_script_request(effective_user_input)
        enforce_python_worker_mode = explicit_script_request and not is_worker
        session_id = self._current_session_slug()
        session_tool_policy = self._session_tool_policy_payload()
        turn_abort_event = asyncio.Event()
        available_tools = {
            name.strip().lower()
            for name in self.tools.list_tools(
                session_id=session_id,
                session_policy=session_tool_policy,
            )
        }
        python_worker_tools_available = {"write", "shell"}.issubset(available_tools)
        python_worker_attempted = False
        list_task_plan: dict[str, Any] = {
            "enabled": False,
            "members": [],
            "strategy": "none",
            "per_member_action": "",
            "confidence": "low",
        }
        task_contract: dict[str, Any] | None = None
        completion_requirements: list[dict[str, Any]] = []
        completion_feedback: str = ""

        def _restore_skill_env_once() -> None:
            nonlocal skill_env_restored
            if skill_env_restored:
                return
            skill_env_restored = True
            try:
                restore_skill_env()
            except Exception:
                pass

        def finish(text: str, success: bool = True) -> str:
            self._emit_thinking("", phase="done")
            if planning_pipeline is not None:
                self._finalize_pipeline(planning_pipeline, success=success)
            self._finalize_turn_usage(turn_usage)
            _restore_skill_env_once()
            self._last_complete_success = success
            if success:
                self._record_timing_event("last_assistant_at")
            return text

        async def _salvage_partial_result(reason: str) -> str:
            """Try to produce a useful partial result when the agent gets
            stuck, exhausts its budget, or fails after retries.

            1. Checks for a substantial assistant text (>100 chars) — if
               found, returns it directly.
            2. Checks whether tool results contain meaningful content —
               if so, makes one final LLM call asking the model to
               summarise what it has gathered so far.
            3. Falls back to the last short assistant text (>20 chars).
            4. Returns empty string if nothing useful is available.
            """
            if not self.session:
                return ""
            turn_msgs = self.session.messages[turn_start_idx:]

            # 1. Look for a substantial assistant response (likely a real answer).
            last_short_assistant = ""
            for _msg in reversed(turn_msgs):
                if _msg.get("role") == "assistant":
                    _c = str(_msg.get("content", "")).strip()
                    if _c and len(_c) > 100:
                        return _c
                    if _c and len(_c) > 20 and not last_short_assistant:
                        last_short_assistant = _c

            # 2. Check if tool results have meaningful content worth
            #    salvaging via a quick LLM summary call.
            has_tool_results = any(
                _msg.get("role") == "tool"
                and len(str(_msg.get("content", ""))) > 80
                for _msg in turn_msgs
            )
            if has_tool_results:
                try:
                    self._set_runtime_status("thinking")
                    salvage_messages = self._build_messages(
                        tool_messages_from_index=turn_start_idx,
                        query=effective_user_input,
                    )
                    salvage_messages.append(Message(
                        role="user",
                        content=(
                            f"[SYSTEM: {reason}. Based on the tools you used "
                            "and data you gathered, provide the best possible "
                            "response to the original request. Summarise what "
                            "you found. Do NOT call any tools.]"
                        ),
                    ))
                    salvage_resp = await self.provider.complete(
                        messages=salvage_messages,
                        tools=None,
                        max_tokens=2048,
                    )
                    salvage_text = str(
                        getattr(salvage_resp, "content", "") or ""
                    ).strip()
                    if salvage_text and len(salvage_text) > 30:
                        return salvage_text
                except Exception as _salv_err:
                    log.debug("Salvage LLM call failed", error=str(_salv_err))

            # 3. Fall back to last short assistant text.
            return last_short_assistant

        async def attempt_finalize(
            output_text: str,
            iteration: int,
            finish_success: bool = True,
        ) -> tuple[bool, str, bool]:
            """Wrapper around _attempt_finalize_response that updates closure vars."""
            nonlocal completion_feedback, python_worker_attempted, list_task_plan
            (
                finalized,
                final_text,
                fin_success,
                completion_feedback,
                python_worker_attempted,
            ) = await self._attempt_finalize_response(
                output_text=output_text,
                iteration=iteration,
                hard_turn_iterations=hard_turn_iterations,
                finish_success=finish_success,
                effective_user_input=effective_user_input,
                user_input=user_input,
                turn_start_idx=turn_start_idx,
                turn_usage=turn_usage,
                session_tool_policy=session_tool_policy,
                planning_pipeline=planning_pipeline,
                list_task_plan=list_task_plan,
                task_contract=task_contract,
                completion_requirements=completion_requirements,
                completion_feedback=completion_feedback,
                enforce_python_worker_mode=enforce_python_worker_mode,
                python_worker_attempted=python_worker_attempted,
            )
            return finalized, final_text, fin_success

        turn_start_idx = len(self.session.messages) if self.session else 0

        # Compute a domain filter so that _collect_recent_source_urls only
        # returns URLs relevant to the current request.  We extract domains
        # from the effective user input *and* the last assistant response
        # (which contains the specific items the user is referring to in
        # follow-up / clarification scenarios).
        domain_filter = self._extract_mentioned_domains(effective_user_input)
        if self.session and self.session.messages:
            for msg in reversed(self.session.messages):
                if msg.get("role") == "assistant":
                    assistant_text = str(msg.get("content", ""))
                    domain_filter |= self._extract_mentioned_domains(assistant_text)
                    break
        recent_source_urls = self._collect_recent_source_urls(
            turn_start_idx, domain_filter=domain_filter or None,
        )
        allowed_user_input, input_guard_error = await self._enforce_guard(
            guard_type="input",
            interaction_label="user_turn",
            content=user_input,
            turn_usage=turn_usage,
        )
        if not allowed_user_input:
            return finish(input_guard_error, success=False)

        # Add user message to session
        self._add_session_message("user", user_input)
        # Stamp the last REAL user-message time (skip cron/scheduler-driven
        # turns — those record their own automated-run timestamp instead).
        if not getattr(self, "_turn_is_automated", False):
            self._record_timing_event("last_user_msg_at")
        await self._auto_compact_if_needed()
        await self._refresh_cron_context_cache()
        await self._refresh_todo_context_cache()
        await self._refresh_contacts_context_cache()
        await self._refresh_scripts_context_cache()
        await self._refresh_apis_context_cache()
        await self._refresh_datastore_context_cache()
        if clarification_context_applied:
            self._emit_tool_output(
                "task_contract",
                {"step": "clarification_context_applied"},
                "step=clarification_context_applied\nstatus=merged_pending_anchor_into_current_turn",
            )
        # ── Automatic task rephrasing ──────────────────────────────
        task_was_rephrased = False
        if not is_worker and not clarification_context_applied:
            if _force_script:
                # Force-script mode: always rephrase into a script spec.
                effective_user_input, task_was_rephrased = await self._rephrase_for_script_mode(
                    user_input=effective_user_input,
                    turn_usage=turn_usage,
                )
            else:
                effective_user_input, task_was_rephrased = await self._rephrase_task(
                    user_input=effective_user_input,
                    turn_usage=turn_usage,
                )
            if task_was_rephrased:
                require_all_sources = self._request_references_all_sources(effective_user_input)

        # ── Force script mode instruction ─────────────────────────
        # Append AFTER rephrasing so the rephrase works on the clean
        # user prompt, and the mandatory instruction is always present.
        if _force_script:
            _credentials_block = await self._collect_script_credentials()
            _force_script_instruction = (
                "\n\n"
                "===== SYSTEM CONSTRAINT: SCRIPT-ONLY MODE =====\n"
                "Your ONLY allowed action sequence is:\n"
                "  Step 1: write(path='scripts/<name>.py', content=<script>)\n"
                "  Step 2: shell(command='python3 scripts/<name>.py')\n"
                "  Step 3: Report results from stdout.\n\n"
                "FORBIDDEN (will cause errors):\n"
                "- shell() for anything other than python3 <script>\n"
                "- Any interactive commands (mkdir, cat, gws, curl, ls, "
                "grep, python3 -c)\n"
                "- Reading or downloading content into the conversation\n"
                "- Multiple shell calls before writing a script\n\n"
                "The script must contain ALL logic: data fetching, "
                "processing, file I/O, API calls. Nothing happens outside "
                "the script.\n\n"
                "GOOGLE SERVICES (inside the script only):\n"
                "Use `gws` CLI via subprocess.run(). It is pre-installed "
                "and authenticated. Example: "
                "subprocess.run(['gws', 'drive_list', '--folder-id', "
                "'...'], capture_output=True, text=True). "
                "Actions: drive_list, drive_search, drive_download, "
                "drive_info, drive_create, docs_read, docs_append, "
                "mail_list, mail_search, mail_read, calendar_list, "
                "calendar_search, calendar_create, calendar_agenda, raw.\n"
                "==============================================\n"
            )
            if _credentials_block:
                _force_script_instruction += _credentials_block
            effective_user_input = effective_user_input + _force_script_instruction
            log.info("Force script mode: instruction injected into user input")

        list_context_excerpt = self._collect_list_extraction_context()
        # Workers execute a single focused task — skip the heavyweight list
        # task extraction / coverage pipeline which can cause endless loops
        # on simple fetch-and-summarize instructions.
        if getattr(self, "_is_worker", False):
            list_task_plan = list_task_plan  # keep default (disabled)
        elif _force_script:
            # Force-script mode: skip list extraction entirely.
            # The credential URLs injected into the prompt would get
            # picked up by the list extractor as "members", blocking
            # finalization.  Script mode doesn't need scale loop.
            list_task_plan = list_task_plan  # keep default (disabled)
        else:
            list_task_plan = await self._generate_list_task_plan(
                user_input=effective_user_input,
                context_excerpt=list_context_excerpt,
                turn_usage=turn_usage,
            )
        # Direct URL extraction fallback: when the user pastes many URLs
        # in their message, the LLM list extractor (1000 max_tokens) may
        # not be able to return all of them as JSON members.  Detect this
        # and augment the list_task_plan with directly-extracted URLs.
        # IMPORTANT: use the *original* user input when a clarification
        # context was merged, otherwise the assistant's previous response
        # (which may list many URLs) leaks into the member list and causes
        # unwanted scale-loop processing of all items.
        # Skip for workers — they execute a single focused task; URL
        # extraction from the worker prompt would re-enable the list task
        # plan that was intentionally disabled above, causing endless
        # coverage-check loops on simple fetch-and-summarize tasks.
        if not is_worker and not _force_script:
            url_extraction_source = user_input if clarification_context_applied else effective_user_input
            input_urls = self._extract_urls(url_extraction_source)
            if len(input_urls) > len(list_task_plan.get("members", [])):
                existing_members = set(
                    str(m).strip() for m in list_task_plan.get("members", [])
                )
                augmented = list(list_task_plan.get("members", []))
                for url in input_urls:
                    if url not in existing_members:
                        augmented.append(url)
                        existing_members.add(url)
                if len(augmented) > len(list_task_plan.get("members", [])):
                    list_task_plan["members"] = augmented[:150]
                    list_task_plan["enabled"] = True
                    if not list_task_plan.get("per_member_action"):
                        list_task_plan["per_member_action"] = "fetch and process"
                    self._emit_tool_output(
                        "task_contract",
                        {
                            "step": "list_members_augmented_from_input_urls",
                            "llm_extracted": len(existing_members),
                            "augmented_total": len(augmented),
                        },
                        (
                            "step=list_members_augmented_from_input_urls\n"
                            f"llm_extracted={len(existing_members)}\n"
                            f"augmented_total={len(augmented)}"
                        ),
                    )
        extracted_strategy = str(list_task_plan.get("strategy", "none")).strip().lower()
        if extracted_strategy == "script" and not explicit_script_request:
            self._emit_tool_output(
                "task_contract",
                {"step": "python_worker_mode_skipped", "reason": "prefer_internal_tools"},
                "step=python_worker_mode_skipped\nreason=prefer_internal_tools\nmode=direct",
            )
        if enforce_python_worker_mode and not python_worker_tools_available:
            enforce_python_worker_mode = False
            self._emit_tool_output(
                "task_contract",
                {"step": "python_worker_mode_skipped", "reason": "missing_tools"},
                "step=python_worker_mode_skipped\nreason=missing_tools\nrequired=write,shell",
            )
        if bool(list_task_plan.get("enabled", False)):
            self._emit_tool_output(
                "task_contract",
                {
                    "step": "list_task_memory_enabled",
                    "members": len(list_task_plan.get("members", [])),
                    "strategy": extracted_strategy,
                },
                (
                    "step=list_task_memory_enabled\n"
                    f"members={len(list_task_plan.get('members', []))}\n"
                    f"strategy={extracted_strategy}"
                ),
            )
        if enforce_python_worker_mode:
            self._emit_tool_output(
                "task_contract",
                {"step": "python_worker_mode_enabled", "strategy": extracted_strategy or "script"},
                "step=python_worker_mode_enabled\nmode=python_worker_tool_execution",
            )

        # ── Pre-flight scale check ────────────────────────────────
        # The scale micro-loop accelerates large list-processing tasks by
        # taking over the extract→write loop.  The preflight check uses
        # _SKIP_SCALE_DETECTION_RE to avoid firing for discovery-only tasks
        # (e.g. "find all files and return the list") so the scale loop
        # only activates for genuine per-item processing tasks.
        scale_advisory = self._preflight_scale_check(effective_user_input, list_task_plan)
        if scale_advisory:
            # Append playbook patterns for scale tasks (batch-processing).
            if hasattr(self, "_build_playbook_block"):
                try:
                    _pb = await self._build_playbook_block(
                        effective_user_input, task_type="batch-processing",
                    )
                    if _pb:
                        scale_advisory = scale_advisory + _pb
                except Exception:
                    pass  # best-effort
            effective_user_input = effective_user_input + scale_advisory
            self._scale_progress = {"total": 0, "completed": 0}
            _out_strategy = str(list_task_plan.get("output_strategy", "single_file")).strip().lower()
            self._scale_progress["_output_strategy"] = _out_strategy
            self._scale_progress["_output_filename_template"] = str(
                list_task_plan.get("output_filename_template", "")
            ).strip()
            self._scale_progress["_final_action"] = str(
                list_task_plan.get("final_action", "reply")
            ).strip()
            if _out_strategy == "no_file":
                self._scale_progress["_sink_collection"] = ""
                self._scale_progress["_sink_email_to"] = ""
            list_members = list_task_plan.get("members", [])
            if list_members:
                self._scale_progress["items"] = list(list_members)
                self._scale_progress["done_items"] = set()
                self._scale_progress["total"] = len(list_members)
                self._scale_progress["_extraction_mode"] = self._classify_item_extraction_mode(
                    list_members,
                    per_member_action=str(list_task_plan.get("per_member_action", "")),
                    user_input=effective_user_input,
                )
                self._scale_progress["_member_context"] = list_task_plan.get("member_context") or {}
                # For inline mode, store source page content for per-item LLM calls
                if self._scale_progress.get("_extraction_mode") == "inline" and list_context_excerpt:
                    self._scale_progress["_source_page_content"] = list_context_excerpt

        # ── Contract pipeline ─────────────────────────────────────
        if use_contract_pipeline:
            planner_source_urls = [] if clarification_context_applied else recent_source_urls
            task_contract = await self._generate_task_contract(
                user_input=effective_user_input,
                recent_source_urls=planner_source_urls,
                require_all_sources=require_all_sources,
                turn_usage=turn_usage,
                list_task_plan=list_task_plan,
            )
            completion_requirements = self._apply_list_requirements(
                base_requirements=list(task_contract.get("requirements", [])),
                list_task_plan=list_task_plan,
            )
            task_contract["requirements"] = completion_requirements
            prefetch_urls = [
                url
                for url in list(task_contract.get("prefetch_urls", []))
                if isinstance(url, str) and url.startswith(("http://", "https://"))
            ]
            sp = getattr(self, "_scale_progress", None)
            _skip_prefetch = sp is not None and bool(sp.get("items"))
            if prefetch_urls and not _skip_prefetch:
                await self._run_source_report_prefetch(
                    source_urls=prefetch_urls,
                    turn_usage=turn_usage,
                    pipeline_label="task_contract",
                )
            elif prefetch_urls and _skip_prefetch:
                self._emit_tool_output(
                    "task_contract",
                    {
                        "step": "prefetch_skipped",
                        "reason": "scale_progress_has_items",
                        "prefetch_urls": len(prefetch_urls),
                        "scale_items": len(sp.get("items", [])),
                    },
                    (
                        "step=prefetch_skipped\n"
                        f"reason=scale_progress_has_items\n"
                        f"prefetch_urls={len(prefetch_urls)}\n"
                        f"scale_items={len(sp.get('items', []))}\n"
                        "note=micro loop will fetch each item individually"
                    ),
                )
            if sp is not None and not sp.get("items") and prefetch_urls:
                sp["items"] = list(prefetch_urls)
                sp["done_items"] = set()
                sp["total"] = len(prefetch_urls)
                sp["_extraction_mode"] = self._classify_item_extraction_mode(
                    prefetch_urls,
                    per_member_action=str(list_task_plan.get("per_member_action", "")),
                    user_input=effective_user_input,
                )

        # ── Planning pipeline setup ───────────────────────────────
        if self.planning_enabled or task_contract is not None:
            planning_pipeline = self._build_task_pipeline(
                effective_user_input,
                tasks_override=(task_contract or {}).get("tasks"),
                completion_checks=completion_requirements,
            )
            if self.planning_enabled and task_contract is not None:
                planning_pipeline["mode"] = "manual_with_contract"
            elif self.planning_enabled:
                planning_pipeline["mode"] = "manual"
            else:
                planning_pipeline["mode"] = "auto_contract"
            self._emit_pipeline_update("created", planning_pipeline)
            created_children = await self.ensure_pipeline_subagent_contexts(planning_pipeline)
            if created_children:
                self._emit_tool_output(
                    "planning",
                    {"event": "subagent_contexts_spawned", "count": len(created_children)},
                    (
                        "event=subagent_contexts_spawned\n"
                        f"count={len(created_children)}"
                    ),
                )

        # ── Lightweight scale-progress for moderate lists ─────────
        _lw_min = get_config().scale.lightweight_progress_min_members
        if (
            self._scale_progress is None
            and bool(list_task_plan.get("enabled", False))
            and len(list_task_plan.get("members", [])) >= _lw_min
        ):
            self._scale_progress = self._init_scale_progress_from_plan(
                list_task_plan, user_input=effective_user_input,
            )
            # For inline mode, store the full source page content so the
            # micro-loop can feed the entire page (not just tiny snippets)
            # to per-item LLM calls.
            if self._scale_progress.get("_extraction_mode") == "inline" and list_context_excerpt:
                self._scale_progress["_source_page_content"] = list_context_excerpt
            self._emit_tool_output(
                "task_contract",
                {
                    "step": "scale_progress_from_list_task",
                    "members": len(list_task_plan.get("members", [])),
                },
                (
                    "step=scale_progress_from_list_task\n"
                    f"members={len(list_task_plan.get('members', []))}\n"
                    "note=activated lightweight progress tracking for moderate list"
                ),
            )

        # ── Early micro-loop takeover ─────────────────────────────
        _sp_early = getattr(self, "_scale_progress", None)
        _early_items = _sp_early.get("items", []) if _sp_early else []
        # Skip micro-loop when extraction mode is "passthrough" — the user
        # wants to save/store items (e.g. create a datastore table) and the
        # main LLM should handle it directly with tool calls.
        _early_passthrough = (
            _sp_early is not None
            and str(_sp_early.get("_extraction_mode", "")).strip() == "passthrough"
        )
        _can_early_takeover = (
            _sp_early is not None
            and len(_early_items) >= 2
            and not self._items_are_source_urls_only(_early_items)
            and not _early_passthrough
        )
        if _can_early_takeover:
            log.info("Early scale micro-loop takeover", items=len(_early_items))
            micro_result = await self._run_micro_loop_and_summarize(
                effective_user_input=effective_user_input,
                list_task_plan=list_task_plan,
                turn_usage=turn_usage,
                session_tool_policy=session_tool_policy,
                planning_pipeline=planning_pipeline,
                step_label="early_scale_micro_loop_takeover",
            )
            if micro_result.get("cancelled"):
                # User cancelled — finalize immediately.
                micro_summary = micro_result["summary"]
                self._update_clarification_state(
                    user_input=user_input,
                    effective_user_input=effective_user_input,
                    assistant_response=micro_summary,
                )
                await self._persist_assistant_response(micro_summary)
                return finish(micro_summary, success=False)
            # Scale processing done — fall through to main loop so the
            # LLM can handle remaining steps (summarise, email, etc.).
            #
            # If the micro-loop processed NOTHING (all items failed),
            # the scale detection was wrong (e.g. items were glob
            # patterns, not actual files).  Clear _scale_progress so
            # subsequent globs don't get their results hijacked into
            # a zombie scale state.
            if micro_result.get("processed", 0) == 0:
                log.info(
                    "Early scale micro-loop processed nothing — clearing scale progress",
                    failed=micro_result.get("failed", 0),
                    total=micro_result.get("total", 0),
                )
                self._scale_progress = None
            _sp_cont = getattr(self, "_scale_progress", None)
            _cont_out = _sp_cont.get("_output_file", "") if _sp_cont else ""
            completion_feedback = (
                f"[Scale loop completed] {micro_result.get('processed', 0)} of "
                f"{micro_result.get('total', 0)} items processed."
                + (f" Output file: {_cont_out}." if _cont_out else "")
                + "\n\nNow review the ORIGINAL user request and continue "
                "with any remaining steps not handled by the scale "
                "processing (e.g. summarising the output, emailing, "
                "analysing, charting). If all steps are already "
                "complete, provide a concise final summary."
            )
            log.info(
                "Scale early takeover done — continuing main loop for post-processing",
                processed=micro_result.get("processed", 0),
                total=micro_result.get("total", 0),
            )

        # ── Main agent loop ───────────────────────────────────────
        base_turn_iterations = self.max_iterations + (2 if completion_requirements else 0)
        planned_turn_iterations = self._compute_turn_iteration_budget(
            base_iterations=base_turn_iterations,
            planning_pipeline=planning_pipeline,
            completion_requirements=completion_requirements,
        )
        if scale_advisory:
            member_count = len(list_task_plan.get("members", []))
            estimated_items = member_count if member_count > 15 else 50
            # Tighter budget: cap at 120 iterations (was 400) and use 2x
            # per-item (was 3x).  Most list tasks complete well within this.
            scale_budget = min(120, 10 + estimated_items * 2)
            if scale_budget > planned_turn_iterations:
                planned_turn_iterations = scale_budget
        # Hard ceiling: 2x planned (was 3x) capped at 200 (was 500).
        # Prevents runaway loops when the critic keeps rejecting.
        hard_turn_iterations = max(planned_turn_iterations, min(200, planned_turn_iterations * 2))
        soft_turn_iterations = planned_turn_iterations
        extension_step = max(6, min(24, max(1, planned_turn_iterations // 3)))
        max_stagnant_iterations = 6
        stagnant_iterations = 0
        progress_window: list[bool] = []
        previous_progress_snapshot: dict[str, Any] | None = None
        last_completion_feedback_signature = ""
        # Write-loop detection: track how many times each file path is written.
        _write_path_counts: dict[str, int] = {}
        _write_loop_nudged = False
        if planned_turn_iterations != base_turn_iterations:
            self._emit_tool_output(
                "completion_gate",
                {
                    "step": "iteration_budget",
                    "base_limit": base_turn_iterations,
                    "effective_limit": planned_turn_iterations,
                    "hard_limit": hard_turn_iterations,
                },
                (
                    "step=iteration_budget\n"
                    f"base_limit={base_turn_iterations}\n"
                    f"effective_limit={planned_turn_iterations}\n"
                    f"hard_limit={hard_turn_iterations}"
                ),
            )

        # Accumulate intermediate text the LLM produces alongside
        # tool calls so the final answer includes the full narrative.
        _intermediate_texts: list[str] = []
        # Whether intermediate narration was streamed live (web/whatsapp). When
        # it was, the final answer is just the last response — the blurbs were
        # already shown during the run, so we don't repeat them combined.
        _narration_streamed = False
        _tools_executed_count = 0
        # Nudge at most once per turn when a reasoning-preamble is about to be
        # shipped as the answer (see the no-tool finalize path below).
        _preamble_nudged = False
        # Nudge at most once per turn when raw planning/contract JSON leaks as
        # the final response (weak models echo the contract instead of answering).
        _plan_leak_nudged = False

        for iteration in range(hard_turn_iterations):
            # ── External cancellation check ───────────────────────
            cancel_ev: asyncio.Event | None = getattr(self, "cancel_event", None)
            if cancel_ev is not None and cancel_ev.is_set():
                self._set_runtime_status("waiting")
                self._emit_thinking("Cancelled", phase="done")
                cancel_ev.clear()
                return finish("Request cancelled by user.", success=False)

            # ── Pipeline runtime tick ─────────────────────────────
            if planning_pipeline is not None:
                runtime_update = self._tick_pipeline_runtime(
                    planning_pipeline,
                    event=f"runtime_tick_{iteration + 1}",
                )
                if bool(runtime_update.get("changed", False)):
                    activated = runtime_update.get("activated", [])
                    if isinstance(activated, list) and activated:
                        await self.ensure_pipeline_subagent_contexts(
                            planning_pipeline,
                            task_ids=[str(item) for item in activated],
                        )
                    self._emit_pipeline_update("runtime_update", planning_pipeline)
                if str(planning_pipeline.get("state", "")).strip().lower() == "failed":
                    self._set_runtime_status("waiting")
                    _partial = await _salvage_partial_result("Retries exhausted")
                    if _partial:
                        return finish(
                            "⚠️ I wasn't able to fully complete this request "
                            "(retries exhausted), but here's what I have so "
                            f"far:\n\n{_partial}",
                            success=False,
                        )
                    return finish(MSG_RETRIES_EXHAUSTED, success=False)

            # ── Iteration budget management ───────────────────────
            if iteration >= soft_turn_iterations:
                recent_progress = any(progress_window[-4:])
                remaining_work = (
                    self._pipeline_has_remaining_work(planning_pipeline)
                    or bool(completion_feedback)
                    or bool(completion_requirements)
                )
                if recent_progress and remaining_work and soft_turn_iterations < hard_turn_iterations:
                    previous_limit = soft_turn_iterations
                    soft_turn_iterations = min(hard_turn_iterations, soft_turn_iterations + extension_step)
                    self._emit_tool_output(
                        "completion_gate",
                        {
                            "step": "iteration_budget_extended",
                            "previous_limit": previous_limit,
                            "new_limit": soft_turn_iterations,
                            "hard_limit": hard_turn_iterations,
                        },
                        (
                            "step=iteration_budget_extended\n"
                            f"previous_limit={previous_limit}\n"
                            f"new_limit={soft_turn_iterations}\n"
                            f"hard_limit={hard_turn_iterations}"
                        ),
                    )
                else:
                    self._emit_tool_output(
                        "completion_gate",
                        {
                            "step": "iteration_budget_exhausted",
                            "soft_limit": soft_turn_iterations,
                            "hard_limit": hard_turn_iterations,
                            "recent_progress": recent_progress,
                            "remaining_work": remaining_work,
                        },
                        (
                            "step=iteration_budget_exhausted\n"
                            f"soft_limit={soft_turn_iterations}\n"
                            f"hard_limit={hard_turn_iterations}\n"
                            f"recent_progress={recent_progress}\n"
                            f"remaining_work={remaining_work}"
                        ),
                    )
                    self._set_runtime_status("waiting")
                    _partial_budget = await _salvage_partial_result(
                        "Iteration budget exhausted"
                    )
                    if _partial_budget:
                        return finish(
                            "⚠️ I wasn't able to fully complete this request "
                            "(iteration budget exhausted), but here's what I "
                            f"have so far:\n\n{_partial_budget}",
                            success=False,
                        )
                    return finish(MSG_BUDGET_EXHAUSTED, success=False)

            # ── Progress / stagnation tracking ────────────────────
            current_snapshot = self._capture_turn_progress_snapshot(turn_start_idx, planning_pipeline)
            if previous_progress_snapshot is not None:
                snapshot_progress = self._has_turn_progress(previous_progress_snapshot, current_snapshot)
                completion_feedback_signature = completion_feedback.strip()
                feedback_progress = bool(
                    completion_feedback_signature
                    and completion_feedback_signature != last_completion_feedback_signature
                )
                if feedback_progress:
                    last_completion_feedback_signature = completion_feedback_signature
                progressed = snapshot_progress or feedback_progress
                progress_window.append(progressed)
                if progressed:
                    stagnant_iterations = 0
                else:
                    stagnant_iterations += 1
                    if (
                        stagnant_iterations >= max_stagnant_iterations
                        and iteration >= max(2, min(base_turn_iterations, soft_turn_iterations) // 2)
                    ):
                        self._emit_tool_output(
                            "completion_gate",
                            {
                                "step": "stuck_detected",
                                "stagnant_iterations": stagnant_iterations,
                                "iteration": iteration + 1,
                            },
                            (
                                "step=stuck_detected\n"
                                f"iteration={iteration + 1}\n"
                                f"stagnant_iterations={stagnant_iterations}"
                            ),
                        )
                        self._set_runtime_status("waiting")
                        _partial_stuck = await _salvage_partial_result(
                            "Agent is stuck and not making progress"
                        )
                        # Conversational turn (no multi-step pipeline, no
                        # completion requirements): a salvaged substantive reply
                        # IS the answer. Return it cleanly instead of the
                        # alarming "I got stuck" framing — the progress detector
                        # simply doesn't count text answers as "progress".
                        _conversational = (
                            planning_pipeline is None and not completion_requirements
                        )
                        if _partial_stuck and _conversational:
                            return finish(_partial_stuck, success=True)
                        if _partial_stuck:
                            return finish(
                                "⚠️ I got stuck and couldn't make further "
                                "progress, but here's what I have so "
                                f"far:\n\n{_partial_stuck}",
                                success=False,
                            )
                        return finish(MSG_STUCK, success=False)
            else:
                progress_window.append(False)
            previous_progress_snapshot = current_snapshot

            # ── LLM call ──────────────────────────────────────────
            self._set_runtime_status("thinking")
            messages = self._build_messages(
                tool_messages_from_index=turn_start_idx,
                query=effective_user_input,
                planning_pipeline=planning_pipeline,
                list_task_plan=list_task_plan,
            )
            if completion_feedback:
                messages.append(
                    Message(
                        role="user",
                        content=completion_feedback,
                    )
                )

            # Refresh Google OAuth connection flag so Google-dependent
            # tools appear/disappear in the agent's tool list the moment
            # the user connects or disconnects via Flight Deck. Cheap:
            # flight-deck client mode re-uses the in-process access-token
            # cache; local mode hits SQLite app_state.
            try:
                from captain_claw.google_oauth_manager import GoogleOAuthManager
                await GoogleOAuthManager(self.session_manager).is_connected()
            except Exception:
                pass

            active_task_tool_policy = self._active_task_tool_policy_payload(planning_pipeline)
            tool_defs = self.tools.get_definitions(
                session_id=session_id,
                session_policy=session_tool_policy,
                task_policy=active_task_tool_policy,
            )
            # ── Nano mode: hard-restrict to a barebone tool set ──
            # Small local models can't reliably pick from a long tool list;
            # cut to the script-writing essentials and let them shell out.
            _nano_active = bool(
                getattr(self, "instructions", None) and self.instructions.use_nano
            )
            if _nano_active and tool_defs:
                _before = len(tool_defs)
                tool_defs = [
                    td for td in tool_defs
                    if td.get("name", "").strip().lower() in _NANO_TOOLS
                ]
                log.info(
                    "Nano mode: restricted tool set",
                    kept=[td["name"] for td in tool_defs],
                    removed=_before - len(tool_defs),
                )

            # ── Force script mode: strip tools the LLM must NOT use ──
            # Instruction alone is insufficient (Haiku ignores it).
            # Keep only tools needed for writing & running a script.
            if _force_script and tool_defs:
                _SCRIPT_MODE_TOOLS = {"shell", "write", "read", "edit", "glob"}
                _before = len(tool_defs)
                tool_defs = [
                    td for td in tool_defs
                    if td.get("name", "").strip().lower() in _SCRIPT_MODE_TOOLS
                ]
                log.info(
                    "Force script mode: restricted tool set",
                    kept=[td["name"] for td in tool_defs],
                    removed=_before - len(tool_defs),
                )

            # ── Eco mode: lazy tool definitions + intent preselection ──
            # When eco mode (micro instructions) is active, only send full
            # definitions for core tools + intent-matched tools.  The rest
            # are listed by name in the system prompt so the LLM knows they
            # exist but we save ~1K tokens per call on schema overhead.
            # The tools remain executable — the registry still has them.
            _eco_active = getattr(self, "instructions", None) and self.instructions.use_micro
            if _eco_active and tool_defs and not _force_script and not _nano_active:
                _intent_tools = _eco_select_tools_by_intent(effective_user_input or "")
                # Also include tools already used in this turn or recent
                # session history (the LLM may want to call them again).
                _turn_used: set[str] = set()
                if hasattr(self, "_turn_tool_call_counts"):
                    _turn_used = set(self._turn_tool_call_counts.keys())
                # Include tools used in recent session messages so the LLM
                # retains access to tools it used in prior turns (e.g.
                # google_mail for drafting after datastore lookup).
                _session_used: set[str] = set()
                if self.session and self.session.messages:
                    for _msg in self.session.messages[-40:]:
                        for _tc in (_msg.get("tool_calls") or []):
                            _tn = _tc.get("name") or _tc.get("function", {}).get("name", "")
                            if _tn:
                                _session_used.add(_tn)
                _keep = _ECO_CORE_TOOLS | _intent_tools | _turn_used | _session_used
                # Always preserve MCP tools (registered by Flight Deck via the
                # MCPProxyConnector — names are prefixed ``mcp_<server>_<tool>``).
                # The user opted into them explicitly by configuring an MCP
                # server in Flight Deck, and the eco intent patterns can't know
                # about arbitrary upstream tool names.  Without this guard the
                # LLM sees the names in the system prompt but has no schema to
                # call them with, so it hallucinates "MCP execution is blocked"
                # and refuses to invoke them.
                _mcp_tools = {
                    td["name"] for td in tool_defs
                    if str(td.get("name", "")).startswith("mcp_")
                }
                _keep = _keep | _mcp_tools
                _full_count = len(tool_defs)
                _deferred = [td["name"] for td in tool_defs if td.get("name") not in _keep]
                tool_defs = [td for td in tool_defs if td.get("name") in _keep]
                if _deferred:
                    log.info(
                        "Eco mode: deferred tool definitions",
                        core=len(tool_defs),
                        deferred=len(_deferred),
                        intent_matched=list(_intent_tools & {td["name"] for td in tool_defs}),
                    )
            log.debug(
                "Tool definitions available",
                count=len(
                    self.tools.list_tools(
                        session_id=session_id,
                        session_policy=session_tool_policy,
                        task_policy=active_task_tool_policy,
                    )
                ),
                tools_sent=bool(tool_defs),
            )

            _ctx = getattr(self, "last_context_window", {})
            _ctx_tokens = int(_ctx.get("prompt_tokens", 0))
            _ctx_budget = int(_ctx.get("context_budget_tokens", 1))
            _ctx_pct = round(_ctx_tokens / _ctx_budget * 100, 1) if _ctx_budget else 0
            _ctx_kb = round(_ctx_tokens * 4 / 1024, 1)
            _session_msgs = len(self.session.messages) if self.session else 0
            log.info(
                "Calling LLM",
                iteration=iteration + 1,
                message_count=len(messages),
                session_messages=_session_msgs,
                context_tokens=_ctx_tokens,
                context_kb=_ctx_kb,
                context_budget=_ctx_budget,
                context_pct=f"{_ctx_pct}%",
                dropped=int(_ctx.get("dropped_messages", 0)),
            )
            # ── LLM call with retry for transient errors ───────────
            _max_llm_retries = 4
            response = None  # type: ignore[assignment]
            for _llm_attempt in range(_max_llm_retries + 1):
                try:
                    response = await self._complete_with_guards(
                        messages=messages,
                        tools=tool_defs if tool_defs else None,
                        interaction_label=f"turn_{iteration + 1}",
                        turn_usage=turn_usage,
                    )
                    break  # success
                except GuardBlockedError as e:
                    final = str(e)
                    self._update_clarification_state(
                        user_input=user_input,
                        effective_user_input=effective_user_input,
                        assistant_response=final,
                    )
                    await self._persist_assistant_response(final)
                    return finish(final, success=False)
                except Exception as e:
                    error_str = str(e)

                    # Check if transient and retriable
                    _is_transient = False
                    if isinstance(e, LLMAPIError) and e.status_code in (408, 429, 502, 503, 529):
                        _is_transient = True
                    elif isinstance(e, (LLMAPIError, LLMError)):
                        _lower = error_str.lower()
                        if "timeout" in _lower or "timed out" in _lower or "overloaded" in _lower:
                            _is_transient = True

                    # Orphaned tool_result → rebuild messages and retry once.
                    if (
                        isinstance(e, LLMAPIError)
                        and e.status_code == 400
                        and "tool_use_id" in error_str
                        and "tool_result" in error_str
                        and _llm_attempt < _max_llm_retries
                    ):
                        log.warning(
                            "Orphaned tool_result detected, rebuilding messages",
                            error=error_str,
                            attempt=_llm_attempt + 1,
                        )
                        # Force compact to clean up the broken session state.
                        try:
                            await self.compact_session(force=True, trigger="orphan_fix")
                        except Exception:
                            pass
                        messages = self._build_messages(
                            tool_messages_from_index=turn_start_idx,
                            query=effective_user_input,
                            planning_pipeline=planning_pipeline,
                            list_task_plan=list_task_plan,
                        )
                        if completion_feedback:
                            messages.append(Message(role="user", content=completion_feedback))
                        continue

                    if _is_transient and _llm_attempt < _max_llm_retries:
                        _delay = min(5 * (2 ** _llm_attempt), 60)  # 5, 10, 20, 40 (cap 60s)
                        log.warning(
                            "Transient LLM error, retrying",
                            error=error_str,
                            attempt=_llm_attempt + 1,
                            max_retries=_max_llm_retries,
                            delay=_delay,
                        )
                        await asyncio.sleep(_delay)
                        continue

                    # Not transient or exhausted retries — handle as before
                    tool_output = self._collect_turn_tool_output(turn_start_idx)
                    if tool_output and "500" in error_str:
                        log.warning("Tool result call failed (500), returning tool output")
                        final = await self._friendly_tool_output_response(
                            user_input=effective_user_input,
                            tool_output=tool_output,
                            turn_usage=turn_usage,
                        )
                        finalized, final_text, finish_success = await attempt_finalize(
                            output_text=final,
                            iteration=iteration,
                            finish_success=False,
                        )
                        if finalized:
                            return finish(final_text, success=finish_success)
                        break  # exit retry loop, continue outer turn loop
                    log.error("LLM call failed", error=error_str, exc_info=True)
                    _restore_skill_env_once()
                    raise
            else:
                # for-loop exhausted without break — 500 fallback used `break`
                continue  # continue outer turn loop

            # Guard: if `break` exited the retry loop after a 500 fallback
            # without a successful LLM response, skip to the next iteration.
            if response is None:
                continue

            self._record_pipeline_task_usage(
                planning_pipeline,
                response.usage if isinstance(getattr(response, "usage", None), dict) else {},
            )

            # ── Explicit tool calls ───────────────────────────────
            if response.tool_calls:
                log.info("Tool calls detected", count=len(response.tool_calls), calls=response.tool_calls)
                # Capture intermediate text emitted alongside tool calls.
                _tc_text = str(response.content or "").strip()
                if _tc_text:
                    _intermediate_texts.append(_tc_text)
                    if self._emit_narration(_tc_text, iteration):
                        _narration_streamed = True
                _tools_executed_count += len(response.tool_calls)
                self._add_session_message(
                    role="assistant",
                    content=str(response.content or ""),
                    tool_calls=self._serialize_tool_calls_for_session(response.tool_calls),
                )
                _tool_results = await self._handle_tool_calls(
                    response.tool_calls,
                    turn_usage=turn_usage,
                    session_policy=session_tool_policy,
                    task_policy=active_task_tool_policy,
                    abort_event=turn_abort_event,
                )

                # ── Write-loop detection ───────────────────────────
                # If the agent keeps rewriting the same file(s), it's
                # stuck in a loop.  Inject a nudge to finalize.
                if _tool_results and not _write_loop_nudged:
                    _write_tools_this_iter = []
                    for _tc_orig in response.tool_calls:
                        _tn = str(_tc_orig.name).lower()
                        if _tn in ("write", "edit", "file_write", "file_edit"):
                            _tc_args = _tc_orig.arguments
                            if isinstance(_tc_args, str):
                                try:
                                    _tc_args = json.loads(_tc_args)
                                except (json.JSONDecodeError, TypeError):
                                    _tc_args = {}
                            if isinstance(_tc_args, dict):
                                _wp = str(_tc_args.get("path", "") or _tc_args.get("file_path", ""))
                                if _wp:
                                    _write_tools_this_iter.append(_wp)
                    for _wp in _write_tools_this_iter:
                        _write_path_counts[_wp] = _write_path_counts.get(_wp, 0) + 1
                    # If ALL tool calls this iteration are rewrites of
                    # already-written files (2+ times), break the loop.
                    if (
                        _write_tools_this_iter
                        and len(_write_tools_this_iter) == len(response.tool_calls)
                        and all(_write_path_counts.get(p, 0) >= 2 for p in _write_tools_this_iter)
                    ):
                        _write_loop_nudged = True
                        log.info(
                            "Write-loop detected: agent rewrote same file(s)",
                            paths=list(set(_write_tools_this_iter)),
                            counts={p: _write_path_counts[p] for p in set(_write_tools_this_iter)},
                        )
                        self._add_session_message(
                            role="user",
                            content=(
                                "[system] STOP — you have already written this file. "
                                "The file is saved and complete. "
                                "Do NOT rewrite it again. Provide your final text "
                                "response now summarizing what you did and referencing "
                                "the file(s) you created. Do NOT call any more tools."
                            ),
                        )

                # ── Free iteration for successful glob ────────────
                # When glob finds files, the iteration was purely
                # informational — give the budget back so the agent
                # can spend it on real work.
                if _tool_results and any(
                    str(r.get("tool_name", "")).lower() == "glob"
                    and r.get("success")
                    and "No files found" not in str(r.get("content", ""))
                    for r in _tool_results
                ):
                    soft_turn_iterations += 1
                    hard_turn_iterations += 1
                if planning_pipeline is not None:
                    activated = self._advance_pipeline(planning_pipeline, event="tool_calls_completed")
                    if activated:
                        await self.ensure_pipeline_subagent_contexts(
                            planning_pipeline,
                            task_ids=[str(item) for item in activated],
                        )

                # ── Glob-based scale init ─────────────────────────
                # When glob discovers files and scale_progress has no
                # items, populate scale progress directly from the glob
                # results.  This is cheaper than LLM re-extraction and
                # fills the gap where glob is (correctly) not in
                # _CONTENT_FETCH_TOOLS.
                if _tool_results and not getattr(self, "_force_script_mode", False) and self._needs_deferred_scale_init():
                    _glob_items = self._try_scale_init_from_glob(
                        tool_results=_tool_results,
                        effective_user_input=effective_user_input,
                        list_task_plan=list_task_plan,
                    )
                    if _glob_items:
                        # Update list_task_plan with discovered members.
                        list_task_plan["members"] = _glob_items
                        list_task_plan["enabled"] = True
                        # Recalculate iteration budget to accommodate
                        # the newly discovered scale items.
                        _glob_scale_budget = min(400, 10 + len(_glob_items) * 3)
                        if _glob_scale_budget > soft_turn_iterations:
                            log.info(
                                "Glob scale init: expanding iteration budget",
                                old_soft=soft_turn_iterations,
                                new_soft=_glob_scale_budget,
                                old_hard=hard_turn_iterations,
                                new_hard=max(hard_turn_iterations, min(500, _glob_scale_budget * 3)),
                            )
                            soft_turn_iterations = _glob_scale_budget
                            hard_turn_iterations = max(
                                hard_turn_iterations,
                                min(500, _glob_scale_budget * 3),
                            )
                            extension_step = max(6, min(24, max(1, _glob_scale_budget // 3)))
                        # Inject the scale advisory into effective_user_input
                        # so the LLM adopts the incremental strategy.
                        _sp_glob = getattr(self, "_scale_progress", None)
                        _glob_advisory = _build_scale_advisory(
                            item_count=len(_glob_items),
                            output_strategy=str(
                                (_sp_glob or {}).get("_output_strategy", "single_file")
                            ),
                            filename_template=str(
                                (_sp_glob or {}).get("_output_filename_template", "")
                            ),
                            final_action=str(
                                (_sp_glob or {}).get("_final_action", "reply")
                            ),
                        )
                        if _glob_advisory:
                            # Only append if no scale advisory was already
                            # injected at preflight.
                            if "--- SCALE ADVISORY" not in effective_user_input:
                                effective_user_input = effective_user_input + _glob_advisory
                            # Inject a concise note into session context so
                            # subsequent LLM calls see the item list.
                            self._add_session_message(
                                role="user",
                                content=(
                                    f"[system] Scale advisory: {len(_glob_items)} "
                                    f"items discovered via glob. "
                                    f"Process them one at a time using the "
                                    f"incremental read→write strategy."
                                ),
                            )

                # ── Deferred scale init after fetch ───────────────
                # Only consider deferred scale init when a content-
                # fetching tool ran *successfully* (web_fetch, web_get,
                # read, pdf_extract, etc.).  Failed fetches, datastore,
                # glob, shell, and todo results don't bring in new
                # article content that warrants list re-extraction.
                # Also skip when a script was just executed — the read
                # is checking the script's output, not fetching new
                # external content for scale processing.
                _had_content_fetch = _tool_results and any(
                    str(r.get("tool_name", "")).lower() in _CONTENT_FETCH_TOOLS
                    and r.get("success", False)
                    for r in _tool_results
                )
                _needs_def = (
                    _had_content_fetch
                    and not getattr(self, "_force_script_mode", False)
                    and self._needs_deferred_scale_init()
                    and not self._turn_has_successful_script_execution(turn_start_idx)
                )
                log.info(
                    "Post-tool deferred check",
                    needs_deferred=_needs_def,
                    had_content_fetch=_had_content_fetch,
                    scale_items=len(
                        (getattr(self, "_scale_progress", None) or {}).get("items", [])
                    ),
                )
                if _needs_def:
                    list_task_plan = await self._deferred_scale_init(
                        effective_user_input=effective_user_input,
                        list_task_plan=list_task_plan,
                        turn_usage=turn_usage,
                    )
                    # If deferred init populated REAL items (not just source
                    # URLs), enter micro-loop.  The source-URL guard prevents
                    # entering the micro-loop with the article URL repeated.
                    # Skip when extraction mode is "passthrough" (save/store intent).
                    # Also skip when item count is very high (>100) — the task
                    # clearly needs a script-based approach, not per-item LLM calls.
                    _sp_deferred = getattr(self, "_scale_progress", None)
                    _deferred_items = _sp_deferred.get("items", []) if _sp_deferred else []
                    _deferred_passthrough = (
                        _sp_deferred is not None
                        and str(_sp_deferred.get("_extraction_mode", "")).strip() == "passthrough"
                    )
                    _deferred_too_many = len(_deferred_items) > 100
                    if _deferred_too_many:
                        log.info(
                            "Deferred scale init: too many items for micro-loop, skipping",
                            item_count=len(_deferred_items),
                        )
                    if (
                        _sp_deferred is not None
                        and len(_deferred_items) >= 2
                        and not self._items_are_source_urls_only(_deferred_items)
                        and not _deferred_passthrough
                        and not _deferred_too_many
                    ):
                        micro_result = await self._run_micro_loop_and_summarize(
                            effective_user_input=effective_user_input,
                            list_task_plan=list_task_plan,
                            turn_usage=turn_usage,
                            session_tool_policy=session_tool_policy,
                            planning_pipeline=planning_pipeline,
                            step_label="deferred_scale_micro_loop_takeover",
                        )
                        if micro_result.get("cancelled"):
                            finalized, final_text, finish_success = await attempt_finalize(
                                output_text=micro_result["summary"],
                                iteration=iteration,
                                finish_success=False,
                            )
                            if finalized:
                                return finish(final_text, success=finish_success)
                            continue
                        # Continue main loop for remaining steps.
                        _sp_cont = getattr(self, "_scale_progress", None)
                        _cont_out = _sp_cont.get("_output_file", "") if _sp_cont else ""
                        completion_feedback = (
                            f"[Scale loop completed] {micro_result.get('processed', 0)} of "
                            f"{micro_result.get('total', 0)} items processed."
                            + (f" Output file: {_cont_out}." if _cont_out else "")
                            + "\n\nNow review the ORIGINAL user request and "
                            "continue with any remaining steps not handled "
                            "by the scale processing (e.g. summarising the "
                            "output, emailing, analysing, charting). If all "
                            "steps are already complete, provide a concise "
                            "final summary."
                        )
                        _post_budget = iteration + 10
                        if soft_turn_iterations < _post_budget:
                            soft_turn_iterations = _post_budget
                        if hard_turn_iterations < _post_budget:
                            hard_turn_iterations = _post_budget
                        stagnant_iterations = 0
                        log.info(
                            "Scale micro-loop done (deferred path) — continuing for post-processing",
                            processed=micro_result.get("processed", 0),
                        )
                        continue

                # ── Micro-loop takeover (tool call path) ──────────
                if self._scale_loop_ready():
                    log.info("Scale micro-loop takeover (tool call path)")
                    sp = getattr(self, "_scale_progress", None)
                    output_file = sp.get("_output_file", "") if sp else ""
                    micro_result = await self._run_micro_loop_and_summarize(
                        effective_user_input=effective_user_input,
                        list_task_plan=list_task_plan,
                        turn_usage=turn_usage,
                        session_tool_policy=session_tool_policy,
                        planning_pipeline=planning_pipeline,
                        step_label="scale_micro_loop_takeover",
                        output_file=output_file,
                    )
                    if micro_result.get("cancelled"):
                        finalized, final_text, finish_success = await attempt_finalize(
                            output_text=micro_result["summary"],
                            iteration=iteration,
                            finish_success=False,
                        )
                        if finalized:
                            return finish(final_text, success=finish_success)
                        continue
                    # Continue main loop for remaining steps.
                    _sp_cont = getattr(self, "_scale_progress", None)
                    _cont_out = _sp_cont.get("_output_file", "") if _sp_cont else ""
                    completion_feedback = (
                        f"[Scale loop completed] {micro_result.get('processed', 0)} of "
                        f"{micro_result.get('total', 0)} items processed."
                        + (f" Output file: {_cont_out}." if _cont_out else "")
                        + "\n\nNow review the ORIGINAL user request and "
                        "continue with any remaining steps not handled "
                        "by the scale processing (e.g. summarising the "
                        "output, emailing, analysing, charting). If all "
                        "steps are already complete, provide a concise "
                        "final summary."
                    )
                    _post_budget = iteration + 10
                    if soft_turn_iterations < _post_budget:
                        soft_turn_iterations = _post_budget
                    if hard_turn_iterations < _post_budget:
                        hard_turn_iterations = _post_budget
                    stagnant_iterations = 0
                    log.info(
                        "Scale micro-loop done (tool path) — continuing for post-processing",
                        processed=micro_result.get("processed", 0),
                    )
                    continue

                # ── Post-write finalization hint ───────────────────
                # When a write_file task already has a successful write
                # in this turn, nudge the LLM to finalize instead of
                # making more tool calls.  Without this, the LLM may
                # loop on duplicate write attempts indefinitely.
                _fa = str(list_task_plan.get("final_action", "")).strip().lower()
                if (
                    _fa == "write_file"
                    and self._turn_has_successful_tool(turn_start_idx, "write")
                    and not completion_feedback
                ):
                    completion_feedback = (
                        "The output file has been written successfully. "
                        "Do NOT write the file again. "
                        "Respond with a brief text summary of what you "
                        "created (file name, key contents). Do NOT call "
                        "any more tools."
                    )
                    log.info("Post-write finalization hint injected")

                if not self._supports_tool_result_followup():
                    output = self._collect_turn_tool_output(turn_start_idx)
                    final = await self._friendly_tool_output_response(
                        user_input=effective_user_input,
                        tool_output=output,
                        turn_usage=turn_usage,
                    )
                    finalized, final_text, finish_success = await attempt_finalize(
                        output_text=final,
                        iteration=iteration,
                        finish_success=True,
                    )
                    if finalized:
                        return finish(final_text, success=finish_success)
                    continue
                continue

            # ── Embedded tool calls (fallback) ────────────────────
            embedded_calls = self._extract_tool_calls_from_content(response.content)
            if embedded_calls:
                log.info(
                    "Tool calls found in response text",
                    count=len(embedded_calls),
                    calls=[(c.name, list(c.arguments.keys())) for c in embedded_calls],
                    content_preview=str(response.content or "")[:300],
                )
                _emb_text = str(response.content or "").strip()
                if _emb_text:
                    _intermediate_texts.append(_emb_text)
                    if self._emit_narration(_emb_text, iteration):
                        _narration_streamed = True
                _tools_executed_count += len(embedded_calls)
                self._add_session_message(
                    role="assistant",
                    content=str(response.content or ""),
                    tool_calls=self._serialize_tool_calls_for_session(embedded_calls),
                )
                _emb_tool_results = await self._handle_tool_calls(
                    embedded_calls,
                    turn_usage=turn_usage,
                    session_policy=session_tool_policy,
                    task_policy=active_task_tool_policy,
                    abort_event=turn_abort_event,
                )

                # Free iteration for successful glob (embedded path).
                if _emb_tool_results and any(
                    str(r.get("tool_name", "")).lower() == "glob"
                    and r.get("success")
                    and "No files found" not in str(r.get("content", ""))
                    for r in _emb_tool_results
                ):
                    soft_turn_iterations += 1
                    hard_turn_iterations += 1

                if planning_pipeline is not None:
                    activated = self._advance_pipeline(planning_pipeline, event="embedded_tool_calls_completed")
                    if activated:
                        await self.ensure_pipeline_subagent_contexts(
                            planning_pipeline,
                            task_ids=[str(item) for item in activated],
                        )

                # ── Deferred scale init (embedded path) ───────────
                # Skip when a script was executed — the LLM is reading
                # its own output, not fetching external list content.
                if (
                    not getattr(self, "_force_script_mode", False)
                    and self._needs_deferred_scale_init()
                    and not self._turn_has_successful_script_execution(turn_start_idx)
                ):
                    list_task_plan = await self._deferred_scale_init(
                        effective_user_input=effective_user_input,
                        list_task_plan=list_task_plan,
                        turn_usage=turn_usage,
                    )
                    _sp_deferred2 = getattr(self, "_scale_progress", None)
                    _deferred_items2 = _sp_deferred2.get("items", []) if _sp_deferred2 else []
                    _deferred_passthrough2 = (
                        _sp_deferred2 is not None
                        and str(_sp_deferred2.get("_extraction_mode", "")).strip() == "passthrough"
                    )
                    if (
                        _sp_deferred2 is not None
                        and len(_deferred_items2) >= 2
                        and not self._items_are_source_urls_only(_deferred_items2)
                        and not _deferred_passthrough2
                    ):
                        micro_result = await self._run_micro_loop_and_summarize(
                            effective_user_input=effective_user_input,
                            list_task_plan=list_task_plan,
                            turn_usage=turn_usage,
                            session_tool_policy=session_tool_policy,
                            planning_pipeline=planning_pipeline,
                            step_label="deferred_scale_micro_loop_takeover",
                        )
                        if micro_result.get("cancelled"):
                            finalized, final_text, finish_success = await attempt_finalize(
                                output_text=micro_result["summary"],
                                iteration=iteration,
                                finish_success=False,
                            )
                            if finalized:
                                return finish(final_text, success=finish_success)
                            continue
                        # Continue main loop for remaining steps.
                        _sp_cont = getattr(self, "_scale_progress", None)
                        _cont_out = _sp_cont.get("_output_file", "") if _sp_cont else ""
                        completion_feedback = (
                            f"[Scale loop completed] {micro_result.get('processed', 0)} of "
                            f"{micro_result.get('total', 0)} items processed."
                            + (f" Output file: {_cont_out}." if _cont_out else "")
                            + "\n\nNow review the ORIGINAL user request and "
                            "continue with any remaining steps not handled "
                            "by the scale processing (e.g. summarising the "
                            "output, emailing, analysing, charting). If all "
                            "steps are already complete, provide a concise "
                            "final summary."
                        )
                        _post_budget = iteration + 10
                        if soft_turn_iterations < _post_budget:
                            soft_turn_iterations = _post_budget
                        if hard_turn_iterations < _post_budget:
                            hard_turn_iterations = _post_budget
                        stagnant_iterations = 0
                        log.info(
                            "Scale micro-loop done (embedded deferred) — continuing for post-processing",
                            processed=micro_result.get("processed", 0),
                        )
                        continue

                # ── Micro-loop takeover (embedded path) ───────────
                if self._scale_loop_ready():
                    log.info("Scale micro-loop takeover (embedded path)")
                    sp = getattr(self, "_scale_progress", None)
                    output_file = sp.get("_output_file", "") if sp else ""
                    micro_result = await self._run_micro_loop_and_summarize(
                        effective_user_input=effective_user_input,
                        list_task_plan=list_task_plan,
                        turn_usage=turn_usage,
                        session_tool_policy=session_tool_policy,
                        planning_pipeline=planning_pipeline,
                        step_label="scale_micro_loop_takeover",
                        output_file=output_file,
                    )
                    if micro_result.get("cancelled"):
                        finalized, final_text, finish_success = await attempt_finalize(
                            output_text=micro_result["summary"],
                            iteration=iteration,
                            finish_success=False,
                        )
                        if finalized:
                            return finish(final_text, success=finish_success)
                        continue
                    # Continue main loop for remaining steps.
                    _sp_cont = getattr(self, "_scale_progress", None)
                    _cont_out = _sp_cont.get("_output_file", "") if _sp_cont else ""
                    completion_feedback = (
                        f"[Scale loop completed] {micro_result.get('processed', 0)} of "
                        f"{micro_result.get('total', 0)} items processed."
                        + (f" Output file: {_cont_out}." if _cont_out else "")
                        + "\n\nNow review the ORIGINAL user request and "
                        "continue with any remaining steps not handled "
                        "by the scale processing (e.g. summarising the "
                        "output, emailing, analysing, charting). If all "
                        "steps are already complete, provide a concise "
                        "final summary."
                    )
                    _post_budget = iteration + 10
                    if soft_turn_iterations < _post_budget:
                        soft_turn_iterations = _post_budget
                    if hard_turn_iterations < _post_budget:
                        hard_turn_iterations = _post_budget
                    stagnant_iterations = 0
                    log.info(
                        "Scale micro-loop done (embedded path) — continuing for post-processing",
                        processed=micro_result.get("processed", 0),
                    )
                    continue

                # ── Post-write finalization hint (embedded path) ──
                _fa_emb = str(list_task_plan.get("final_action", "")).strip().lower()
                if (
                    _fa_emb == "write_file"
                    and self._turn_has_successful_tool(turn_start_idx, "write")
                    and not completion_feedback
                ):
                    completion_feedback = (
                        "The output file has been written successfully. "
                        "Do NOT write the file again. "
                        "Respond with a brief text summary of what you "
                        "created (file name, key contents). Do NOT call "
                        "any more tools."
                    )
                    log.info("Post-write finalization hint injected (embedded path)")

                if not self._supports_tool_result_followup():
                    output = self._collect_turn_tool_output(turn_start_idx)
                    final = await self._friendly_tool_output_response(
                        user_input=effective_user_input,
                        tool_output=output,
                        turn_usage=turn_usage,
                    )
                    finalized, final_text, finish_success = await attempt_finalize(
                        output_text=final,
                        iteration=iteration,
                        finish_success=True,
                    )
                    if finalized:
                        return finish(final_text, success=finish_success)
                    continue

                # ── Worker finalization for embedded tool calls ────
                # Workers execute a single focused task.  When the LLM
                # response contained text *and* embedded tool-like
                # patterns, the text portion is the actual answer but
                # the embedded extraction prevented the text-only
                # finalization path from being reached.  Try to
                # finalize using the response text so workers don't
                # loop endlessly on spurious embedded tool matches.
                if is_worker and str(response.content or "").strip():
                    finalized, final_text, finish_success = await attempt_finalize(
                        output_text=response.content,
                        iteration=iteration,
                        finish_success=True,
                    )
                    if finalized:
                        return finish(final_text, success=finish_success)

                continue

            # ── Inline command fallback ────────────────────────────
            command = self._extract_command_from_response(response.content)
            if command:
                _tools_executed_count += 1
                _inline_text = str(response.content or "").strip()
                if _inline_text:
                    _intermediate_texts.append(_inline_text)
                    if self._emit_narration(_inline_text, iteration):
                        _narration_streamed = True
                log.info("Executing inline command", command=command)
                self._emit_thinking(f"Running: {command[:60]}", tool="shell", phase="tool")
                try:
                    result = await self._execute_tool_with_guard(
                        name="shell",
                        arguments={"command": command},
                        interaction_label="inline_command",
                        turn_usage=turn_usage,
                        session_policy=session_tool_policy,
                        task_policy=active_task_tool_policy,
                        abort_event=turn_abort_event,
                    )
                    tool_result = result.content if result.success else f"Error: {result.error}"
                except Exception as e:
                    result = None
                    tool_result = f"Error: {str(e)}"

                # If the "command" was rejected as not-a-command (tree diagram,
                # prose, etc.), the LLM response likely already contains the
                # answer.  Use it directly instead of triggering a rewrite.
                if result and not result.success and "not a shell command" in str(result.error or "").lower():
                    original_text = str(response.content or "").strip()
                    if original_text:
                        log.info("Inline command rejected as non-command, using original response text")
                        finalized, final_text, finish_success = await attempt_finalize(
                            output_text=original_text,
                            iteration=iteration,
                            finish_success=True,
                        )
                        if finalized:
                            return finish(final_text, success=finish_success)
                        continue

                self._add_session_message(
                    role="tool",
                    content=tool_result,
                    tool_name="shell",
                    tool_arguments={"command": command},
                )
                self._emit_tool_output("shell", {"command": command}, tool_result)
                if planning_pipeline is not None:
                    activated = self._advance_pipeline(planning_pipeline, event="inline_command_completed")
                    if activated:
                        await self.ensure_pipeline_subagent_contexts(
                            planning_pipeline,
                            task_ids=[str(item) for item in activated],
                        )
                if not self._supports_tool_result_followup():
                    final = await self._friendly_tool_output_response(
                        user_input=effective_user_input,
                        tool_output=tool_result,
                        turn_usage=turn_usage,
                    )
                    finalized, final_text, finish_success = await attempt_finalize(
                        output_text=final,
                        iteration=iteration,
                        finish_success=True,
                    )
                    if finalized:
                        return finish(final_text, success=finish_success)
                    continue

                messages = self._build_messages(
                    tool_messages_from_index=turn_start_idx,
                    query=effective_user_input,
                    planning_pipeline=planning_pipeline,
                    list_task_plan=list_task_plan,
                )
                try:
                    response = await self._complete_with_guards(
                        messages=messages,
                        tools=None,
                        interaction_label="inline_command_followup",
                        turn_usage=turn_usage,
                    )
                    finalized, final_text, finish_success = await attempt_finalize(
                        output_text=response.content,
                        iteration=iteration,
                        finish_success=True,
                    )
                    if finalized:
                        return finish(final_text, success=finish_success)
                    continue
                except Exception:
                    final = await self._friendly_tool_output_response(
                        user_input=effective_user_input,
                        tool_output=tool_result,
                        turn_usage=turn_usage,
                    )
                    finalized, final_text, finish_success = await attempt_finalize(
                        output_text=final,
                        iteration=iteration,
                        finish_success=False,
                    )
                    if finalized:
                        return finish(final_text, success=finish_success)
                    continue

            # ── Stall-retry nudge ─────────────────────────────────
            # Detect when a small/weak model emits an intent-only
            # announcement ("Let me look that up", "I'll fetch the
            # file now.") without calling any tool. We silently
            # re-roll up to ``MAX_STALL_RETRIES`` times with a
            # corrective instruction, and ask the provider to force
            # ``tool_choice="required"`` on the retry when tools are
            # available so the model cannot stall the same way again.
            # The stall text is committed to the session so the model
            # can see (and avoid repeating) what it just emitted.
            _stall_resp_text = str(response.content or "")
            # False-action-claim gate: the reply claims it delegated/sent to a
            # peer, but no flight_deck/consult_peer tool was called THIS TURN —
            # i.e. the model is lying. Force a corrective retry instead of
            # letting the lie reach the user. (Truthful "Poslao sam… čekam"
            # AFTER a real delegate is allowed, since the tool is in turn use.)
            # _turn_tool_call_counts is keyed by "<tool>|<args>" (see the
            # duplicate-call blocker), so match on the tool-name prefix — NOT
            # the bare name, or this never matches and the gate misfires after
            # a real delegate (causing a redundant re-delegation + doubled reply).
            _delegated_this_turn = any(
                str(k).split("|", 1)[0] in ("flight_deck", "consult_peer")
                for k in getattr(self, "_turn_tool_call_counts", {})
            )
            _false_claim = (
                not response.tool_calls
                and not _delegated_this_turn
                and _claims_delegation(_stall_resp_text)
            )
            # False web-research claim: the reply says it searched/fetched the
            # web, but no web tool ran THIS turn — i.e. the model fabricated the
            # research (often pulling from memory instead). Force a corrective
            # retry that actually calls the web tools.
            _WEB_TOOLS = ("web_search", "web_fetch", "web_fetch_batch", "web_get")
            _web_used_this_turn = any(
                str(k).split("|", 1)[0] in _WEB_TOOLS
                for k in getattr(self, "_turn_tool_call_counts", {})
            )
            _false_web_claim = (
                not response.tool_calls
                and not _web_used_this_turn
                and _claims_web_research(_stall_resp_text)
            )
            # A turn that already produced a real DELIVERABLE (a successful
            # write/edit) or a substantive assistant reply is summarizing
            # completed work — its wrap-up must never be treated as a stall OR
            # a false claim. The model isn't evading work or lying; it did the
            # work this turn. Without this guard a perfectly good "✅ Done —
            # saved to <path>" summary gets killed (e.g. the false-web-claim
            # gate misfires because the research happened in an EARLIER turn),
            # and the forced-tool retry derails a weak model into a generic
            # greeting.
            _turn_produced_deliverable = False
            if self.session:
                for _m in self.session.messages[turn_start_idx:]:
                    if (
                        _m.get("role") == "assistant"
                        and len(str(_m.get("content", "")).strip()) > 200
                    ):
                        _turn_produced_deliverable = True
                        break
            if not _turn_produced_deliverable and (
                self._turn_has_successful_tool(turn_start_idx, "write")
                or self._turn_has_successful_tool(turn_start_idx, "edit")
            ):
                _turn_produced_deliverable = True
            if (
                not response.tool_calls
                and not _turn_produced_deliverable
                and (_looks_like_stall(_stall_resp_text) or _false_claim or _false_web_claim)
                and self._stall_retry_count < MAX_STALL_RETRIES
            ):
                self._stall_retry_count += 1
                _has_tools = bool(tool_defs)
                if _false_web_claim:
                    _retry_instruction = (
                        "You claimed you searched/fetched the web, but you did NOT call "
                        "any web tool this turn — that claim is false. Call web_search NOW, "
                        "then web_fetch_batch on the result URLs to read them in parallel. "
                        "Do NOT use memory as a substitute for fresh web data, and never "
                        "claim research you didn't perform."
                    )
                elif _false_claim:
                    _retry_instruction = (
                        "You said you delegated/sent the task to a peer, but you "
                        "did NOT call any tool this turn — that claim is false. To "
                        "hand it off, CALL the flight_deck tool NOW (action='delegate' "
                        "or 'consult', with agent_name and file/message). Otherwise do "
                        "the task yourself. Never claim an action you didn't perform."
                    )
                else:
                    _retry_instruction = (
                        "You announced intent without acting. Do NOT narrate "
                        "what you're about to do. "
                        + (
                            "Call the appropriate tool now to produce the deliverable."
                            if _has_tools
                            else "Produce the final answer now."
                        )
                    )
                log.warning(
                    "Stall detected, silent retry",
                    attempt=self._stall_retry_count,
                    max_retries=MAX_STALL_RETRIES,
                    force_tool=_has_tools,
                    preview=_stall_resp_text[:120],
                )
                self._add_session_message(role="assistant", content=_stall_resp_text)
                self._add_session_message(role="user", content=_retry_instruction)
                # Force tool use on the very next provider call so the
                # retry can't repeat the same intent-only stall. The
                # override is consumed inside the provider payload
                # builder and resets to "auto" automatically.
                if _has_tools:
                    try:
                        setattr(self.provider, "_tool_choice_override", "required")
                    except Exception:
                        pass
                continue

            # ── Tool-avoidance nudge ──────────────────────────────
            # Detect when the LLM dumps actionable content as text
            # instead of using an available tool (e.g. writing out
            # email drafts instead of calling create_draft).  Inject a
            # corrective nudge and re-enter the loop.
            if not getattr(self, "_tool_avoidance_nudged", False):
                _resp_text = str(response.content or "")
                _avail_tool_names = {td.get("name", "") for td in (tool_defs or [])}
                _nudge_msg = None
                if "google_mail" in _avail_tool_names and _re.search(
                    r"(?:\*\*(?:To|Subject):\*\*|\bSubject:\s).+\n.*(?:\*\*(?:To|Subject):\*\*|\bSubject:\s)",
                    _resp_text,
                ):
                    _nudge_msg = (
                        "STOP. You have the google_mail tool available with create_draft action. "
                        "Do NOT output email drafts as text. Call google_mail with action=create_draft "
                        "for EACH recipient right now. Use the to, subject, and body parameters."
                    )
                if _nudge_msg:
                    log.warning("Tool-avoidance detected, nudging LLM", tool="google_mail")
                    self._tool_avoidance_nudged = True
                    self._add_session_message(role="assistant", content=_resp_text)
                    self._add_session_message(role="user", content=_nudge_msg)
                    continue

            # ── No tool calls — final response ────────────────────
            # Guard: a text-only response that is just the model's reasoning /
            # decision *about* answering (not the answer) shouldn't be shipped
            # as the final answer — that's what leaves the user nudging "ok,
            # answer". Nudge once to produce the real answer instead. Scoped to
            # no-tool turns so it can't disrupt a genuine end-of-work summary.
            # Leaked planning/contract JSON as the final answer — catch it and
            # nudge for a plain reply (weak models echo the contract blob).
            if (
                _tools_executed_count == 0
                and not _plan_leak_nudged
                and _looks_like_leaked_plan_json(response.content)
            ):
                _plan_leak_nudged = True
                log.warning(
                    "Leaked plan/contract JSON detected as final response; nudging to answer",
                    preview=str(response.content or "")[:160],
                )
                self._add_session_message(role="assistant", content=str(response.content or ""))
                self._add_session_message(
                    role="user",
                    content=(
                        "That was internal planning data (a task/contract object), NOT a "
                        "response to the user. Never output planning JSON. Answer the user's "
                        "message directly, in plain friendly language."
                    ),
                )
                continue

            if (
                _tools_executed_count == 0
                and not _preamble_nudged
                and _looks_like_planning_preamble(response.content)
            ):
                # Prefer salvaging the real answer hiding behind the preamble
                # over nudging-and-looping: strip the leading meta/decision
                # sentences and, if a substantive answer remains, ship it.
                _stripped = _strip_planning_preamble(str(response.content or ""))
                if _stripped and len(_stripped) > 80:
                    log.info(
                        "Reasoning-preamble stripped; finalizing the remaining answer",
                        kept_preview=_stripped[:160],
                    )
                    finalized, final_text, finish_success = await attempt_finalize(
                        output_text=_stripped,
                        iteration=iteration,
                        finish_success=True,
                    )
                    if finalized:
                        return finish(final_text, success=finish_success)
                _preamble_nudged = True
                log.warning(
                    "Reasoning-preamble detected as final response; nudging to answer",
                    preview=str(response.content or "")[:160],
                )
                self._add_session_message(role="assistant", content=str(response.content or ""))
                self._add_session_message(
                    role="user",
                    content=(
                        "That was your reasoning, not the answer. Give the user the complete "
                        "answer now, directly — don't describe what you're going to do."
                    ),
                )
                continue

            # When tools were executed in previous iterations, combine
            # the intermediate narrative with the final response so the
            # user sees a complete account of what was done rather than
            # just the last LLM output (which may be a generic guide).
            _final_output = response.content
            if _tools_executed_count > 0 and _intermediate_texts and not _narration_streamed:
                # Build combined output: intermediate narrative + final
                _combined_parts = list(_intermediate_texts)
                _final_text_str = str(response.content or "").strip()
                if _final_text_str:
                    _combined_parts.append(_final_text_str)
                _final_output = "\n\n---\n\n".join(_combined_parts)
                log.info(
                    "Combined intermediate + final text",
                    intermediate_count=len(_intermediate_texts),
                    tools_executed=_tools_executed_count,
                    combined_len=len(_final_output),
                )
            log.info(
                "Text-only response (no tool calls), attempting finalize",
                iteration=iteration,
                content_preview=str(_final_output or "")[:200],
            )
            finalized, final_text, finish_success = await attempt_finalize(
                output_text=_final_output,
                iteration=iteration,
                finish_success=True,
            )
            log.info(
                "Finalize result for text-only response",
                finalized=finalized,
                finish_success=finish_success,
                completion_feedback_set=bool(completion_feedback),
                completion_feedback_preview=str(completion_feedback)[:200] if completion_feedback else "",
            )
            if finalized:
                return finish(final_text, success=finish_success)
            continue

        # Hard iteration cap reached
        self._set_runtime_status("waiting")
        return finish("Max iterations reached. Could not complete the request.", success=False)

    # ------------------------------------------------------------------
    # stream() — streaming wrapper
    # ------------------------------------------------------------------

    async def stream(self, user_input: str) -> AsyncIterator[str]:
        """Stream response for user input.

        Args:
            user_input: User's message

        Yields:
            Response chunks
        """
        if not self._initialized:
            await self.initialize()
        self._last_memory_debug_signature = None
        self._last_semantic_memory_debug_signature = None
        self.last_usage = self._empty_usage()

        # Tool-calling and streaming over a single pass is currently limited.
        # Preserve tool behavior and guard checks by using complete() and
        # yielding chunked output when tools/guards are enabled.
        if self.tools.list_tools() or self.guards_enabled():
            self._set_runtime_status("thinking")
            content = await self.complete(user_input)
            chunk_size = 24
            self._set_runtime_status("streaming")
            for idx in range(0, len(content), chunk_size):
                yield content[idx : idx + chunk_size]
            self._set_runtime_status("waiting")
            return

        # Add user message to session
        self._add_session_message("user", user_input)
        await self._auto_compact_if_needed()
        planning_pipeline: dict[str, Any] | None = None
        if self.planning_enabled:
            planning_pipeline = self._build_task_pipeline(user_input)
            self._emit_pipeline_update("created", planning_pipeline)
            await self.ensure_pipeline_subagent_contexts(planning_pipeline)

        # Get tool definitions
        tool_defs = self.tools.get_definitions()

        # For streaming, we currently don't support tool calling
        # This is a limitation - full streaming with tools needs more work
        messages = self._build_messages(query=user_input, planning_pipeline=planning_pipeline)

        # Stream the response
        full_content = ""
        self._set_runtime_status("streaming")
        async for chunk in self.provider.complete_streaming(
            messages=messages,
            tools=tool_defs if tool_defs else None,
        ):
            full_content += chunk
            yield chunk

        # Add assistant response to session. Pull the just-produced
        # reasoning_content (if any) off the provider so DeepSeek
        # thinking mode can round-trip it on the next turn. The
        # provider sets ``last_reasoning_content`` to '' for
        # non-thinking models, so this is a no-op outside thinking
        # mode.
        if self.session:
            _stream_reasoning = getattr(
                self.provider, "last_reasoning_content", ""
            ) or ""
            try:
                self._pending_reasoning_content = _stream_reasoning
            except Exception:
                pass
            self._add_session_message("assistant", full_content)
            await self.session_manager.save_session(self.session)
            memory = getattr(self, "memory", None)
            if memory is not None:
                memory.schedule_background_sync("assistant_stream_saved")

        if planning_pipeline is not None:
            self._finalize_pipeline(planning_pipeline, success=True)

        prompt_tokens = sum(self._count_tokens(m.content) for m in messages)
        completion_tokens = self._count_tokens(full_content)
        self._finalize_turn_usage({
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        })
        self._set_runtime_status("waiting")

    # ── Script-mode credential injection ─────────────────────────

    async def _collect_script_credentials(self) -> str:
        """Collect API keys, passwords, and credentials from config + DB.

        Returns a formatted block to inject into the force-script prompt,
        or an empty string if no credentials are configured.
        """
        cfg = get_config()
        lines: list[str] = []

        # -- Typesense --
        ts = cfg.tools.typesense
        if ts.api_key:
            lines.append("## Typesense")
            lines.append(f"  Host: {ts.host}")
            lines.append(f"  Port: {ts.port}")
            lines.append(f"  Protocol: {ts.protocol}")
            lines.append(f"  API Key: {ts.api_key}")
            if ts.default_collection:
                lines.append(f"  Default Collection: {ts.default_collection}")

        # -- Deep Memory (Typesense-backed) --
        dm = cfg.deep_memory
        if dm.enabled and dm.api_key:
            # Only add if different from the Typesense tool config
            _dm_same = (
                dm.host == ts.host
                and dm.port == ts.port
                and dm.api_key == ts.api_key
            )
            if not _dm_same:
                lines.append("## Deep Memory (Typesense)")
                lines.append(f"  Host: {dm.host}")
                lines.append(f"  Port: {dm.port}")
                lines.append(f"  Protocol: {dm.protocol}")
                lines.append(f"  API Key: {dm.api_key}")
                lines.append(f"  Collection: {dm.collection_name}")

        # -- Web Search (Brave) --
        ws = cfg.tools.web_search
        if ws.api_key:
            lines.append("## Web Search (Brave)")
            lines.append(f"  API Key: {ws.api_key}")
            lines.append(f"  Base URL: {ws.base_url}")

        # -- SendMail --
        sm_cfg = cfg.tools.send_mail
        if sm_cfg.mailgun_api_key:
            lines.append("## Mailgun")
            lines.append(f"  API Key: {sm_cfg.mailgun_api_key}")
            lines.append(f"  Domain: {sm_cfg.mailgun_domain}")
            lines.append(f"  Base URL: {sm_cfg.mailgun_base_url}")
        if sm_cfg.sendgrid_api_key:
            lines.append("## SendGrid")
            lines.append(f"  API Key: {sm_cfg.sendgrid_api_key}")
            lines.append(f"  Base URL: {sm_cfg.sendgrid_base_url}")
        if sm_cfg.smtp_password:
            lines.append("## SMTP")
            lines.append(f"  Host: {sm_cfg.smtp_host}")
            lines.append(f"  Port: {sm_cfg.smtp_port}")
            lines.append(f"  Username: {sm_cfg.smtp_username}")
            lines.append(f"  Password: {sm_cfg.smtp_password}")
            lines.append(f"  TLS: {sm_cfg.smtp_use_tls}")
        if sm_cfg.from_address:
            lines.append("## SendMail (General)")
            lines.append(f"  From Address: {sm_cfg.from_address}")
            if sm_cfg.from_name:
                lines.append(f"  From Name: {sm_cfg.from_name}")

        # -- LLM Provider API Key (for scripts that call LLM APIs) --
        if cfg.model.api_key:
            lines.append("## LLM Provider")
            lines.append(f"  Provider: {cfg.model.provider}")
            lines.append(f"  API Key: {cfg.model.api_key}")
            if cfg.model.base_url:
                lines.append(f"  Base URL: {cfg.model.base_url}")

        # -- Provider keys from settings UI --
        pk = cfg.provider_keys
        if pk.openai:
            lines.append("## OpenAI API")
            lines.append(f"  API Key: {pk.openai}")
        if pk.anthropic:
            lines.append("## Anthropic API")
            lines.append(f"  API Key: {pk.anthropic}")
        if pk.gemini:
            lines.append("## Google Gemini API")
            lines.append(f"  API Key: {pk.gemini}")
        if pk.xai:
            lines.append("## xAI API")
            lines.append(f"  API Key: {pk.xai}")
        if pk.brave:
            lines.append("## Brave Search (Provider Keys)")
            lines.append(f"  API Key: {pk.brave}")

        # -- Telegram bot token --
        if cfg.telegram.enabled and cfg.telegram.bot_token:
            lines.append("## Telegram Bot")
            lines.append(f"  Bot Token: {cfg.telegram.bot_token}")

        # -- Direct API entries from session DB --
        try:
            sm = getattr(self, "session_manager", None)
            if sm:
                api_entries = await sm.list_direct_api_calls(limit=50)
                for entry in api_entries:
                    if entry.auth_token:
                        label = entry.app_name or entry.name or entry.url
                        lines.append(f"## Direct API: {label}")
                        lines.append(f"  URL: {entry.url}")
                        lines.append(f"  Method: {entry.method}")
                        lines.append(f"  Auth Type: {entry.auth_type or 'bearer'}")
                        lines.append(f"  Auth Token: {entry.auth_token}")
                        if entry.headers:
                            lines.append(f"  Headers: {entry.headers}")
        except Exception as e:
            log.debug("Failed to load direct API credentials", error=str(e))

        if not lines:
            return ""

        block = (
            "\n===== AVAILABLE CREDENTIALS =====\n"
            "Use these credentials in your script as needed.\n"
            "Hardcode them directly in the script (this is a local, "
            "single-user environment).\n\n"
            + "\n".join(lines)
            + "\n=================================\n"
        )
        log.info(
            "Force script mode: credentials injected",
            credential_sections=sum(1 for l in lines if l.startswith("## ")),
        )
        return block
