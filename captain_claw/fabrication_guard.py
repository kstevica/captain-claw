"""Replies that report tool work nobody did.

A weak model that sees earlier sends replayed as "Checking… / **Sent** — ID …"
(the calls behind them are not replayed) writes that shape without making
the call: it says an email went out, quotes a Message ID no tool returned,
narrates searches it never ran. The agent loop
(:mod:`captain_claw.agent_orchestration_mixin`) gathers what the turn did into
:class:`TurnFacts` and asks :func:`judge_reply` about each reply before it
ships; the :class:`Verdict` says whether the reply is false, what to tell the
model, and the line to append once the retries are spent. Four checks, one
decision:

* A mail claim (:func:`claims_mail_sent`) with no email out this turn — no
  send that answered "Email sent.", no send relayed by a peer — and no quoted
  Message ID an earlier send returned.
* A calendar claim (:func:`claims_calendar_change`: "Meeting request sent to
  Borjan", "Created — Tue 17:20") with no calendar change this turn — no
  create / update / delete / respond that answered without an error, none
  relayed by a peer — and no quoted event ID an earlier change returned.
* A made-up ID (:func:`fabricated_ids`): a mail / calendar / drive / Graph
  style ID nothing the model was given ever held.
* A narrated lookup (:func:`narrated_lookups`): a turn that made no tool call
  whose reply still says "Searching for the thread…", "Checked the calendar —".
* A WhatsApp send claim (:func:`_whatsapp_claim`): "I messaged Ana on WhatsApp"
  with no ``whatsapp_web`` send that answered "Sent to … on WhatsApp".

The corrective only ever asks the model to make its reply true. It never asks
for a send or a calendar change: whether the user wanted one is theirs to
say, and a second copy, an email or an invitation nobody asked for is worse
than a false "Sent". A flagged reply
costs a full-context call, so the claim detector lets ambiguous lines pass: a
digest of other people's mail or of the user's own sends, a letter drafted for
the user, a file, picture or voice note delivered to the chat, a calendar
invite, a hand-off, the user's own send, a question, a send dated before this
turn. A claim the session's real send may back is told how to prove it, never
that nothing was sent.

Pure functions over text and the facts the caller gathers: no agent state, no
I/O. The model-written lane note cut (:func:`strip_forged_lane_notes`) lives
here too: a forged "[… — from the AUTO lane]" block is how the model dressed
up its false sends as another run's confirmation.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from captain_claw import msg_origin
from captain_claw.gmail_compose import is_sent_repeat_refusal
from captain_claw.mail_authority import strip_automated_prefix

MAIL_TOOLS = frozenset({"google_mail", "microsoft_mail"})
SEND_ACTIONS = frozenset({"send", "send_draft"})
# Correctives per turn for a reply this module turns down, shared by every
# kind. Bounded like the stall retries, on its own counter: a long false
# reply is never exempt as a "deliverable".
MAX_FABRICATION_RETRIES = 2

# Verdict kinds, in the order they lead a corrective; each is also the
# ``origin_detail`` of the rejected reply and its corrective.
MAIL_CLAIM = "false_mail_send"        # says an email went out; none did this turn
CALENDAR_CLAIM = "false_calendar_change"  # says the calendar changed; it did not this turn
NARRATED_LOOKUP = "narrated_lookup"   # no tool call, yet narrates lookups
FABRICATED_ID = "fabricated_id"       # quotes a tool-style ID nothing showed
PAYMENT_CLAIM = "false_payment"       # says something was paid; the ledger doesn't say so
WHATSAPP_CLAIM = "false_whatsapp_send"  # says a WhatsApp message went out; no send did


# ── Tool results ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class ToolStep:
    """One stored tool result: the tool, its ``action`` argument, the text."""

    name: str
    action: str = ""
    arguments: Mapping[str, Any] = field(default_factory=dict)
    content: str = ""

    @property
    def ok(self) -> bool:
        text = self.content.strip().lower()
        return bool(text) and not text.startswith("error")


def tool_steps(messages: Iterable[Mapping[str, Any]]) -> tuple[ToolStep, ...]:
    """The ``role="tool"`` messages of a session slice as :class:`ToolStep`,
    their text without the call's own IDs echoed back (see
    :func:`without_echo`). Monitor rows are left out: an ``llm_trace`` row
    holds the model's own reply, not a tool's answer."""
    steps: list[ToolStep] = []
    for msg in messages:
        if (not isinstance(msg, Mapping) or msg.get("role") != "tool"
                or msg_origin.is_monitor_only(msg)):
            continue
        args = msg.get("tool_arguments")
        args = args if isinstance(args, Mapping) else {}
        steps.append(ToolStep(
            name=str(msg.get("tool_name", "") or "").strip().lower(),
            action=str(args.get("action", "") or "").strip().lower(),
            arguments=args,
            content=without_echo(str(msg.get("content", "") or "").strip(), args),
        ))
    return tuple(steps)


# ── Echoes of the call's own arguments ──
#
# A tool's answer can name the call's arguments back: "No threads found for:
# <query>", "Search results for '<query>'", "Failed to load
# reply_to_message_id=<id>", "Draft <id> not found". An ID the model made up
# and passed in comes back there — its own word, not the tool's, so it proves
# nothing. Only those lines (all of an error) lose it: a get_thread that
# answers with the thread does show the ID it was called with.
_ECHO_LINE_RE = re.compile(
    r"(?i)\b(?:no\s+[\w-]+(?:\s+[\w-]+)?\s+found|not\s+found|(?:search\s+)?results\s+for"
    r"|failed\s+to|could\s*n[o']?t|cannot|can't|invalid|unknown|does\s*n[o']?t\s+exist"
    r"|no\s+such)\b"
)
_ID_ARG_KEY_RE = re.compile(r"(?i)(?:^|_)ids?$")


def _argument_ids(arguments: Any, key: str = "") -> set[str]:
    """The IDs a call's arguments hold: tool-style ones in any value, and
    the whole value of an ``…_id`` argument."""
    found: set[str] = set()
    if isinstance(arguments, Mapping):
        for k, value in arguments.items():
            found |= _argument_ids(value, str(k))
    elif isinstance(arguments, (list, tuple)):
        for value in arguments:
            found |= _argument_ids(value, key)
    elif isinstance(arguments, str) and key != "action":
        found.update(_shaped_ids(arguments))
        value = arguments.strip()
        if _ID_ARG_KEY_RE.search(key) and re.fullmatch(r"[\w-]{3,}", value):
            found.add(value)
    return found


def without_echo(content: str, arguments: Any) -> str:
    """*content* (a tool's answer) without the IDs of its call's *arguments*
    where it only names them back: on its echo lines (see _ECHO_LINE_RE), or
    anywhere in an error."""
    ids = _argument_ids(arguments) if content else set()
    if not ids:
        return content
    pattern = re.compile(
        r"(?<![\w-])(?:" + "|".join(re.escape(t) for t in sorted(ids, key=len, reverse=True))
        + r")(?![\w-])", re.I)
    if content.strip().lower().startswith("error"):
        return pattern.sub("…", content)
    return "\n".join(pattern.sub("…", line) if _ECHO_LINE_RE.search(line) else line
                     for line in content.split("\n"))


# ── Mail claims ──────────────────────────────────────────────────────
#
# Where a claim starts: a line, a sentence ("**Done.** Sent to Ana."), or a
# "Done —" / "OK," / "Yes —" / "Gotovo —" lead-in. A digest line attributed to
# someone else ("- **Calendly**: Confirmation email sent…", "Ana: mail je
# poslan…") never starts one there.
_START = (
    r"(?:^|(?:(?<=[.!?])|(?<=[.!?]\*)|(?<=[.!?]\*\*))[ \t]+"
    r"|\b(?:done|gotovo|ok(?:ay)?|evo|all\s+set)\b(?=[^\w\n])"
    r"|\b(?:yes|da|super|great)\b(?=\s*[,—–!-]))"
)
# Bullets, bold, a dash after a lead-in — not a label, heading or quote.
_PAD = r"[^\w\n#:\"'“”„(\[]{0,6}"
# "**Sent** to Ana", "- Sent to X", "Poslano —", "Emailed JC —", "Replied —".
# The verb must read as a report: closed in bold, or followed by "to"/"na", a
# dash, the sentence end, the end of the line, a Message ID, or the person it
# went to ("poslano Ani", "Emailed JC"). A label ("Sent:", "Sent (2)", "**Sent
# today:**"), "Sent by …" or the Sent folder is about mail, not a send.
_SENT_LINE_RE = re.compile(
    rf"(?im){_START}{_PAD}"
    r"(?:sent|now\s+sent|poslan[oaie]?|replied|odgovoren[oaie]?|e-?mailed)\b"
    r"(?!\**\s*:)(?!\**\s*\(\d)(?!\**\s+(?:by|from|folder|items)\b)"
    r"(?:\*\*"
    r"|(?=\s+(?:to|na)\b|\s*[—–-]|\s*[.!✓✅]|[ \t]*$|\s+(?:message\s+)?id\b)"
    r"|(?:(?<=poslan)|(?<=poslan[oaie])|(?<=mailed)|(?<=replied)|(?<=odgovoren[oaie]))"
    r"(?=\s+(?-i:[A-ZČĆŽŠĐ])))"
)
# The same verb as a label, when the line names who it went to or an ID:
# "**Sent:** one-line ack to JC, ID `…`". A count ("Sent: 3 emails") is not.
_SENT_LABEL_RE = re.compile(
    r"(?im)^[^\w\n]{0,6}(?:sent|poslan[oaie]?|replied|odgovoren[oaie]?)\**[ \t]*:\**"
    r"(?=[^\n]*(?:\bto\s+(?-i:[A-ZČĆŽŠĐ])|\b(?:message\s+)?id\b))"
)
# Who an email went to, between the email and "sent": "to Ana", "za JC". Never
# across a negation — "The email to Ana has not been sent" is no claim.
_NOT_SPAN = r"(?!\b(?:not|never|nije|nisu|nisam)\b|n't\b)"
_TO_SPAN = rf"(?:{_NOT_SPAN}[^\n.,]){{1,40}}?"
_ZA_SPAN = rf"(?:{_NOT_SPAN}[^\n.,]){{1,30}}?"
# The email named as sent: a claim on its own, anchored where a claim
# starts ("Email sent.", "Reply sent to Ana.", "The email to Ana was sent",
# "Mail Ani je poslan", "Mail za JC je poslan", "Odgovor je poslan", "Mail je
# otišao", "the reply is out"). Not mail the user got ("the reply sent by
# Ana", "mail sent to you") or a routine ("is sent every Monday"); "Your
# campaign email was sent…" or "Confirmation email sent…" lead with someone
# else's noun, so they never start one.
_MAIL_NOUN_SENT_RE = re.compile(
    rf"""(?imx){_START}{_PAD}
    (?:(?:the|my|this|that|taj|ovaj)\s+)?
    (?:e-?mail|mail|mejl|reply|odgovor)\s+
    (?:to\s+{_TO_SPAN}\s+|za\s+{_ZA_SPAN}\s+|(?-i:[A-ZČĆŽŠĐ])[\w-]*\s+)?
    (?:
      (?:(?:has\s+been|was|is|je)\s+)?(?:sent|delivered|poslan[oaie]?|otiša[oa])\b
      (?!\s+(?:to\s+(?:you|me)|by|from|every|each|automatically|daily|weekly|once|when|if
             |after|od|svak\w*|automatsk\w*)\b)
     |(?:went|is|je)\s+out\b(?!\s+of\b)
    )
    """
)
# "Emailed JC — …", "I've emailed Ana": mail by the verb itself.
_EMAILED_RE = re.compile(
    rf"(?im){_START}{_PAD}e-?mailed\s+(?:to\s+)?(?-i:[A-ZČĆŽŠĐ])"
    r"|\bi(?:'ve|\s+have|\s+just|\s+already)?\s+(?:just\s+|now\s+|already\s+)?e-?mailed\b"
)
# Sentence forms that need mail words around them: "poslao sam" / "I've
# sent" are just as often a file or a task handed to a peer.
_MAIL_SENT_CLAIM_RE = re.compile(
    rf"""(?imx)
    \bi(?:'ve|\s+have|\s+just|\s+already)?\s+(?:just\s+|now\s+|already\s+)?sent\b
    |\bposla(?:o|la)\s+sam\b
    |\b(?:ack\w*|answer|follow-?up)\s+(?:to\s+{_TO_SPAN}\s+)?
     (?:has\s+been\s+|was\s+|is\s+|went\s+|je\s+)(?:sent|out|delivered|poslan[oaie]?)\b
    """
)
# Replying / confirming in the first person: mail when mail words are around
# or a mail tool ran lately ("Odgovorio sam Gordani ✓", "Potvrdio sam
# Rogeru…", "I've replied to Ana").
_REPLIED_CLAIM_RE = re.compile(
    r"(?i)\b(?:odgovori|potvrdi|javi)(?:o|la)\s+sam\b"
    r"|\bi(?:'ve|\s+have|\s+just|\s+already)?\s+(?:just\s+|already\s+)?replied\s+to\b"
)
# "not sent", "nisam poslao", "once sent", "if sent…", "no mail was sent"
# just before the claim.
_CLAIM_NEGATED_RE = re.compile(
    r"(?i)(?:(?:\bnot\b|n't\b|\bnever\b|\bnothing\b|\bnije\b|\bnisam\b|\bunsent\b|\bif\b"
    r"|\bonce\b|\bbefore\b|\bwhen\b|\bafter\b|\bako\b|\bkad[a]?\b|\bčim\b|\bprije\b|\bnakon\b)"
    r"[^.\n]{0,25}|\bno\s+)$"
)
_MAIL_WORDS_RE = re.compile(
    r"(?i)\b(e-?mail(?:s|ed)?|mails?|gmail|inbox|repl(?:y|ies|ied)|thread(?:ed)?|cc|bcc|subject"
    r"|footer|mejl\w*)\b"
)
# Not an email: a file delivered to the chat ("Sent — `deck.pdf`", "**Sent** —
# one-pager.pdf"), a WhatsApp / Telegram send, a calendar invite.
_NOT_MAIL_TARGET_RE = re.compile(
    r"(?i)`[^`\n]+\.[a-z0-9]{2,5}`|\bwhatsapp\b|\btelegram\b"
    r"|\battendees?\b|\binvit(?:e|es|ation|ations)\b|\bpozivnic\w*|\bsudionic\w*"
    r"|^[^\w\n]*(?:now\s+)?(?:sent|poslan[oaie]?)\W{0,8}[\w.-]+\."
    r"(?:pdf|docx?|xlsx?|pptx?|png|jpe?g|gif|webp|md|csv|txt|zip|html?|mp3|mp4|ogg)\b"
)
# Someone else's line: "- **Ana**: I've sent…", "Marko: sent the contract".
_ATTRIBUTED_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])?\s*\**[^\W\d][^\n:]{0,40}?\**\s*:\**\s+\S")
# A delivery verb after the participle: the email sent to X bounced / reached
# them / arrived — a report about mail, not a send.
_DELIVERY_AFTER_RE = re.compile(
    r"(?i)\b(?:bounced|bounces|reached|arrived|failed|came\s+back|was\s+returned|undeliver\w*"
    r"|vraćen\w*|vratio|vratila|stigao|stigla|nije\s+stig\w*)\b"
)
# A send dated before this turn, right after the verb (or its recipient):
# "sent this morning", "went out on 12 Oct", "sent two days ago", "poslan
# jučer", "poslan prije dva dana", "poslan u petak". A clock time is not one —
# "Sent at 09:53" is as often this turn's made-up send — nor is "minutes ago".
_AGO = (r"(?:\d+|an?|one|two|three|four|five|six|seven|few|a\s+few|couple(?:\s+of)?|several)"
        r"\s+(?:hours?|days?|weeks?|months?)\s+ago")
_HR_WEEKDAY = r"u\s+(?:ponedjeljak|utorak|srijedu|[čc]etvrtak|petak|subotu|nedjelju)"
_EARLIER_AFTER_RE = re.compile(
    r"(?i)^\**\s*(?:to\s+[^\n.,—–]{1,40}?\s+|(?-i:[A-ZČĆŽŠĐ])\w*\s+)?(?:already\s+)?"
    r"(?:this\s+morning|yesterday|earlier(?:\s+today)?|last\s+\w+|" + _AGO +
    r"|on\s+(?:mon|tues|wednes|thurs|fri|satur|sun)day"
    r"|on\s+\d{1,2}(?:st|nd|rd|th)?\.?\s+[a-z]{3,9}|on\s+[a-z]{3,9}\s+\d{1,2}\b"
    r"|jučer|jucer|prekjučer|ranije|jutros|prošl\w+|prije\s+\w+(?:\s+\w+)?|" + _HR_WEEKDAY +
    r"|\d{1,2}\.\s?\d{1,2}\.)"
)
_EARLIER_BEFORE_RE = re.compile(
    r"(?i)\b(?:this\s+morning|yesterday|earlier(?:\s+today)?|last\s+\w+|" + _AGO +
    r"|jučer|jucer|jutros|ranije|prošl\w+|" + _HR_WEEKDAY + r")\b[^.\n]{0,25}$"
)
# The user reports their own send: "I sent your draft to Richard", "poslao
# sam Ani mail". An acknowledgement ("OK, the email to Ted was sent") is not a
# claim of the agent's.
_USER_SENT_RE = re.compile(
    r"(?i)\b(?:i(?:'ve|\s+have|\s+just|\s+already)?\s+(?:just\s+|already\s+)?"
    r"(?:sent|replied|forwarded|e-?mailed)"
    r"|posla(?:o|la)\s+sam|odgovori(?:o|la)\s+sam|proslijedi(?:o|la)\s+sam)\b"
)
# A lead line that lists the user's own sends ("Today you sent two emails:",
# "Your sent mail:", "Poslao si:"): the list or paragraph under it is a digest.
_YOU_SENT_RE = re.compile(
    r"(?i)\byou(?:'ve|\s+have)?\s+(?:also\s+)?sent\b(?!\s+(?:me|us)\b)|\byour\s+sent\b"
    r"|\byou\s+e-?mailed\b|\bposla(?:o|la|li)\s+(?:si|ste)\b(?!\s+(?:mi|nam)\b)"
)
_LIST_ITEM_RE = re.compile(r"[ \t]*(?:[-*•]|\d+[.)])\s")


@dataclass(frozen=True)
class _Claim:
    line_index: int
    earlier: bool
    line_start: int = 0


def _line_bounds(text: str, pos: int) -> tuple[int, int]:
    start = text.rfind("\n", 0, pos) + 1
    end = text.find("\n", pos)
    return start, (end if end >= 0 else len(text))


_CLAUSE_END_RE = re.compile(r"[.!?\n—–:;]")


def _is_question(text: str, end: int) -> bool:
    """The sentence the claim is in ends with "?" and nothing (a dash, a
    colon) sets the claim apart from the question: "Is the reply sent
    already?", "Sent to Ana?" — not "**Sent** to Ana — want me to CC Ivo?"."""
    m = _CLAUSE_END_RE.search(text, end)
    return bool(m and m.group() == "?")


def _in_users_digest(text: str, line_start: int) -> bool:
    """The line at *line_start* hangs from a lead line that lists the user's
    own sends (see _YOU_SENT_RE): the one above its list (past one blank
    line), or its paragraph's first line."""
    above = text[:line_start].split("\n")[:-1]
    lead = ""
    if _LIST_ITEM_RE.match(text, line_start):
        while above and _LIST_ITEM_RE.match(above[-1]):
            above.pop()
        if above and not above[-1].strip():
            above.pop()
        lead = above[-1] if above else ""
    else:
        while above and above[-1].strip():
            lead = above.pop()
    return lead.rstrip(" *_").endswith(":") and bool(_YOU_SENT_RE.search(lead))


def _find_claims(
    text: str, *, request: str = "", recent_mail: bool = False, strict: bool = False,
) -> list[_Claim]:
    """The places *text* says an email was sent.

    The named-email forms ("Email sent.", "Mail je poslan") always count. The
    bare line form ("**Sent** to Ana"), its label variant and the first-person
    reply forms count when the reply or the request is about mail or a mail
    tool ran lately; "I've sent…" / "poslao sam" need mail words. *strict*
    (something else was delivered this turn): only the forms that name the
    email. A letter drafted for the user and a digest of the user's sends
    hold no claim.
    """
    if not text:
        return []
    # Blanked, not cut: line numbers stay the reply's.
    text = _LETTER_RE.sub(lambda m: re.sub(r"[^\n]", " ", m.group()), text)
    patterns: list[re.Pattern[str]] = [_MAIL_NOUN_SENT_RE, _EMAILED_RE]
    if not strict:
        worded = bool(_MAIL_WORDS_RE.search(text) or _MAIL_WORDS_RE.search(request or ""))
        if worded or recent_mail:
            patterns += [_SENT_LINE_RE, _SENT_LABEL_RE, _REPLIED_CLAIM_RE]
        if worded:
            patterns.append(_MAIL_SENT_CLAIM_RE)
    claims: list[_Claim] = []
    seen: set[int] = set()
    for pattern in patterns:
        for match in pattern.finditer(text):
            start = match.start()
            # The lead-in / line start the match opened on is not the claim.
            while start < match.end() and not text[start].isalnum():
                start += 1
            if start in seen or _CLAIM_NEGATED_RE.search(text[max(0, start - 30):start]):
                continue
            line_start, line_end = _line_bounds(text, start)
            head = text[line_start:start]
            if head.strip() and ":" in head and _ATTRIBUTED_RE.match(text[line_start:line_end]):
                continue
            if head.count('"') % 2 or head.count("“") > head.count("”"):
                continue
            if _NOT_MAIL_TARGET_RE.search(text[start:line_end][:120]):
                continue
            if _is_question(text, match.end()) or _in_users_digest(text, line_start):
                continue
            after = text[match.end():line_end]
            if pattern is _MAIL_NOUN_SENT_RE and _DELIVERY_AFTER_RE.search(
                    re.split(r"[.!?](?:\s|$)", after, maxsplit=1)[0]):
                continue
            seen.add(start)
            claims.append(_Claim(
                line_index=text.count("\n", 0, line_start),
                earlier=bool(_EARLIER_AFTER_RE.search(after.lstrip())
                             or _EARLIER_BEFORE_RE.search(head)),
                line_start=line_start,
            ))
    return claims


def claims_mail_sent(
    text: str, request: str = "", recent_mail: bool = False, *, strict: bool = False,
) -> bool:
    """Whether *text* says an email was sent (see :func:`_find_claims`).
    Negated ("not sent"), conditional ("once sent"), questioned, attributed
    to someone else, or about a file / chat / calendar send: not a claim."""
    return bool(_find_claims(text, request=request, recent_mail=recent_mail, strict=strict))


# ── Sends that went out ──────────────────────────────────────────────

# A Message ID a text quotes: "Message ID m-1", "ID `1a2b3c4d5e6f7a8b`",
# "Thread `1a11db7e9c023144`", or a bare backticked Gmail ID.
_QUOTED_ID_RE = re.compile(
    r"(?i)(?P<label>\b(?:message|msg|thread)\s+id\b|\bid\b|\bthread\b)[:\s]*`?(?P<id>[A-Za-z0-9][\w-]{2,})`?"
    r"|`(?P<bare>[0-9a-f]{16})`"
)
_GMAIL_ID_RE = re.compile(r"[0-9a-f]{16}")
# Only a Message ID proves one email: a thread holds every reply in it.
_RESULT_ID_RE = re.compile(r"(?i)\bmessage\s+id:?\s+([A-Za-z0-9][\w-]{2,})")
# Flight Deck's duplicate refusal: "… was already sent at <time> (Gmail
# message <id>). Not sent again."
_FD_DUPLICATE_ID_RE = re.compile(r"\(Gmail message ([A-Za-z0-9][\w-]{2,})\)")
# An ID quoted as a thread's ("threaded on `X`", "Thread ID: X"): Gmail gives
# a new thread its first message's ID, so a later reply in it quotes a sent
# ID without being that send.
_THREAD_LABEL_RE = re.compile(r"(?i)\bthread(?:ed)?(?:\s+(?:id|on|in))?\b[\s:#`'\"(]*$")


def _quoted_ids(text: str) -> list[tuple[str, str]]:
    """``(id, label)`` for each ID *text* quotes (a token with a digit)."""
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for m in _QUOTED_ID_RE.finditer(text or ""):
        token = m.group("id") or m.group("bare") or ""
        if not any(c.isdigit() for c in token) or token in seen:
            continue
        seen.add(token)
        found.append((token, " ".join((m.group("label") or "").lower().split())))
    return found


def _quotes_message_id(token: str, text: str) -> bool:
    """*token* appears in *text*, at least once not labelled as a thread."""
    return any(not _THREAD_LABEL_RE.search(text[max(0, m.start() - 20):m.start()])
               for m in re.finditer(rf"(?<![\w-]){re.escape(token)}(?![\w-])", text or ""))


def mail_send_went_out(content: str) -> bool:
    """A google_mail / microsoft_mail send / send_draft answer that means the
    email went out: "Email sent.", an unknown outcome ("may or may not
    have", "may have gone out"), or a refusal naming the SENT email it
    repeats (the agent's check, or Flight Deck's "already sent at … (Gmail
    message …)"). A repeat of a DRAFT is not one — that email still waits in
    Drafts."""
    text = str(content or "").strip()
    return (text.startswith("Email sent.") or is_sent_repeat_refusal(text)
            or "may or may not have" in text or "may have gone out" in text
            or ("was already sent at" in text and "(Gmail message " in text))


def _is_mcp_mail_write(name: str) -> bool:
    """A mail MCP proxy tool ("mcp_<server>_<tool>") that writes mail — a
    send, a reply, a forward, a draft (tools.mcp_connector decides)."""
    if not name.startswith("mcp_"):
        return False
    from captain_claw.tools.mcp_connector import _MCP_MAIL_HINT, _mcp_mail_write_name

    if not _MCP_MAIL_HINT.search(name):
        return False
    # The server's name and the tool's are joined by "_": the tool's starts
    # at one of the parts.
    parts = name[4:].split("_")
    return any(_mcp_mail_write_name("_".join(parts[i:])) for i in range(len(parts)))


# Of those, the ones that send: "send_message", "send_draft", "reply",
# "reply_all", "forward" — not "create_reply", "draft_reply", "compose…".
_MCP_SEND_WORD_RE = re.compile(r"(?:^|_)send(?:_|$)")
_MCP_REPLY_WORD_RE = re.compile(r"(?:^|_)(?:reply|forward)(?:_|$)")
_MCP_DRAFT_WORD_RE = re.compile(r"(?:^|_)(?:drafts?|compose\w*|create|update|save|upsert)(?:_|$)")


def _other_mail_send(step: ToolStep) -> bool:
    """An email out through send_mail (SMTP) or a mail MCP tool that sends,
    replies or forwards."""
    if not step.ok:
        return False
    if step.name == "send_mail":
        return True
    name = step.name[4:]
    return _is_mcp_mail_write(step.name) and bool(
        _MCP_SEND_WORD_RE.search(name)
        or (_MCP_REPLY_WORD_RE.search(name) and not _MCP_DRAFT_WORD_RE.search(name)))


def _mail_sends(steps: Iterable[ToolStep]) -> list[ToolStep]:
    return [s for s in steps if s.name in MAIL_TOOLS and s.action in SEND_ACTIONS]


# ── Sends another run made ──
#
# A peer's answer (flight_deck / consult_peer), a delegated result or a lane
# note can relay that run's send: "Mailer replied: Email sent. … Message ID X",
# or that run's final reply ("**Sent** to Ana — Message ID X"), which its own
# gate let through. Its tool calls ran in that run's session, so this one
# holds no result of them; the relayed report is the evidence.
RELAY_TOOLS = frozenset({"flight_deck", "consult_peer"})
_RELAYED_SEND_RE = re.compile(r"(?i)(?<!\bno\s)(?<!\bnot\s)\be-?mail\s+sent\b")
_RELAY_ID_SPAN = 600


def _relay_starts(text: str) -> list[int]:
    """Where *text* (another run's) reports a send: after each "Email sent",
    and at each line that claims one this turn. None when it carries a
    backstop ("Note: no email was sent…")."""
    if has_backstop(text):
        return []
    starts = {m.end() for m in _RELAYED_SEND_RE.finditer(text)}
    starts.update(c.line_start for c in _find_claims(text, recent_mail=True) if not c.earlier)
    return sorted(starts)


def relayed_sent_ids(texts: Iterable[str]) -> frozenset[str]:
    """The Message IDs another run's reported send names in *texts* (what
    follows each report, up to the next)."""
    ids: set[str] = set()
    for text in texts:
        text = str(text or "")
        starts = _relay_starts(text)
        for i, start in enumerate(starts):
            end = starts[i + 1] if i + 1 < len(starts) else len(text)
            span = text[start: min(end, start + _RELAY_ID_SPAN)]
            ids.update(token for token, label in _quoted_ids(span)
                       if label in ("message id", "msg id", "id")
                       or (not label and _GMAIL_ID_RE.fullmatch(token)))
    return frozenset(ids)


def _is_handoff(step: ToolStep) -> bool:
    """A peer this run asked or handed work to answered (a fleet listing or
    a spawn answers with no peer's work)."""
    return (step.name in RELAY_TOOLS and step.ok
            and step.action not in ("list_agents", "spawn_agent"))


def _relays_send(step: ToolStep) -> bool:
    """A peer's answer this run got that reports its send."""
    return _is_handoff(step) and bool(_relay_starts(step.content))


def is_relay_message(msg: Mapping[str, Any]) -> bool:
    """Another run's result kept in this session: a lane note, a delegated
    result."""
    return msg_origin.is_lane_note(msg) or (
        str(msg.get("role", "") or "").strip().lower() == "user"
        and msg_origin.origin_of(msg) == "delegated_result")


def turn_sent_mail(steps: Sequence[ToolStep]) -> bool:
    """Whether an email went out in *steps*: a mail send / send_draft that
    answered "Email sent." (or may have), was refused because that email was
    already SENT, went out another mail tool, or a peer this run asked
    reports its send."""
    return (any(mail_send_went_out(s.content) for s in _mail_sends(steps))
            or any(_other_mail_send(s) or _relays_send(s) for s in steps))


def turn_refused_send(steps: Sequence[ToolStep]) -> str:
    """The answer of the last mail send / send_draft in *steps* that did not
    send (sending off, not on the allowlist, no grant, a repeat of a draft,
    an error), or "" when none did."""
    refused = ""
    for step in _mail_sends(steps):
        if not mail_send_went_out(step.content):
            refused = step.content or "Error: the send returned nothing."
    return refused


def sent_message_ids(steps: Iterable[ToolStep]) -> frozenset[str]:
    """The Message IDs of emails that went out: what "Email sent." and a
    sent-repeat refusal name, and what a peer's reported send names."""
    steps = list(steps)
    ids: set[str] = set()
    for step in _mail_sends(steps):
        if step.content.startswith("Email sent.") or is_sent_repeat_refusal(step.content):
            ids.update(_RESULT_ID_RE.findall(step.content))
        ids.update(_FD_DUPLICATE_ID_RE.findall(step.content))
    ids.update(relayed_sent_ids(s.content for s in steps if _relays_send(s)))
    return frozenset(ids)


# ── Other deliveries this turn ──

_CHAT_DELIVERY_TOOLS = frozenset({"write", "edit", "image_gen", "visualize", "pocket_tts", "audio"})
_READ_ONLY_TOOLS = frozenset({
    "read", "glob", "grep", "ls", "web_search", "web_fetch", "web_fetch_batch", "web_get",
    "document_extract", "image_ocr", "summarize_files", "session_history", "codemap",
    "researchmap", "screen_capture", "video_vision", "stt", "cv", "clipboard",
})
_READ_ACTION_RE = re.compile(
    r"^(?:list|search|get|read|info|find|query|fetch|download|describe|show|status|check|lookup|view)"
)


def _delivers_to_chat(step: ToolStep) -> bool:
    """A file, picture or voice note made or sent to the chat: a bare "Sent"
    after it is about that."""
    if not step.ok or step.name in MAIL_TOOLS or _is_mcp_mail_write(step.name):
        return False
    return ("send" in step.name or step.name in _CHAT_DELIVERY_TOOLS
            or "saved/" in step.content)


def _acts_elsewhere(step: ToolStep) -> bool:
    """A successful action that is not mail: a hand-off, a calendar change, a
    post. "Sent to the Researcher" / "Sent to all attendees" after one is about
    that, so recent mail alone no longer makes a bare "Sent" an email."""
    if (not step.ok or step.name in MAIL_TOOLS or step.name in _READ_ONLY_TOOLS
            or _is_mcp_mail_write(step.name)):
        return False
    return not (step.action and _READ_ACTION_RE.match(step.action))


# ── Calendar claims ──────────────────────────────────────────────────
#
# The calendar twin of the mail claim: "Meeting request sent to Borjan — Tue
# 20 Oct, 17:20–18:20", "Created — Tue 20 Oct, 17:20", "I've added it to your
# calendar", "Pomaknuo sam sastanak na 15:00" in a turn whose calendar tool
# changed nothing. Anchored and filtered like the mail forms (a line, a
# sentence or a "Done —" lead-in; not negated, questioned, attributed, quoted,
# dated before this turn, or in a letter drafted for the user). What a
# calendar shows, read back — "The sync is scheduled for 15:00", "the IC was
# moved by Ana", "It's in your calendar" — is not a change: only the agent's
# own voice, a headline, or "now …" counts.

CALENDAR_TOOLS = frozenset({"google_calendar", "microsoft_calendar"})
CALENDAR_WRITE_ACTIONS = frozenset({
    "create_event", "update_event", "delete_event", "respond_event"})

# A calendar MCP proxy tool ("mcp_<server>_<tool>"), and the ones that change
# the calendar ("create_event", "quickAdd", "respond_to_event").
_MCP_CALENDAR_HINT = re.compile(r"(?i)calendar|gcal|event|meeting|booking|schedul")
_MCP_CALENDAR_WRITE_RE = re.compile(
    r"(?:^|_)(?:create|insert|add|quick_?add|update|patch|edit|modify|move|reschedule|delete|remove"
    r"|cancel|respond|accept|decline|book|schedule)(?:_|$)"
)


def is_calendar_tool(name: str) -> bool:
    name = str(name or "").strip().lower()
    return name in CALENDAR_TOOLS or (name.startswith("mcp_")
                                      and bool(_MCP_CALENDAR_HINT.search(name)))


def calendar_tools_in(tools: Iterable[str]) -> frozenset[str]:
    """The calendar tools among *tools* (see :func:`is_calendar_tool`)."""
    return frozenset(t for t in tools if is_calendar_tool(t))


def _snake(name: str) -> str:
    snake = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name or "")
    return re.sub(r"[^A-Za-z0-9]+", "_", snake).strip("_").lower()


def _calendar_writes(steps: Iterable[ToolStep]) -> list[ToolStep]:
    """The calendar calls in *steps* that change it (answered or refused)."""
    return [s for s in steps
            if (s.name in CALENDAR_TOOLS and s.action in CALENDAR_WRITE_ACTIONS)
            or (s.name.startswith("mcp_") and is_calendar_tool(s.name)
                and bool(_MCP_CALENDAR_WRITE_RE.search(_snake(s.name[4:]))))]


# A peer's answer that relays its calendar change ("Created event 'X' on
# Google Calendar.", "Event created", "Invite sent to …").
_RELAYED_CALENDAR_RE = re.compile(
    r"(?i)\b(?:created|updated|deleted|accepted|declined)\s+event\b"
    r"|\b(?:event|meeting|invit\w*)\s+(?:was\s+|has\s+been\s+)?"
    r"(?:created|updated|moved|sent|booked)\b"
)


def _relays_calendar_change(step: ToolStep) -> bool:
    return (_is_handoff(step) and not has_backstop(step.content)
            and bool(_RELAYED_CALENDAR_RE.search(step.content)))


def turn_changed_calendar(steps: Sequence[ToolStep]) -> bool:
    """Whether the calendar changed in *steps*: a calendar create / update /
    delete / respond call that answered without an error, or a peer this run
    asked reports its change."""
    return (any(s.ok for s in _calendar_writes(steps))
            or any(_relays_calendar_change(s) for s in steps))


def turn_refused_calendar(steps: Sequence[ToolStep]) -> str:
    """The answer of the last calendar change in *steps* that did not go
    through (a refusal, an error), or "" when none failed."""
    refused = ""
    for step in _calendar_writes(steps):
        if not step.ok:
            refused = step.content or "Error: the calendar call returned nothing."
    return refused


# The event IDs a create / update answers with ("  ID: …"), and the one a
# delete or an answer to an invitation names ("Deleted event X …").
_EVENT_ID_LINE_RE = re.compile(r"(?m)^[ \t]*ID:[ \t]*([A-Za-z0-9][\w=+/-]{2,})")
_EVENT_ACTED_ID_RE = re.compile(
    r"\b(?:Accepted|Declined|Tentatively accepted|Deleted) event "
    r"([A-Za-z0-9][\w=+/-]{2,}?)(?=[.\s]|$)")


def changed_event_ids(steps: Iterable[ToolStep]) -> frozenset[str]:
    """The IDs of events a calendar change in *steps* went through on."""
    ids: set[str] = set()
    for step in _calendar_writes(steps):
        if step.ok:
            ids.update(_EVENT_ID_LINE_RE.findall(step.content))
            ids.update(_EVENT_ACTED_ID_RE.findall(step.content))
    return frozenset(ids)


# What makes a reply, a request or a line about the calendar.
_CALENDAR_WORDS_RE = re.compile(
    r"(?i)\b(?:calendars?|events?|meetings?|invites?|invitations?|attendees?|guests?|appointments?"
    r"|bookings?|reschedul\w*|kalendar\w*|doga[đd]aj\w*|sastan\w*|termin\w*|pozivnic\w*"
    r"|sudionik\w*)\b"
)
# Words that name the calendar or an event outright: with another action
# this turn (a draft, a file, a hand-off), only these make a claim.
_CALENDAR_NAMED_RE = re.compile(
    r"(?i)\b(?:calendars?|events?|meetings?|appointments?|kalendar\w*|doga[đd]aj\w*|sastan\w*"
    r"|termin\w*)\b"
)
# What an invitation is called, for a "Sent …" line: without it, "sent" is mail's.
_INVITE_RE = re.compile(r"(?i)\binvit\w*|\bmeeting\s+requests?\b|\bpozivnic\w*|\bpoziv\b")
# A date or a time: "17:20", "5pm", "Tue", "20 Oct", "sutra", "20.10.".
_WHEN_RE = re.compile(
    r"(?i)\b\d{1,2}[:.]\d{2}\b|\b\d{1,2}\s*(?:am|pm)\b"
    r"|\b(?:monday|mon|tuesday|tues?|wednesday|thursday|thu(?:rs?)?|friday|fri|saturday|sunday)\b"
    r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2}\b"
    r"|\b\d{1,2}(?:st|nd|rd|th)?\.?\s+"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b"
    r"|\b(?:today|tomorrow|tonight|danas|sutra|prekosutra|ponedjelj\w*|utor\w*|srijed\w*"
    r"|[čc]etvrt(?:ak|ka|kom)|pet(?:ak|ka|kom)|subot\w*|nedjelj\w*)\b"
    r"|\b\d{1,2}\.\s?\d{1,2}\."
)

_CAL_NOUN = (
    r"(?:(?:calendar|google|outlook|teams|zoom)\s+)?"
    r"(?:event|meeting(?:\s+(?:request|invite|invitation))?|invite|invitation|appointment|booking"
    r"|calendar\s+(?:entry|hold|block|item)|hold|call)"
    r"|doga[đd]aj|sastanak|termin|pozivnic[aeu]|poziv"
)
# Made this turn whether bare or after "is" / "was" / "has been".
_CAL_MADE = (
    r"created|added|set\s+up|put\s+in|placed|sent"
    r"|kreiran[aoie]?|dodan[aoie]?|napravljen[aoie]?|upisan[aoie]?|poslan[aoie]?|stavljen[aoie]?"
)
# These also say what a calendar shows ("is scheduled for 15:00", "je
# zakazan"): bare, "now …" or "has been …" only.
_CAL_BOOKED = r"booked|scheduled|reserved|zakazan[aoie]?|rezerviran[aoie]?"
# And these as often report someone else's change ("was moved by Ana", "je
# otkazan"): bare or "now …" only.
_CAL_CHANGED = (
    r"moved|rescheduled|updated|changed|cancell?ed|deleted|removed|accepted|declined"
    r"|pomaknut[aoie]?|premje[šs]ten[aoie]?|preba[čc]en[aoie]?|a[žz]uriran[aoie]?"
    r"|promijenjen[aoie]?|otkazan[aoie]?|obrisan[aoie]?|izbrisan[aoie]?|uklonjen[aoie]?"
    r"|prihva[ćc]en[aoie]?|odbijen[aoie]?"
)
# An invitation or a meeting request "sent": an email out this turn may be it.
_CAL_SENT_RE = re.compile(r"(?i)\b(?:sent|poslan\w*|posla(?:o|la))\b")
# Mail the user got, someone else's change, a routine.
_CAL_NOT_AFTER = (
    r"(?!\s+(?:to\s+(?:you|me|us)\b|by\b|from\b|od\b|every\b|each\b|automatically\b|daily\b"
    r"|weekly\b|svak\w*|automatsk\w*))"
)
# Who or when, between the event and its verb ("with Borjan", "to Ana", "za
# utorak") — never across a copula or a negation: "The meeting with Ana is
# scheduled for 15:00" says what the calendar shows.
_WHO_SPAN = (r"(?:(?!\b(?:is|are|was|were|has|have|had|je|su|got|not|never|nije|nisu|nisam)\b"
             r"|n't\b)[^\n.,]){1,40}?")
# "Meeting request sent to Borjan", "Event created.", "The invite to Ana was
# sent", "Termin je upisan u kalendar", "The meeting is now booked".
_CAL_NAMED_CLAIM_RE = re.compile(
    rf"""(?imx){_START}{_PAD}
    (?:(?:the|my|this|that|your|a|an|new|taj|ovaj|novi)\s+){{0,2}}
    (?:{_CAL_NOUN})\s+
    (?:(?:with|to|for|s|sa|za)\s+{_WHO_SPAN}\s+|(?-i:[A-ZČĆŽŠĐ])[\w-]*\s+)?
    (?:
      (?:(?:is|are|was|were|has\s+been|have\s+been|got|je|su)\s+(?:now\s+|sada?\s+)?)?
      (?P<made>{_CAL_MADE})
     |(?P<now>(?:has\s+been|have\s+been|is\s+now|are\s+now|je\s+sada?)\s+)?(?P<booked>{_CAL_BOOKED})
     |(?P<now2>(?:is\s+now|are\s+now|je\s+sada?)\s+)?(?P<changed>{_CAL_CHANGED})
    )\b{_CAL_NOT_AFTER}
    """
)
# A headline verb: "**Created** — Tue 20 Oct, 17:20", "Moved to 15:00.",
# "Booked for Friday 10:00", "Sent — invite to Borjan", "Dodano u kalendar".
# Not a label ("Created: 2026-10-01", "Updated (2)") or "Created by Ana".
_CAL_BARE_CLAIM_RE = re.compile(
    rf"""(?imx){_START}{_PAD}(?:now\s+)?
    (?P<verb>created|added|booked|scheduled|rescheduled|moved|cancell?ed|deleted|invited|set\s+up
      |placed|sent|kreiran[oa]?|dodan[oa]?|zakazan[oa]?|upisan[oa]?|pomaknut[oa]?|premje[šs]ten[oa]?
      |preba[čc]en[oa]?|otkazan[oa]?|obrisan[oa]?|pozvan[aoi]?|rezervirano|poslan[oaie]?)\b
    (?!\**\s*:)(?!\**\s*\(\d){_CAL_NOT_AFTER}
    """
)
# The same verb as a label, when a date or a time follows: "**Created:** Tue
# 20 Oct, 17:20". An event's own fields ("Created: 2026-10-01T09:12") are not.
_CAL_LABEL_CLAIM_RE = re.compile(
    r"(?im)^[^\w\n]{0,6}(?P<verb>created|booked|scheduled|added|moved|rescheduled|kreirano"
    r"|zakazano|dodano|pomaknuto)\**[ \t]*:\**[ \t]*(?![^\n]*\d{4}-\d{2}-\d{2})(?=[^\n]*\S)"
)
# Of those, the verbs a digest uses for what the calendar shows ("- Moved to
# Thu", "- Cancelled", "- Booked for 10:00"): not a claim in a list item.
_CAL_SETTLED_VERB_RE = re.compile(
    r"(?i)^(?:booked|scheduled|rescheduled|moved|cancell?ed|deleted|zakazan|pomaknut|premje[šs]ten"
    r"|preba[čc]en|otkazan|obrisan|rezervirano)"
)
# The agent's own voice: "I've added it to your calendar", "I moved the sync
# to 15:00", "Zakazao sam sastanak", "Poslao sam pozivnicu Borjanu".
_CAL_I_CLAIM_RE = re.compile(
    r"""(?imx)
    \bi(?:'ve|\s+have|\s+just|\s+already)?\s+(?:just\s+|now\s+|already\s+|also\s+)?
    (?P<verb>created|added|booked|scheduled|rescheduled|moved|cancell?ed|deleted|removed|set\s+up
      |put|placed|sent|invited|accepted|declined|updated|blocked|reserved)\b
    |\b(?P<hr>(?:kreira|doda|zakaza|upisa|stavi|napravi|pomaknu|premjesti|prebaci|otkaza|obrisa
       |izbrisa|posla|pozva|prihvati|odbi|a[žz]urira|rezervira|blokira)(?:o|la)\s+sam
     |sam\s+(?:kreira|doda|zakaza|upisa|stavi|napravi|pomaknu|premjesti|prebaci|otkaza|obrisa
       |izbrisa|posla|pozva|prihvati|odbi|a[žz]urira|rezervira|blokira)(?:o|la))\b
    """
)
# "It's now in your calendar", "Now it's there", "Sad je u kalendaru".
_CAL_NOW_CLAIM_RE = re.compile(
    r"(?i)\bnow\s+(?:it'?s\s+|it\s+is\s+)?(?:in|on)\s+(?:your|the|both)\s+calendars?\b"
    r"|\b(?:it'?s|it\s+is|that'?s|is|are)\s+now\s+(?:in|on)\s+(?:your|the|both)\s+calendars?\b"
    r"|\b(?:in|on)\s+(?:your|the)\s+calendars?\s+now\b"
    r"|\bnow\s+it'?s\s+there\b"
    r"|\bsada?\s+je\s+u\s+(?:tvom\s+|va[šs]em\s+)?kalendar\w*"
)
# What a headline or first-person verb did when it was not the calendar:
# "Created `report.md`", "Added a row to the sheet", "Scheduled the job".
_CAL_OTHER_TARGET_RE = re.compile(
    r"(?i)^[^\w`\n]*(?:`[^`\n]+\.[a-z0-9]{2,5}`|[\w.-]+\.(?:pdf|docx?|xlsx?|pptx?|png|jpe?g|gif"
    r"|webp|md|csv|txt|zip|html?|mp3|mp4|ogg)\b"
    r"|(?:(?:it|them|him|her|ana|\w+)\s+)?(?:(?:to|in|into|on|for|na|u)\s+)?"
    r"(?:(?:the|a|an|your|my|new|that|this|one|another)\s+){0,2}"
    r"(?:sheet|spreadsheet|doc|document|file|folder|draft|reminder|task|to-?do|job|cron|note|label"
    r"|filter|contact|row|tab|list|page|repo|branch|issue|ticket|card|footer|signature|blocker"
    r"|e-?mail|mail|reply|message|post|tablic\w*|dokument\w*|datotek\w*|podsjetnik\w*|zadatak"
    r"|nacrt\w*|mejl\w*|poruk\w*)s?\b)"
)
# A change dated before this turn, right after the verb: "created yesterday",
# "booked this morning", "dodan jučer". (A date or a time after it is the
# event's: "Booked for 20 Oct", "Zakazano 20.10.")
_CAL_EARLIER_AFTER_RE = re.compile(
    r"(?i)^[\s,*—–-]*(?:already\s+)?(?:this\s+morning|yesterday|earlier(?:\s+today)?"
    r"|last\s+(?:week|night|time|month)|" + _AGO + r"|jučer|jucer|prekjučer|ranije|jutros"
    r"|prošl\w+\s+tjed\w+|prije\s+(?:\d+|dva|dvije|tri|par|nekoliko)\s+\w+)\b"
)
# The user reports their own change: "I moved the sync to 3pm", "dodao sam
# termin" — an acknowledgement is no claim of the agent's.
_USER_CAL_RE = re.compile(
    r"(?i)\bi(?:'ve|\s+have|\s+just|\s+already)?\s+(?:just\s+|already\s+)?"
    r"(?:created|added|booked|scheduled|rescheduled|moved|cancell?ed|deleted|accepted|declined"
    r"|invited|sent\s+(?:the\s+|an?\s+)?(?:invite|invitation|meeting\s+request))\b"
    r"|\b(?:kreira|doda|zakaza|upisa|pomaknu|premjesti|otkaza|obrisa|prihvati|odbi|pozva)(?:o|la)"
    r"\s+sam\b"
)


@dataclass(frozen=True)
class _CalendarClaim:
    line_index: int
    earlier: bool
    send: bool        # an invitation "sent": an email out this turn may be it


def _find_calendar_claims(
    text: str, *, request: str = "", recent: bool = False, strict: bool = False,
) -> list[_CalendarClaim]:
    """The places *text* says the calendar was changed.

    The forms that name an event ("Meeting request sent to Borjan", "Event
    created") and "now in your calendar" always count; a headline verb
    ("Created — Tue 20 Oct, 17:20", "Moved to 15:00") and the first-person
    forms count when their line or sentence names the calendar, or carries a
    date or a time and the reply or the request is about the calendar (or a
    calendar tool ran lately). *strict* (something else was done this turn):
    only the ones that name an event or the calendar outright.
    """
    if not text:
        return []
    text = _LETTER_RE.sub(lambda m: re.sub(r"[^\n]", " ", m.group()), text)
    context = recent or bool(_CALENDAR_WORDS_RE.search(text)
                             or _CALENDAR_WORDS_RE.search(request or ""))
    patterns: list[re.Pattern[str]] = [_CAL_NAMED_CLAIM_RE, _CAL_NOW_CLAIM_RE, _CAL_I_CLAIM_RE]
    if not strict:
        patterns += [_CAL_BARE_CLAIM_RE, _CAL_LABEL_CLAIM_RE]
    claims: list[_CalendarClaim] = []
    seen: set[int] = set()
    for pattern in patterns:
        for match in pattern.finditer(text):
            start = match.start()
            while start < match.end() and not text[start].isalnum():
                start += 1
            if start in seen or _CLAIM_NEGATED_RE.search(text[max(0, start - 30):start]):
                continue
            line_start, line_end = _line_bounds(text, start)
            line = text[line_start:line_end]
            head = text[line_start:start]
            if head.strip() and ":" in head and _ATTRIBUTED_RE.match(line):
                continue
            if head.count('"') % 2 or head.count("“") > head.count("”"):
                continue
            if _is_question(text, match.end()):
                continue
            _before, sentence = _sentence_at(text, start)
            send = bool(_CAL_SENT_RE.search(match.group()))
            after = text[match.end():line_end]
            in_list = bool(_LIST_ITEM_RE.match(line))
            # A digest's labelled item ("- New event added: Q3 review").
            if in_list and re.match(r"\**[ \t]*:", after):
                continue
            if pattern is _CAL_NOW_CLAIM_RE:
                if not _CALENDAR_NAMED_RE.search(sentence):
                    continue
            elif pattern is _CAL_NAMED_CLAIM_RE:
                if strict and not _CALENDAR_NAMED_RE.search(match.group()):
                    continue
                if in_list and ((match.group("booked") and not match.group("now"))
                                or (match.group("changed") and not match.group("now2"))):
                    continue
            else:
                # A headline or first-person verb: what it did must be the
                # calendar's, and "sent" an invitation (else it is mail's).
                if _CAL_OTHER_TARGET_RE.match(after):
                    continue
                scope = sentence if pattern is _CAL_I_CLAIM_RE else line
                if send and not _INVITE_RE.search(scope):
                    continue
                if strict and not _CALENDAR_NAMED_RE.search(scope):
                    continue
                if not (_CALENDAR_WORDS_RE.search(scope)
                        or (context and _WHEN_RE.search(scope))):
                    continue
                if (pattern is not _CAL_I_CLAIM_RE and in_list
                        and _CAL_SETTLED_VERB_RE.match(match.group("verb"))):
                    continue
                if pattern is _CAL_LABEL_CLAIM_RE and not _WHEN_RE.search(line):
                    continue
            seen.add(start)
            claims.append(_CalendarClaim(
                line_index=text.count("\n", 0, line_start),
                earlier=bool(_CAL_EARLIER_AFTER_RE.search(after)
                             or _EARLIER_BEFORE_RE.search(head)),
                send=send,
            ))
    return claims


def claims_calendar_change(
    text: str, request: str = "", recent_calendar: bool = False, *, strict: bool = False,
) -> bool:
    """Whether *text* says a calendar event was created, changed, sent or
    answered (see :func:`_find_calendar_claims`)."""
    return bool(_find_calendar_claims(text, request=request, recent=recent_calendar,
                                      strict=strict))


def _acts_off_calendar(step: ToolStep) -> bool:
    """A successful action this turn that is not the calendar's: a draft or
    a send, a file, a hand-off, a sheet edit. "Created — …" after one is
    about that."""
    if is_calendar_tool(step.name) or not step.ok:
        return False
    if step.name in MAIL_TOOLS or _is_mcp_mail_write(step.name):
        return not (step.action and _READ_ACTION_RE.match(step.action))
    return _delivers_to_chat(step) or _acts_elsewhere(step)


# ── Made-up IDs ──────────────────────────────────────────────────────

# Text the model quotes rather than says: fenced code, blockquotes and quoted
# spans hold someone else's words (an email body, the user's message).
_FENCE_RE = re.compile(r"```.*?(?:```|$)", re.S)
_BLOCKQUOTE_RE = re.compile(r"(?m)^[ \t]*>.*$")
_QUOTED_SPAN_RE = re.compile(r"\"[^\"\n]{1,400}\"|“[^”\n]{1,400}”|„[^“”\n]{1,400}[“”]")


def _own_words(text: str) -> str:
    t = _FENCE_RE.sub(" ", str(text or ""))
    t = _BLOCKQUOTE_RE.sub(" ", t)
    return _QUOTED_SPAN_RE.sub('""', t)


# A reply that takes a claim back ("I fabricated it", "that thread does not
# exist", "nisam provjerio"): quoting the made-up ID or the false check is the
# correction, not a new claim (no apology loop).
_RETRACTION_RE = re.compile(
    r"(?i)\b(?:fabricat\w*|invent(?:ed|ing)|made[\s-]up|hallucinat\w*|"
    r"(?:does|did|do)\s*(?:n't|\s+not)\s+exist|never\s+existed|no\s+such|not\s+real|"
    r"wasn'?t\s+real|i\s+was\s+wrong|i\s+apolog\w*|my\s+mistake|"
    r"izmisli\w*|ne\s+postoj\w*|nije\s+postoja\w*|la[žz]n\w*|ispri[čc]\w*|"
    r"gre[šs]k\w*)"
)

# The shapes the google_mail / google_calendar / google_drive and Microsoft
# tools print ("ID: …", "Thread …", "Draft ID: r…", a file id in a Drive URL).
# Each needs digits, so words never match.
_TOKEN_EDGE_L = r"(?<![0-9A-Za-z_\-])"
_TOKEN_EDGE_R = r"(?![0-9A-Za-z_\-])"
_ID_SHAPES: tuple[tuple[str, re.Pattern[str]], ...] = (
    # Gmail message / thread id: 16 lowercase hex (15 for very old mail);
    # also longer hex ids (calendar iCalUIDs, 32-hex event ids).
    ("hex", re.compile(_TOKEN_EDGE_L + r"([0-9a-f]{15,16}|[0-9a-f]{20,64})" + _TOKEN_EDGE_R)),
    # Gmail draft id: r + 15-20 digits (r-123… for negative ids).
    ("draft", re.compile(_TOKEN_EDGE_L + r"(r-?\d{15,20})" + _TOKEN_EDGE_R)),
    # Google Calendar event id: base32hex, often "_"-prefixed, maybe with a
    # recurring-instance suffix "_20261013T080000Z".
    ("event", re.compile(r"(?<![0-9A-Za-z\-])(_?[a-v0-9]{20,}(?:_\d{8}(?:T\d{6}Z)?)?)" + _TOKEN_EDGE_R)),
    # Google Drive file / folder id: 28-44 url-safe chars, mixed case.
    ("drive", re.compile(_TOKEN_EDGE_L + r"([A-Za-z0-9_\-]{28,44})" + _TOKEN_EDGE_R)),
    # Microsoft Graph message / event / conversation id.
    ("graph", re.compile(
        r"(?<![0-9A-Za-z_+/\-])(A[AQ]Mk[A-Za-z0-9+/=_\-]{30,}|AAQk[A-Za-z0-9+/=_\-]{30,})")),
    # OneDrive item id: business "01" + 32 upper-case base32, personal
    # "<16 hex>!<n>".
    ("onedrive", re.compile(
        _TOKEN_EDGE_L + r"(01[A-Z0-9]{32}|[0-9A-Fa-f]{16}!\d{1,9})" + _TOKEN_EDGE_R)),
    # A payment Flight Deck prepared: pay_ + 12 hex.
    ("payment", re.compile(_TOKEN_EDGE_L + r"(pay_[0-9a-f]{12})" + _TOKEN_EDGE_R)),
)


def _id_shape_ok(kind: str, tok: str) -> bool:
    digits = sum(c.isdigit() for c in tok)
    if kind == "hex":
        return digits >= 2 and any(c in "abcdef" for c in tok)
    if kind == "event":
        core = tok.lstrip("_")
        return (digits >= 4 and sum(c.isalpha() for c in core) >= 4
                and not re.fullmatch(r"[0-9a-f]+", core))
    if kind == "drive":
        # Random url-safe chars, not a hyphenated file name ("IC-Agenda-2026-…").
        words = sum(bool(re.fullmatch(r"[A-Z]?[a-z]{2,}|\d{1,4}", seg))
                    for seg in re.split(r"[-_]", tok))
        return (digits >= 2 and any(c.isupper() for c in tok) and any(c.islower() for c in tok)
                and words < 2)
    if kind == "onedrive":
        return digits >= 4
    return True


def _shaped_ids(text: str) -> dict[str, None]:
    """Every token of *text* with an ID's shape, without repeats."""
    out: dict[str, None] = {}
    for kind, pattern in _ID_SHAPES:
        for match in pattern.finditer(str(text or "")):
            tok = match.group(1)
            if _id_shape_ok(kind, tok):
                out.setdefault(tok)
    return out


def quoted_tool_ids(text: str) -> tuple[str, ...]:
    """The tool-style IDs *text* quotes, in order, without repeats."""
    out = list(_shaped_ids(text))
    # A hex id inside a longer id already listed (drive / event) is that id.
    return tuple(t for t in out if not any(t != o and t in o for o in out))


def _line_of(text: str, tok: str) -> str:
    i = text.find(tok)
    if i < 0:
        return ""
    start = text.rfind("\n", 0, i) + 1
    end = text.find("\n", i)
    return text[start: end if end >= 0 else len(text)]


# The request asks for values the model makes up: a password, a token, sample
# data, example code. Its IDs are the answer, not a lookup's result.
_MAKE_UP_REQUEST_RE = re.compile(
    r"(?i)\b(?:generat\w*|random\w*|passwords?|passphrase|secrets?|tokens?|examples?|samples?"
    r"|placeholders?|dummy|fake|mock\w*|snippet|code|script"
    r"|generir\w*|nasumi[čc]n\w*|lozink\w*|primjer\w*|uzor\w*)\b"
)


def fabricated_ids(reply: str, known: Iterable[str], request: str = "") -> tuple[str, ...]:
    """IDs *reply* quotes that appear nowhere in *known* — the texts of every
    tool result, user / system message, lane note and request the model was
    given (never its own replies or tool-call arguments). Case is ignored.

    Not counted: an ID quoted to take it back ("No `1a11…` — I made it up"),
    one in a fenced code block (an example, a listing), and every ID when the
    request asks for made-up values ("generate a password", "example code").
    """
    reply = _FENCE_RE.sub(" ", str(reply or ""))
    if not reply.strip() or _MAKE_UP_REQUEST_RE.search(strip_automated_prefix(request or "")):
        return ()
    tokens = quoted_tool_ids(reply)
    if not tokens:
        return ()
    known_l = [str(text or "").lower() for text in known]
    missing = []
    for tok in tokens:
        if any(tok.lower() in text for text in known_l):
            continue
        line = _line_of(reply, tok)
        if _RETRACTION_RE.search(line) or re.search(r"(?i)\bno\s+[`'\"]?" + re.escape(tok), line):
            continue
        missing.append(tok)
    return tuple(missing)


# ── The ID ledger ──
#
# The IDs the session's tool results (and people's messages, lane notes,
# compaction summaries) showed, kept in the session's metadata as they arrive:
# compaction drops the results, not what the model learned from them. Bounded,
# oldest out first; append-only otherwise. It also keeps the Message IDs sends
# returned, so an earlier send's ID still proves a claim after compaction.

ID_LEDGER_KEY = "fabrication_id_ledger"
_LEDGER_IDS = 1500
_LEDGER_SENT = 200
# A page of hashes (a web fetch, a git log) does not push out the mail IDs.
_LEDGER_PER_MESSAGE = 100


def ledger_ids(text: str) -> tuple[str, ...]:
    """The IDs *text* shows that a reply may quote later: the tool-style
    shapes and labelled ones ("Message ID: m-1", "Thread t-1"). (An ID inside
    a longer one is kept too: the checks match the ledger as text.)"""
    found = _shaped_ids(text)
    for token, _ in _quoted_ids(text):
        found.setdefault(token)
    return tuple(found)


def is_outside_message(msg: Mapping[str, Any]) -> bool:
    """A message whose IDs the model was given from outside itself: a tool
    result (not a monitor row), a person's or an automation's message (not a
    corrective — it quotes made-up IDs back), a lane note, a compaction
    summary or a command's output, a system message."""
    role = str(msg.get("role", "") or "").strip().lower()
    origin = msg_origin.origin_of(msg)
    if role == "tool":
        return not msg_origin.is_monitor_only(msg)
    if role == "user":
        return origin != "corrective" or msg.get("turn_input") is True
    if role == "assistant":
        return origin in ("system_note", "command")
    return role == "system"


SUMMARY_HEAD = "Conversation summary of earlier messages"
# The model's own replies a compaction digest quotes: a "  last reply: "…""
# line and the "→ "…"" end of an "- <date> "<ask>" → "<reply>"" line (to the
# line's end), the same inside the flattened "Before that: …" line (to the
# closing quote: the digest writes replies without double quotes), and a
# fallback summary's "- assistant: …" line (to the next "- <role>:" there).
_DIGEST_REPLY_LINE_RE = re.compile(r'^[ \t]*last reply: ".*$|(?<=") → ".*$|^- assistant: .*$')
_DIGEST_REPLY_INLINE_RE = re.compile(
    r'(?:\blast reply: |→ )"[^"\n]*"?'
    r"|(?<!\S)- assistant: .*?(?=\s- (?:user|assistant|tool|system): |$)"
)


def is_compaction_summary(msg: Mapping[str, Any]) -> bool:
    return (str(msg.get("tool_name", "") or "") == "compaction_summary"
            or msg.get("origin_detail") == "compaction")


def without_digest_replies(text: str) -> str:
    """A compaction summary without the model's replies it quotes (see
    _DIGEST_REPLY_LINE_RE): those are the model's words, not what it was
    given."""
    lines = []
    for line in str(text or "").split("\n"):
        if line.startswith("Before that:"):
            lines.append(_DIGEST_REPLY_INLINE_RE.sub(" ", line))
        else:
            lines.append(_DIGEST_REPLY_LINE_RE.sub("", line))
    return "\n".join(lines)


def outside_text(msg: Mapping[str, Any]) -> str:
    """The text of an outside message (see :func:`is_outside_message`) the ID
    checks may read: a tool result without its call's IDs echoed back, a
    compaction summary without the model's replies it quotes."""
    content = str(msg.get("content", "") or "")
    if str(msg.get("role", "") or "").strip().lower() == "tool":
        args = msg.get("tool_arguments")
        return without_echo(content, args if isinstance(args, Mapping) else {})
    if is_compaction_summary(msg):
        return without_digest_replies(content)
    return content


def ledger_with(ledger: Any, msg: Mapping[str, Any]) -> dict[str, list[str]] | None:
    """*ledger* with the IDs *msg* shows added (and, for a mail send that went
    out — here, or in the run a lane note, a delegated result or a peer's
    answer relays — the Message IDs it returned; for a calendar change that
    went through, its event IDs), or None when nothing is new."""
    if not isinstance(msg, Mapping) or not is_outside_message(msg):
        return None
    found = ledger_ids(outside_text(msg))[:_LEDGER_PER_MESSAGE]
    steps = tool_steps([msg])
    sent = sorted(sent_message_ids(steps) | (
        relayed_sent_ids([str(msg.get("content", "") or "")]) if is_relay_message(msg)
        else frozenset()))
    events = sorted(changed_event_ids(steps))
    if not found and not sent and not events:
        return None
    old = ledger if isinstance(ledger, Mapping) else {}
    ids = [str(i) for i in (old.get("ids") or []) if isinstance(i, str)]
    sent_ids = [str(i) for i in (old.get("sent") or []) if isinstance(i, str)]
    event_ids = [str(i) for i in (old.get("events") or []) if isinstance(i, str)]
    have, have_sent, have_events = set(ids), set(sent_ids), set(event_ids)
    new_ids = [i for i in dict.fromkeys(found) if i not in have]
    new_sent = [i for i in sent if i not in have_sent]
    new_events = [i for i in events if i not in have_events]
    if not new_ids and not new_sent and not new_events:
        return None
    updated = {"ids": (ids + new_ids)[-_LEDGER_IDS:], "sent": (sent_ids + new_sent)[-_LEDGER_SENT:]}
    if event_ids or new_events:
        updated["events"] = (event_ids + new_events)[-_LEDGER_SENT:]
    return updated


def ledger_with_all(
    ledger: Any, messages: Iterable[Mapping[str, Any]],
) -> dict[str, list[str]] | None:
    """*ledger* with every message of *messages* taken in (see
    :func:`ledger_with`), or None when nothing is new."""
    changed = None
    for msg in messages:
        updated = ledger_with(changed if changed is not None else ledger, msg)
        if updated is not None:
            changed = updated
    return changed


def ledger_text(ledger: Any) -> str:
    """The ledger's IDs as one text, for the ID check's known sources."""
    if not isinstance(ledger, Mapping):
        return ""
    return "\n".join(str(i) for i in (ledger.get("ids") or []) if isinstance(i, str))


def ledger_sent_ids(ledger: Any) -> frozenset[str]:
    """The Message IDs earlier sends returned, as the ledger kept them."""
    if not isinstance(ledger, Mapping):
        return frozenset()
    return frozenset(str(i) for i in (ledger.get("sent") or []) if isinstance(i, str))


def ledger_event_ids(ledger: Any) -> frozenset[str]:
    """The event IDs earlier calendar changes returned, as the ledger kept them."""
    if not isinstance(ledger, Mapping):
        return frozenset()
    return frozenset(str(i) for i in (ledger.get("events") or []) if isinstance(i, str))


# ── Narrated lookups ─────────────────────────────────────────────────
#
# What a lookup is aimed at. Most verbs need one of these in their sentence,
# so "Checking your math", "Finding the right word" or "Reading it as a no"
# (thinking, not looking) stay quiet.
_LOOKUP_TARGET_RE = re.compile(
    r"(?i)\b(?:threads?|e-?mails?|mail\w*|gmail|outlook|inbox|drafts?|sent|sends?|outbox|calendar"
    r"|events?|invites?|invitations?|meetings?|bookings?|docs?|documents?|files?|folders?|drive"
    r"|sheets?|spreadsheets?|pages?|sites?|websites?|links?|sources?|web|search(?:es)?|records?"
    r"|database|datastore|tables?"
    r"|crm|fric|notes?|messages?|repl(?:y|ies)|history|logs?|account|agenda|attachments?|contacts?"
    r"|conversation|pipeline|listing|tickets?|repo|memory|acceptance|slots?|availability|schedule"
    r"|deck|context"
    # Croatian stems
    r"|mejl\w*|poruk\w*|nit[iu]?|po[šs]t\w*|kalendar\w*|termin\w*|doga[đd]aj\w*|sastan\w*"
    r"|pozivnic\w*|dokument\w*|datotek\w*|fajl\w*|mapu|tablic\w*|stranic\w*|izvor\w*|bilje[šs]k\w*"
    r"|nacrt\w*|poslan\w*|baz[aeiu]|povijest\w*|razgovor\w*|odgovor\w*|inbox\w*|draft\w*|thread\w*"
    r"|sandu[čc]\w*)\b"
)
# A sentence start: line start (after bullets / markdown / an emoji) or after
# a sentence end.
_LOOKUP_START = (r"(?:^|(?<=[.!?…])[ \t]+|(?<=[.!?…]\*\*)[ \t]+)[ \t>*_•\-–]*"
                 r"(?:[^\w\s\"'“„(\[#]{1,3}[ \t]*)?\**")
# Progress narration ("Searching for…", "Checking for an existing reply
# first.", "Let me find…", "Provjeravam…"), in three strengths. STRONG verbs
# only ever mean an outside lookup and fire alone;
_PROGRESS_STRONG = (
    r"searching|looking\s+(?:for|up|through)|pulling(?:\s+up)?|fetching|scanning|locating|retrieving"
    r"|querying|browsing"
    r"|pretra[žz]ujem|dohva[ćc]am|skeniram|potra[žz]ujem"
)
# ACTION verbs also mean thinking ("Checking your math", "Finding the right
# word"): they need a lookup target in the sentence;
_PROGRESS_ACTION = (
    r"checking|re-?checking|double-checking|cross-checking|verifying|finding|opening|loading|grabbing"
    r"|looking\s+into|digging\s+(?:into|through|up)"
    r"|provjeravam|tra[žz]im|pronalazim|otvaram|u[čc]itavam"
)
# ANALYSIS verbs also lead into an answer about what is already in the chat
# ("Looking at your calendar, you have…"): a target AND no lead-in clause.
_PROGRESS_ANALYSIS = (
    r"reading|re-reading|reviewing|going\s+through|inspecting|examining|looking\s+at"
    r"|[čc]itam|pregledavam|gledam"
)
_LET_ME = (
    r"let\s+me\s+(?:find|check|verify|look\s+(?:for|up|into|through|at)|pull(?:\s+up)?|search|read"
    r"|open|fetch|grab|scan|locate|dig\s+(?:into|through|up)|double-check|cross-check|re-?check"
    r"|query|retrieve|review)"
    r"|daj\s+(?:da|mi\s+da)\s+(?:provjerim|pogledam|na[đd]em|potra[žz]im|pro[čc]itam|otvorim"
    r"|prona[đd]em)"
    r"|samo\s+(?:da|[šs]to)\s+(?:provjerim|pogledam|na[đd]em|potra[žz]im)"
)
_PROGRESS_RE = re.compile(
    r"(?im)" + _LOOKUP_START + r"(?P<verb>" + _PROGRESS_STRONG + r"|" + _PROGRESS_ACTION + r"|"
    + _PROGRESS_ANALYSIS + r"|" + _LET_ME + r")\b"
)
_STRONG_RE = re.compile(r"(?i)^(?:" + _PROGRESS_STRONG + r")\b")
# "Looking for steak, Rovinj isn't the place" leads into an answer as well.
_ANALYSIS_RE = re.compile(r"(?i)^(?:" + _PROGRESS_ANALYSIS + r"|looking\s+for)\b")
_LET_ME_RE = re.compile(r"(?i)^(?:let\s+me|daj|samo)\b")
# "Let me search / look up / fetch / pull up / query / retrieve" only ever
# mean an outside lookup; the other "let me" verbs need a target.
_LET_ME_STRONG_RE = re.compile(
    r"(?i)^let\s+me\s+(?:search|look\s+(?:up|for|through)|fetch|pull\s+up|query|retrieve|locate"
    r"|scan)\b"
)
# Not a lookup: idioms ("checking in", "looking forward", "pulling it all
# together"), labels / table cells ("Finding: …", "Finding 2", "**Searching**:",
# "Finding | Detail |", "**Searching by sender**: type `from:…`"), and a label
# that leads into an instruction ("Checking availability: click 'Find a
# time'.", "Searching by sender: type from:… in the search box.").
_INSTRUCTION = (
    r"(?:type|add|click|tap|use|enter|select|choose|press|paste|scroll|go\s+to|navigate"
    r"|upi[šs]i|dodaj|klikni|koristi|odaberi|izaberi|stisni|pritisni|unesi|idi\s+na)\b"
)
_PROGRESS_NOT_RE = re.compile(
    r"(?i)^(?:checking\s+(?:in|out)\b|reading\s+between\b|looking\s+(?:forward|good|great|ahead)\b"
    r"|pulling\s+(?:(?:it|this|that|everything|things|these)\s+)?(?:all\s+)?together\b"
    r"|pulling\s+back\b"
    r"|searching\s+for\s+(?:a|the\s+right|better|the\s+best)\s+(?:word|way|phrase|term|framing"
    r"|balance)"
    r"|\S+\**\s*[:(|\d]"
    r"|\S+(?:\s+[^\s:.!?*]+){0,4}\*\*\s*:"
    r"|\S+(?:\s+[^\s:.!?*]+){0,4}\**\s*:\s*\**\s*[`'\"“]?" + _INSTRUCTION + r")"
)
# The gerund is the sentence's subject ("Finding the right slot is tricky",
# "Checking the calendar takes a second"): a remark, not a narration.
# Besides the auxiliaries, a general present-tense verb ("Searching Gmail
# works best…", "…lets you…", "…saves time"), or any "-s" word before a word
# a verb takes ("…gives you", "…works well"). Not a result verb — "Checking
# the calendar shows you're free" is a narration.
_GERUND_SUBJECT_RE = re.compile(
    r"(?i)^\S+(?:\s+(?:into|through|up|at|for))?"
    r"(?:\s+(?!(?:whether|if|that|there|what|who|which|when|where|how|to|so)\b)[^\s—–:;,.!?]+){0,6}?"
    r"\s+(?:is|was|isn't|wasn't|are|were|will\s+be|would\s+be|can\s+be"
    r"|takes|took|means|seems|tends|helps|matters|requires|needs|won't|wouldn't|will|would|can't"
    r"|could|should|might|may|je|bi|mo[žz]e|zna[čc]i|tra[žz]i"
    r"|works|lets|makes|saves|costs|depends|involves|uses|gets|goes|beats|narrows|filters|limits"
    r"|catches|misses|skips|keeps|stays|feels|sounds|becomes|fails|pays|wins|hurts|beats"
    r"|(?!(?:shows|returns|reveals|turns|brings|confirms|finds|yields|surfaces|lists|pulls|comes"
    r"|gives)\b)[a-z]+s\s+(?:best|better|well|fine|great|faster|wonders|you|me|us))\b"
)
# A lead-in clause after an analysis verb ("Looking at your calendar, you
# have…"): a comma not followed by more of the narration ("…, then sending").
_LEAD_IN_RE = re.compile(
    r",\s*(?!then\b|and\b|plus\b|too\b|first\b|now\b|so\b|before\b|to\b|pa\b|i\b|zatim\b)\S")
# What the user put in the chat ("the list you sent me", "the thread you
# forwarded", "your message", "above", "koji si poslao"): reading it is no
# lookup.
_IN_CONTEXT_RE = re.compile(
    r"(?i)\b(?:you(?:'ve)?\s+(?:just\s+)?(?:pasted|shared|uploaded|forwarded|attached)"
    r"|you(?:'ve)?\s+(?:just\s+)?(?:sent|gave|wrote|listed)\s+(?:me\b|here\b|above\b"
    r"|in\s+(?:this\s+)?chat)"
    r"|(?:the\s+)?(?:above|below)\b|pasted|your\s+(?:message|text|question|list|voice\s+note"
    r"|screenshot|photo|image)\b|si\s+mi\s+(?:posla|zalijepi|proslijedi|napisa)\w*"
    r"|(?:si|ste)\s+(?:zalijepi|podijeli|proslijedi)\w*|tvoj\w*\s+(?:poruk|tekst|pitanj)\w*"
    r"|gore\b|ispod\b)"
)
# When the request carries pasted content, reading or reviewing a text
# ("Reviewing your draft:", "Pročitao sam dokument") is about it: an analysis
# verb, or a past-tense look at nothing but these, is no lookup. A store
# (inbox, Sent, calendar, thread) still is, and so are search / check verbs.
_PASTED_TARGET_RE = re.compile(
    r"(?i)draft|e-?mail|mail|docs?|documents?|notes?|nacrt\w*|dokument\w*|mejl[a-z]?|bilje[šs]k\w*")
_READ_VERB_RE = re.compile(
    r"(?i)^(?:" + _PROGRESS_ANALYSIS + r"|let\s+me\s+(?:read|review|look\s+at))\b")
_BRACKET_LINE_RE = re.compile(r"(?m)^[ \t]*[\[(][^\n]*[\])][ \t]*$")


def has_pasted_content(request: str) -> bool:
    """*request* carries content to read, not only an ask: a fenced or
    quoted block, or two lines or more of 200+ characters (attachment lines
    aside — an attached file is read with a tool)."""
    text = _BRACKET_LINE_RE.sub("", strip_automated_prefix(request or "")).strip()
    if "```" in text or _BLOCKQUOTE_RE.search(text):
        return True
    return len(text) >= 200 and sum(bool(line.strip()) for line in text.split("\n")) >= 2


def _about_pasted(sentence: str) -> bool:
    targets = [m.group() for m in _LOOKUP_TARGET_RE.finditer(sentence)]
    return bool(targets) and all(_PASTED_TARGET_RE.fullmatch(t) for t in targets)


# A routine, not a step now ("- Checking the inbox daily at 09:00").
_ROUTINE_RE = re.compile(
    r"(?i)\b(?:daily|hourly|weekly|nightly|every\s+(?:day|morning|evening|night|hour|week"
    r"|\d+\s*\w+|mon|tue|wed|thu|fri|sat|sun)\w*|each\s+(?:day|morning|evening|hour|week)"
    r"|on\s+a\s+schedule|svak\w*\s+(?:dan|jutro|ve[čc]er|sat|tjedan)\w*|dnevno|tjedno)\b"
)
# Past-tense claims of a lookup: "I checked drafts and sent mail", "Opened
# the event (…)", "Re-confirmed from the calendar", "Provjerio sam mail",
# "the sent-mail check returns it".
_CLAIM_I_RE = re.compile(
    r"(?i)\bI(?:'ve|\s+have|\s+had)?\s+(?:just\s+|already\s+|also\s+|first\s+|now\s+|quickly\s+)?"
    r"(?P<verb>checked|searched|looked\s+(?:up|through|into|for|at|in)|verified|went\s+through"
    r"|pulled(?:\s+up)?|scanned|reviewed|opened|fetched|double-checked|cross-checked|re-?checked"
    r"|dug\s+(?:into|through)|queried|browsed|read|had\s+a\s+(?:quick\s+)?look"
    r"|took\s+a\s+(?:quick\s+)?look"
    r"|ran\s+(?:a\s+|the\s+)?(?:quick\s+)?(?:search|query|check|lookup))\b"
)
_CLAIM_BARE_RE = re.compile(
    r"(?m)" + _LOOKUP_START
    + r"(?P<verb>(?i:opened|checked|searched|scanned|pulled|re-?checked|re-?confirmed"
    r"|double-checked|cross-checked|went\s+through|looked\s+(?:up|through|into|at|in)|fetched"
    r"|reviewed|verified|dug\s+(?:into|through)|queried|had\s+a\s+(?:quick\s+)?look"
    r"|took\s+a\s+(?:quick\s+)?look"
    r"|ran\s+(?:a\s+|the\s+)?(?:quick\s+)?(?:search|query|check|lookup)))\b(?!\**\s*[:(])"
)
_CLAIM_VERIFIED_DASH_RE = re.compile(
    r"(?m)" + _LOOKUP_START + r"(?P<verb>(?i:verified|provjereno))\**\s*[—–-]")
_CLAIM_RESULT_RE = re.compile(
    r"(?i)\b(?P<verb>(?P<scope>sent-mail\s+|sent\s+mail\s+|inbox\s+|calendar\s+|drive\s+|gmail\s+"
    r"|mail\s+|drafts?\s+)?(?:search(?:es)?|check|look-?up|query))\b"
    r"(?:[^.\n]|\.(?=\S)){0,60}?\b(?:returns?|returned|shows?|showed|found|turns?\s+up|turned\s+up"
    r"|comes?\s+back|came\s+back)\b"
)
_CLAIM_HR_RE = re.compile(
    r"(?i)\b(?P<verb>(?:provjeri|pogleda|pretra[žz]i|pro[čc]ita|otvori|pregleda|dohvati)(?:o|la|li)"
    r"\s+sam"
    r"|(?:prona[šs]|na[šs])(?:ao|la|li)\s+sam"
    r"|sam\s+(?:(?:provjeri|pogleda|pretra[žz]i|pro[čc]ita|otvori|pregleda|dohvati)(?:o|la)"
    r"|(?:prona|na)[šs](?:ao|la))"
    r"|provjereno\s+(?:u|na|iz)\b)"
)
# Before the claim, in its sentence: negated, reported, or conditional.
_LOOKUP_GUARD_RE = re.compile(
    r"(?i)(?:\b(?:not|never|no|without|haven't|hadn't|didn't|wasn't|couldn't|can't|cannot|if|once"
    r"|when|after|before|said|claimed|wrote|told|says|nisam|nije|ne|bez|ako|kad)\b[^.!?\n]{0,30})$"
)
# A claim about an earlier turn's look ("I checked the thread earlier").
# "Nothing new since my last check" is not one: "since" compares a look now
# with an earlier one.
_LOOKUP_EARLIER_RE = re.compile(
    r"(?i)(?<!since\s)(?<!since\smy\s)(?<!since\sthe\s)(?<!od\s)\b(?:earlier|yesterday"
    r"|last\s+(?:time|night|week)|before|previously|this\s+morning|an\s+hour\s+ago|my\s+last\s+check"
    r"|last\s+check|ranije|ju[čc]er"
    r"|(?:back\s+)?then(?=\s*[)\].,;!?]|\s*$)"
    r"|prije|jutros|sino[ćc])\b"
)
# A sentence ends at . ! ? … before a space, a closing mark or the end — not
# inside an address, a URL or "13.30" — or at a line break.
_SENT_END_RE = re.compile(r"[.!?…](?=[\s*_)\]\"'”`]|$)|\n")


def _sentence_at(text: str, start: int) -> tuple[str, str]:
    """(text before *start* in its sentence, the sentence from *start*)."""
    left = 0
    for match in _SENT_END_RE.finditer(text, 0, start):
        left = match.end()
    match = _SENT_END_RE.search(text, start)
    end = match.end() if match else len(text)
    return text[left:start], text[start:end]


def _line_start(text: str, pos: int) -> str:
    return text[text.rfind("\n", 0, pos) + 1: pos]


# A letter the reply drafts for the user, from its greeting line to its
# sign-off ("Hi Ana,\n\nI checked my calendar and Friday works.\n\nBest,\nK"):
# its "I" is the user's.
_LETTER_RE = re.compile(
    r"(?ims)^[ \t>*_]*(?:hi|hello|hey|dear|hej|bok|zdravo|po[šs]tovan\w*|drag[aio])\b[^\n]{0,40}?,"
    r"[^\n]*$.*?^[ \t*_]*(?:best|regards|kind\s+regards|best\s+regards|warm\s+regards|thanks"
    r"|thank\s+you|cheers|sincerely|lp|lijep\w*\s+pozdrav\w*|srda[čc]n\w*|pozdrav\w*|hvala)\b"
    r"[^\n]{0,40}$"
)


def _denies_lookup(text: str) -> bool:
    """A retraction that says the lookup never happened ("I didn't actually
    check — that thread doesn't exist")."""
    return bool(_RETRACTION_RE.search(text) and re.search(
        r"(?i)\b(?:didn'?t|did\s+not|haven'?t|have\s+not|never|nisam)\s+(?:actually\s+|really\s+)?"
        r"(?:check|search|look|open|read|verify|pull|fetch|provjeri|pogleda|pretra[žz]i|otvori)",
        text))


def narrated_lookups(reply: str, pasted: bool = False) -> tuple[str, ...]:
    """The lookups *reply* narrates as done or under way, for a turn that made
    no tool call: progress lines ("Searching for the thread…", "Provjeravam
    inbox."), "Let me find X." followed by what it "found", and past-tense
    claims ("I checked drafts", "Opened the event (…)", "Provjerio sam mail").

    Not counted: offers ("I can check…", "Want me to look?"), plans ("I'll
    check"), negations ("I haven't checked"), quotes of other people's words,
    routines ("checking daily"), reading what the user put in the chat (with
    *pasted*, the request's content: see _PASTED_TARGET_RE), an earlier
    turn's check, a bare "Let me check." (the stall gate's), a letter drafted
    for the user, and a retraction.
    """
    text = _LETTER_RE.sub(" ", _own_words(reply))
    if not text.strip() or _denies_lookup(text):
        return ()
    hits: list[str] = []
    for match in _PROGRESS_RE.finditer(text):
        verb_start = match.start("verb")
        if _line_start(text, verb_start).lstrip().startswith("#"):
            continue  # a heading
        _before, sentence = _sentence_at(text, verb_start)
        if sentence.rstrip().endswith("?") or len(sentence) > 220:
            continue
        if _PROGRESS_NOT_RE.match(sentence) or (
                not _LET_ME_RE.match(sentence) and _GERUND_SUBJECT_RE.match(sentence)):
            continue
        if _IN_CONTEXT_RE.search(sentence) or _ROUTINE_RE.search(sentence) or (
                pasted and _READ_VERB_RE.match(sentence) and _about_pasted(sentence)):
            continue
        if _LET_ME_RE.match(sentence):
            # "Let me check." with nothing after it is a stall; narration is
            # followed by what the look found.
            if len(text[verb_start + len(sentence):].strip()) < 40:
                continue
            if not _LET_ME_STRONG_RE.match(sentence) and not _LOOKUP_TARGET_RE.search(sentence):
                continue
        else:
            if not _STRONG_RE.match(sentence) and not _LOOKUP_TARGET_RE.search(sentence):
                continue
            if _ANALYSIS_RE.match(sentence) and _LEAD_IN_RE.search(sentence):
                continue
        hits.append(sentence.strip()[:80])
    for pattern in (_CLAIM_I_RE, _CLAIM_BARE_RE, _CLAIM_HR_RE):
        for match in pattern.finditer(text):
            if _line_start(text, match.start("verb")).lstrip().startswith("#"):
                continue
            before, sentence = _sentence_at(text, match.start("verb"))
            if _LOOKUP_GUARD_RE.search(before) or sentence.rstrip().endswith("?"):
                continue
            whole = before + sentence
            if (_LOOKUP_EARLIER_RE.search(whole) or not _LOOKUP_TARGET_RE.search(whole)
                    or _IN_CONTEXT_RE.search(whole) or (pasted and _about_pasted(whole))):
                continue
            hits.append(sentence.strip()[:80])
    for pattern in (_CLAIM_VERIFIED_DASH_RE, _CLAIM_RESULT_RE):
        for match in pattern.finditer(text):
            before, sentence = _sentence_at(text, match.start("verb"))
            whole = before + sentence
            if (_LOOKUP_GUARD_RE.search(before) or _LOOKUP_EARLIER_RE.search(whole)
                    or _IN_CONTEXT_RE.search(whole)):
                continue
            if pattern is _CLAIM_RESULT_RE and not match.group("scope") and not (
                    _LOOKUP_TARGET_RE.search(re.sub(r"(?i)\bsearch(?:es)?\b", " ", whole))):
                continue
            hits.append(sentence.strip()[:80])
    return tuple(dict.fromkeys(hits))


# ── Payment claims ───────────────────────────────────────────────────
#
# Captain never moves money: the `payments` tool prepares a payment and hands
# over a barcode; the user approves it in their bank app. So the agent saying
# *it* paid is always false, and saying a bill *is* paid needs the ledger to
# say so — a `payments` result showing ``status: user_paid`` (the user marked
# it) or ``settled``. Judged only while the turn offers the payments tool.

PAYMENT_TOOLS = frozenset({"payments"})

# The agent says it paid: "I paid", "I've transferred", "I sent the money",
# "platio sam", "račun sam platio", "izvršio sam plaćanje".
_PAY_I_CLAIM_RE = re.compile(
    r"(?i)\bI(?:'ve|\s+have|\s+just|\s+already)?\s+(?:just\s+|already\s+|now\s+)?"
    r"(?:paid|transferred|wired|sent\s+(?:the\s+|your\s+)?(?:money|payment|funds|transfer))\b"
    r"|\b(?:platio|platila|uplatio|uplatila|prebacio|prebacila)\s+sam\b"
    r"|\bsam\s+(?:ve[ćc]\s+)?(?:platio|platila|uplatio|uplatila|prebacio|prebacila)\b"
    r"|\bizvr[šs]i(?:o|la)\s+sam\s+(?:pla[ćc]anje|uplatu|transakciju|prijenos)\b"
    r"|\bposla(?:o|la)\s+sam\s+(?:novac|uplatu|lovu)\b"
)
# A bill or payment said to be paid / done.
_PAY_STATE_CLAIM_RE = re.compile(
    r"(?i)\b(?:invoice|bill|payment|transfer)s?\s+(?:is|was|were|are|has\s+been|have\s+been)\s+"
    r"(?:now\s+|already\s+|successfully\s+)?(?:paid|made|sent|completed|done|processed|settled)\b"
    r"|\bpayment\s+(?:went\s+through|succeeded|is\s+successful|complete[d]?)\b"
    r"|\b(?:ra[čc]un|uplatnica|pla[ćc]anje|uplata|transakcija|prijenos)\w*\s+(?:je\s+)?(?:sada\s+|ve[ćc]\s+)?"
    r"(?:pla[ćc]en\w*|izvr[šs]en\w*|pro[šs]l\w*|obavljen\w*|upla[ćc]en\w*)\b"
    r"|\bstatus\W{0,3}(?:paid|pla[ćc]eno|user_paid)\b"
    r"|^[^\w\n]*(?:paid|pla[ćc]eno|upla[ćc]eno)[^\w\n]*$"
)
# The user paid (or marked it): "you paid", "you marked it paid", "platio si".
_PAY_USER_RE = re.compile(
    r"(?i)\byou(?:'ve|\s+have)?\s+(?:already\s+|just\s+)?(?:paid|marked)\b|\bmarked\s+(?:it\s+)?as\s+paid\b"
    r"|\b(?:platio|platila|uplatio|uplatila|ozna[čc]io|ozna[čc]ila)\s+si\b"
)
_PAID_EVIDENCE_RE = re.compile(r"\bstatus: (?:user_paid|settled)\b")
# A purchase Captain itself paid (Phase 3: one-time card, approved by the user)
# that the checkout confirmed — the one case where "I paid / I ordered" is true.
_PURCHASED_EVIDENCE_RE = re.compile(r"\bpurchase_intent: pur_[0-9a-f]{12} status: (?:ordered|completed)\b")
# "I bought / ordered / placed the order", "kupio sam", "naručio sam".
_PURCHASE_CLAIM_RE = re.compile(
    r"(?i)\bI(?:'ve|\s+have|\s+just|\s+already)?\s+(?:just\s+|already\s+)?(?:bought|purchased|ordered"
    r"|placed\s+(?:the|an|your)\s+order|checked\s+out)\b"
    r"|\b(?:kupio|kupila|naru[čc]io|naru[čc]ila)\s+sam\b|\bsam\s+(?:kupio|kupila|naru[čc]io|naru[čc]ila)\b"
    r"|\b(?:order|narud[žz]ba)\s+(?:is|was|has\s+been|je)\s+(?:now\s+)?(?:placed|confirmed|complete[d]?"
    r"|zaprimljen\w*|potvr[đd]en\w*|poslan\w*)\b"
)


def _payment_evidence(steps: Iterable[ToolStep]) -> bool:
    return any(step.name in PAYMENT_TOOLS and _PAID_EVIDENCE_RE.search(step.content) for step in steps)


def _purchase_evidence(steps: Iterable[ToolStep]) -> bool:
    return any(step.name in PAYMENT_TOOLS and _PURCHASED_EVIDENCE_RE.search(step.content) for step in steps)


def _payment_claim(reply: str, facts: TurnFacts) -> tuple[frozenset[int], bool]:
    """(lines that say something was paid, whether any of them is false)."""
    if not facts.payment_tools:
        return frozenset(), False
    text = str(reply or "")
    purchased = _purchase_evidence(facts.turn_steps) or _purchase_evidence(facts.session_steps)
    evidence = purchased or _payment_evidence(facts.turn_steps) or _payment_evidence(facts.session_steps)
    lines: set[int] = set()
    false = False
    for pattern, first_person in ((_PAY_I_CLAIM_RE, True), (_PAY_STATE_CLAIM_RE, False),
                                  (_PURCHASE_CLAIM_RE, True)):
        for match in pattern.finditer(text):
            start, end = _line_bounds(text, match.start())
            line = text[start:end]
            before = text[start:match.start()]
            if (_CLAIM_NEGATED_RE.search(before) or _is_question(text, match.end())
                    or _ATTRIBUTED_RE.match(line) or _PAY_USER_RE.search(line)
                    or before.count('"') % 2 == 1 or before.count("“") > before.count("”")):
                continue
            lines.add(text.count("\n", 0, start))
            # Captain pays only for a purchase the user approved and the
            # checkout confirmed; a bill is paid only when the ledger says so.
            if (first_person and not purchased) or not evidence:
                false = True
    return frozenset(lines), false


def payment_corrective() -> str:
    """What the model is told about a reply that says something was paid
    when nothing it can see says so. It asks for a true reply."""
    return ("Your reply says a payment was made, an order placed or a bill paid, but nothing in this "
            "conversation shows it. Captain pays only for a purchase the user approved and whose checkout "
            "confirmed the order; a bill counts as paid only once the user says so or their bank confirms "
            "it. Rewrite your reply so it is true: say what is prepared or waiting for the user's approval.")


PAYMENT_BACKSTOP = ("Note: Captain did not make any payment — it can only prepare one for you to "
                    "approve in your bank app.")


# ── WhatsApp sends ───────────────────────────────────────────────────
#
# The ``whatsapp_web`` tool sends one text to a whitelisted chat and answers
# "Sent to <chat> on WhatsApp …" — the only proof a message went out. A line
# that mentions WhatsApp and says the agent sent / messaged / replied (or that
# a message was sent) needs that proof somewhere in the conversation. Judged
# only while the turn offers ``whatsapp_web``. A line about delivering to the
# user ("sent you the file on WhatsApp" — the chat channel does that) and a turn
# where ``whatsapp_send_file`` delivered something are left alone.

WHATSAPP_TOOLS = frozenset({"whatsapp_web"})
_WA_WORD_RE = re.compile(r"(?i)\bwhats\s?app\b")
_WA_I_CLAIM_RE = re.compile(
    r"(?i)\bI(?:'ve|\s+have|\s+just|\s+already)?\s+(?:just\s+|already\s+|now\s+)?"
    r"(?:sent|messaged|texted|replied|wrote(?:\s+to)?|pinged|answered)\b"
    r"|\b(?:poslao|poslala|napisao|napisala|odgovorio|odgovorila|javio|javila)\s+sam\b"
    r"|\bsam\s+(?:ve[ćc]\s+)?(?:poslao|poslala|napisao|napisala|odgovorio|odgovorila|javio|javila)\b"
)
_WA_STATE_CLAIM_RE = re.compile(
    r"(?i)\b(?:message|reply|text|answer)s?\b(?:\s+[^\s.,;:!?]+){0,4}?\s+"
    r"(?:is|was|were|has\s+been|have\s+been)\s+(?:now\s+|already\s+|successfully\s+)?(?:sent|delivered)\b"
    r"|^[^\w\n]*sent\b"
    r"|\b(?:poruk[ae]|odgovor)\s+(?:je\s+|su\s+)?(?:poslan\w*|isporu[čc]en\w*)\b"
)
# Delivered to the user themself, through the chat they're talking in: "sent
# you …" right after the verb, or "here" / "this chat" on the line.
_WA_TO_USER_AFTER_RE = re.compile(r"(?i)\s*(?:(?:it|this|that|the\s+\w+)\s+)?(?:to\s+)?(?:you|ti|vam|tebi)\b")
_WA_HERE_RE = re.compile(r"(?i)\bhere\b|\bthis\s+chat\b|\bovdje\b|\bu\s+ovaj\s+chat\b")
_WA_EVIDENCE_RE = re.compile(r"(?m)^Sent to .{1,200} on WhatsApp\b")


def _whatsapp_evidence(steps: Iterable[ToolStep]) -> bool:
    return any(step.name in WHATSAPP_TOOLS and step.action == "send" and _WA_EVIDENCE_RE.search(step.content)
               for step in steps)


def _whatsapp_claim(reply: str, facts: TurnFacts) -> tuple[frozenset[int], bool]:
    """(lines that say a WhatsApp message went out, whether any is false)."""
    if not facts.whatsapp_tools:
        return frozenset(), False
    if any(step.name == "whatsapp_send_file" for step in facts.turn_steps):
        return frozenset(), False
    lines = _whatsapp_claim_lines(str(reply or ""))
    evidence = _whatsapp_evidence(facts.turn_steps) or _whatsapp_evidence(facts.session_steps)
    return lines, bool(lines) and not evidence


def claims_whatsapp_sent(text: str) -> bool:
    """Whether *text* says a WhatsApp message went out."""
    return bool(_whatsapp_claim_lines(str(text or "")))


def _whatsapp_claim_lines(text: str) -> frozenset[int]:
    """The lines of *text* that say a WhatsApp message went out."""
    lines: set[int] = set()
    for pattern in (_WA_I_CLAIM_RE, _WA_STATE_CLAIM_RE):
        for match in pattern.finditer(text):
            start, end = _line_bounds(text, match.start())
            line = text[start:end]
            before = text[start:match.start()]
            after = text[match.end():end]
            if not _WA_WORD_RE.search(line):
                continue
            if (_CLAIM_NEGATED_RE.search(before) or _is_question(text, match.end())
                    or _ATTRIBUTED_RE.match(line) or _YOU_SENT_RE.search(line)
                    or _WA_TO_USER_AFTER_RE.match(after) or _WA_HERE_RE.search(line)
                    or before.count('"') % 2 == 1 or before.count("“") > before.count("”")):
                continue
            lines.add(text.count("\n", 0, start))
    return frozenset(lines)


def whatsapp_corrective() -> str:
    """What the model is told about a reply that says a WhatsApp message went
    out when no send shows it. It asks for a true reply, never for a send."""
    return ("Your reply says a WhatsApp message was sent, but no whatsapp_web send in this conversation "
            "answered \"Sent to … on WhatsApp\". A message goes out only that way. Rewrite your reply so it "
            "is true: say what you prepared, or what stopped the send.")


WHATSAPP_BACKSTOP = "Note: no WhatsApp message was sent."


# ── One decision for a reply ─────────────────────────────────────────


@dataclass(frozen=True)
class TurnFacts:
    """What the turn did, as :func:`judge_reply` needs it."""

    # The turn's request (judge_reply strips an automated turn's prefix line).
    request: str = ""
    # This turn's tool results, in order, and every tool result in the session.
    turn_steps: tuple[ToolStep, ...] = ()
    session_steps: tuple[ToolStep, ...] = ()
    # Message IDs earlier sends returned whose results the session no longer
    # holds (the ID ledger's), and other runs' results the session holds (lane
    # notes, delegated results): an "Email sent" one relays proves its IDs.
    sent_ids: frozenset[str] = frozenset()
    relayed: tuple[str, ...] = ()
    # A mail tool ran in the last few turns.
    recent_mail: bool = False
    # The mail tools this turn offers: none, no mail check.
    mail_tools: frozenset[str] = MAIL_TOOLS
    # Event IDs earlier calendar changes returned (the ledger's), a calendar
    # tool ran in the last few turns, and the calendar tools this turn
    # offers: none, no calendar check.
    event_ids: frozenset[str] = frozenset()
    recent_calendar: bool = False
    calendar_tools: frozenset[str] = CALENDAR_TOOLS
    # Everything the model could have read an ID in (tool results, people's
    # and automated messages, lane notes, summaries, the request it was sent,
    # the ID ledger); never its own words, a rejected reply or a corrective.
    # None: no ID check.
    known: tuple[str, ...] | None = None
    # The turn offered tools and made no tool call: the lookup check runs.
    no_tool_call: bool = False
    # The tools this turn offers (the lookup corrective names the lookups).
    tools: frozenset[str] = frozenset()
    # The payments tools this turn offers: none, no payment check.
    payment_tools: frozenset[str] = frozenset()
    # The WhatsApp send tool, when this turn offers it: none, no WhatsApp check.
    whatsapp_tools: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Verdict:
    """:func:`judge_reply`'s answer. ``kinds`` is empty for a reply that may
    ship; ``claim_lines`` are the lines that report an email as sent or a
    calendar change (true or not), for the checks that run after this one."""

    kinds: tuple[str, ...] = ()
    claim_lines: frozenset[int] = frozenset()
    refused: str = ""
    calendar_refused: str = ""
    narrated: tuple[str, ...] = ()
    fabricated: tuple[str, ...] = ()
    corrective: str = ""
    backstop: str = ""

    @property
    def false(self) -> bool:
        return bool(self.kinds)

    @property
    def origin_detail(self) -> str:
        """The stored rejected reply's ``origin_detail``: the leading kind."""
        return self.kinds[0] if self.kinds else ""


MAIL_BACKSTOP = "Note: no email was sent in this turn."
MAIL_BACKSTOP_REFUSED = "Note: no email was sent in this turn — the send was refused."
CALENDAR_BACKSTOP = "Note: no calendar event was created or changed in this turn."
CALENDAR_BACKSTOP_REFUSED = ("Note: no calendar event was created or changed in this turn — the "
                             "change was refused.")
LOOKUP_BACKSTOP = "Note: the searches and checks described above did not run."
_ID_BACKSTOP_TAIL = " returned by any tool."
# The lookups a corrective may point at, read-only, when the turn offers them.
_LOOKUP_CALLS = (
    ("google_mail", "google_mail search / read_message"),
    ("microsoft_mail", "microsoft_mail search / read_message"),
    ("google_calendar", "google_calendar list_events"),
    ("microsoft_calendar", "microsoft_calendar list_events"),
    ("google_drive", "google_drive search / read"),
    ("microsoft_onedrive", "microsoft_onedrive search / read"),
    ("web_search", "web_search"),
)
_HOW_TO_RE = re.compile(
    r"(?i)^\W*(?:how\s+(?:do|can|could|should|would)\s+(?:i|we|you|one)|how\s+to"
    r"|kako(?:\s+(?:da|mogu|se))?)\b"
)
_QUOTE_CHARS = 200


def has_backstop(text: str) -> bool:
    """Whether *text* already carries a backstop line."""
    text = text or ""
    return (MAIL_BACKSTOP.rstrip(".") in text or CALENDAR_BACKSTOP.rstrip(".") in text
            or LOOKUP_BACKSTOP in text or (_ID_BACKSTOP_TAIL in text and "Note: `" in text)
            or PAYMENT_BACKSTOP in text)


def _quote(text: str) -> str:
    quote = " ".join(str(text or "").split())
    return quote if len(quote) <= _QUOTE_CHARS else quote[:_QUOTE_CHARS].rstrip() + "…"


def _first_sentence(text: str) -> str:
    """What a refusal says happened, without what it says to do next ("To
    send that draft use …", "call again with allow_repeat=true")."""
    flat = " ".join(str(text or "").split())
    match = re.search(r"[.!?](?=\s|$)", flat)
    return _quote(flat[: match.end()] if match else flat)


def _id_list(ids: Sequence[str]) -> str:
    quoted = [f"`{i}`" for i in ids]
    return quoted[0] if len(quoted) == 1 else ", ".join(quoted[:-1]) + " and " + quoted[-1]


def _them(ids: Sequence[str], one: str, many: str) -> str:
    return one if len(ids) == 1 else many


def _lookup_quotes(narrated: Sequence[str]) -> str:
    return ", ".join(f'"{_quote(line)}"' for line in narrated[:2])


def mail_corrective(refused: str = "", earlier_send: bool = False) -> str:
    """What the model is told about a reply that says an email went out when
    none did. It asks for a true reply, never for a send. *earlier_send*:
    the session holds a real send, which the reply may be about — it is not
    told nothing was sent."""
    if refused:
        return ("Your reply says an email was sent, but the send in this turn did not go "
                f'through — the tool answered: "{_first_sentence(refused)}" Nothing was sent. '
                "Rewrite your reply so it is true: say plainly that it was not sent, and why.")
    if earlier_send:
        return ("Your reply says an email was sent, but no send ran in this turn. An email sent in "
                "an earlier turn may be reported only with the Message ID its \"Email sent.\" "
                "returned; otherwise don't claim a send, a Message ID, a search or a check that "
                "didn't happen.")
    return ("Your reply says an email was sent, but no send ran in this turn, so nothing was "
            "sent. Rewrite your reply so it is true — don't claim a send, a Message ID, a search "
            "or a check that didn't happen.")


def calendar_corrective(refused: str = "", earlier_change: bool = False) -> str:
    """What the model is told about a reply that says the calendar changed
    when it did not. Like the mail one, it asks for a true reply, never for
    the change. *earlier_change*: the session holds a real change, which the
    reply may be about."""
    if refused:
        return ("Your reply says a calendar event was created or changed, but the calendar change "
                "in this turn did not go through — the tool answered: "
                f'"{_first_sentence(refused)}" Nothing changed in the calendar. Rewrite your reply '
                "so it is true: say plainly that it was not done, and why.")
    if earlier_change:
        return ("Your reply says a calendar event was created or changed, but no calendar change "
                "ran in this turn. A change made in an earlier turn may be reported only with the "
                "event ID its result returned; otherwise don't claim an event, an invitation, an "
                "event ID or a check that didn't happen.")
    return ("Your reply says a calendar event was created or changed, but no calendar change ran "
            "in this turn, so nothing was created, changed or sent. Rewrite your reply so it is "
            "true — don't claim an event, an invitation, an event ID or a check that didn't "
            "happen.")


def lookup_corrective(narrated: Sequence[str], tools: Iterable[str] = ()) -> str:
    """What the model is told about a reply that narrates lookups in a turn
    with no tool call. It points at read-only lookups only."""
    offered = set(tools)
    calls = [call for name, call in _LOOKUP_CALLS if name in offered]
    return (f"Your reply describes lookups ({_lookup_quotes(narrated)}) but you made no tool "
            "call in this turn, so none of them happened. If you need that information, look it "
            "up now" + (f" ({', '.join(calls)})" if calls else "") + " and report only what it "
            "returns; otherwise answer without describing any search, check or lookup, and say "
            "plainly what you don't know.")


def id_corrective(ids: Sequence[str]) -> str:
    """What the model is told about IDs no tool returned."""
    return (f"Your reply quotes {_id_list(ids)} — no tool in this conversation returned "
            f"{_them(ids, 'that ID', 'those IDs')}, so {_them(ids, 'it is', 'they are')} "
            f"invented. Leave {_them(ids, 'it', 'them')} out, or look "
            f"{_them(ids, 'it', 'them')} up now and quote only what the tool returns.")


def id_backstop(ids: Sequence[str]) -> str:
    return f"Note: {_id_list(ids)} {_them(ids, 'was', 'were')} not" + _ID_BACKSTOP_TAIL


def _name_re(name: Any) -> re.Pattern[str] | None:
    """An agent's name as a reply writes it: any case, any separator, maybe
    inflected ("deep_researcher" → "Deep Researcher", "Researcheru")."""
    words = re.findall(r"[^\W_]+", str(name or ""))
    if sum(map(len, words)) < 3:
        return None
    return re.compile(r"(?i)(?<![^\W_])" + r"[\s_-]*".join(map(re.escape, words)))


def _mail_claim(
    reply: str, request: str, facts: TurnFacts,
) -> tuple[frozenset[int], bool, str, bool]:
    """(the lines that report an email as sent, whether the report is false,
    the refusal of this turn's send, whether the session holds a real send)
    — see :func:`judge_reply`."""
    if not facts.mail_tools or not reply.strip():
        return frozenset(), False, "", False
    steps = facts.turn_steps
    # Something else was delivered (a file, a picture, a voice note): a bare
    # "Sent" is about that. Recent mail alone makes a bare "Sent" an email
    # only in a turn that did nothing but mail and lookups.
    strict = any(_delivers_to_chat(s) for s in steps)
    recent = facts.recent_mail and not strict and not any(_acts_elsewhere(s) for s in steps)
    claims = _find_claims(reply, request=request, recent_mail=recent, strict=strict)
    # A line naming the agent this turn handed work to ("**Sent** to the
    # Researcher", "Poslano Researcheru") reports that hand-off.
    agents = [p for p in (_name_re(s.arguments.get("agent_name")) for s in steps if _is_handoff(s))
              if p]
    if agents:
        reply_lines = reply.split("\n")
        claims = [c for c in claims if not any(p.search(reply_lines[c.line_index]) for p in agents)]
    lines = frozenset(c.line_index for c in claims)
    # A send dated before this turn is history; "I sent your draft to Ted" →
    # "OK, the email to Ted was sent" acknowledges the user's own send.
    claims = [c for c in claims if not c.earlier]
    if not claims or _USER_SENT_RE.search(request) or turn_sent_mail(steps):
        return lines, False, "", False
    # A Message ID an "Email sent." returned (in any turn, or in the run a
    # lane note or a peer's answer relays): the claim is about that email.
    proven = sent_message_ids(facts.session_steps) | facts.sent_ids | relayed_sent_ids(
        facts.relayed)
    if any(_quotes_message_id(token, reply) for token in proven):
        return lines, False, "", False
    return lines, True, turn_refused_send(steps), bool(proven)


def _calendar_claim(
    reply: str, request: str, facts: TurnFacts,
) -> tuple[frozenset[int], bool, str, bool]:
    """(the lines that report a calendar change, whether the report is
    false, the refusal of this turn's change, whether the session holds a
    real change) — see :func:`judge_reply`."""
    if not facts.calendar_tools or not reply.strip():
        return frozenset(), False, "", False
    steps = facts.turn_steps
    # Something else was done (a draft, a file, a hand-off): a headline
    # "Created —" is about that; only a line naming the event counts.
    strict = any(_acts_off_calendar(s) for s in steps)
    claims = _find_calendar_claims(reply, request=request, recent=facts.recent_calendar,
                                   strict=strict)
    agents = [p for p in (_name_re(s.arguments.get("agent_name")) for s in steps if _is_handoff(s))
              if p]
    if agents:
        reply_lines = reply.split("\n")
        claims = [c for c in claims if not any(p.search(reply_lines[c.line_index]) for p in agents)]
    # An email out this turn may be the meeting request the reply says went.
    if turn_sent_mail(steps):
        claims = [c for c in claims if not c.send]
    lines = frozenset(c.line_index for c in claims)
    claims = [c for c in claims if not c.earlier]
    if not claims or _USER_CAL_RE.search(request) or turn_changed_calendar(steps):
        return lines, False, "", False
    # An event ID a change returned (in any turn): the claim is about that
    # event.
    proven = changed_event_ids(facts.session_steps) | facts.event_ids
    if any(_quotes_message_id(token, reply) for token in proven):
        return lines, False, "", False
    earlier = bool(proven) or any(s.ok for s in _calendar_writes(facts.session_steps))
    return lines, True, turn_refused_calendar(steps), earlier


def judge_reply(raw: str, reply: str, facts: TurnFacts) -> Verdict:
    """One decision for a reply, every check together.

    *raw* is the reply as the model wrote it, *reply* what would ship (the
    model-written lane notes cut): the ID check reads *raw* (a forged lane
    note is where made-up IDs sat), the mail, calendar and lookup checks
    *reply*.

    False when *reply* says an email went out and none did this turn (nor a
    peer relayed one), unless it quotes a Message ID an earlier send returned
    (not as a thread's); when it says the calendar changed and no calendar
    change went through this turn, unless it quotes an event ID an earlier
    change returned; when it narrates lookups in a turn that made no tool
    call (not on a how-to question); when *raw* quotes an ID nothing the
    model was given held.

    One corrective: the leading kind's, then one "Also:" sentence for each
    other kind. The backstop has a line per kind; an ID line names only IDs
    that still ship.
    """
    raw, reply = str(raw or ""), str(reply or "")
    request = strip_automated_prefix(facts.request or "")
    lines, mail_false, refused, earlier_send = _mail_claim(reply, request, facts)
    cal_lines, cal_false, cal_refused, earlier_change = _calendar_claim(reply, request, facts)
    lines |= cal_lines
    _pay_lines, pay_false = _payment_claim(reply, facts)
    wa_lines, wa_false = _whatsapp_claim(reply, facts)
    lines |= wa_lines
    narrated = (narrated_lookups(reply, has_pasted_content(request))
                if facts.no_tool_call and not _HOW_TO_RE.match(request) else ())
    fabricated = () if facts.known is None else fabricated_ids(raw, facts.known, request)
    kinds = tuple(kind for kind, on in (
        (MAIL_CLAIM, mail_false), (CALENDAR_CLAIM, cal_false), (PAYMENT_CLAIM, pay_false),
        (WHATSAPP_CLAIM, wa_false), (NARRATED_LOOKUP, bool(narrated)), (FABRICATED_ID, bool(fabricated)),
    ) if on)
    if not kinds:
        return Verdict(claim_lines=lines)
    parts: list[str] = []
    backstop: list[str] = []
    if mail_false:
        parts.append(mail_corrective(refused, earlier_send))
        backstop.append(MAIL_BACKSTOP_REFUSED if refused else MAIL_BACKSTOP)
    if cal_false:
        if not parts:
            parts.append(calendar_corrective(cal_refused, earlier_change))
        elif cal_refused:
            parts.append("Also: your reply says the calendar changed, but the calendar change in "
                         "this turn did not go through — the tool answered: "
                         f'"{_first_sentence(cal_refused)}" Say so plainly.')
        else:
            parts.append("Also: your reply says a calendar event was created or changed, but no "
                         "calendar change ran in this turn — don't claim one.")
        backstop.append(CALENDAR_BACKSTOP_REFUSED if cal_refused else CALENDAR_BACKSTOP)
    if pay_false:
        parts.append(payment_corrective() if not parts else
                     "Also: your reply says a payment was made or a bill is paid — Captain can't "
                     "pay, and nothing shows it paid. Don't claim it.")
        backstop.append(PAYMENT_BACKSTOP)
    if wa_false:
        parts.append(whatsapp_corrective() if not parts else
                     "Also: your reply says a WhatsApp message was sent, but no whatsapp_web send shows "
                     "it — don't claim one.")
        backstop.append(WHATSAPP_BACKSTOP)
    if narrated:
        parts.append(lookup_corrective(narrated, facts.tools) if not parts else
                     f"Also: your reply describes lookups ({_lookup_quotes(narrated)}) that never "
                     "ran — no tool was called in this turn.")
        backstop.append(LOOKUP_BACKSTOP)
    if fabricated:
        parts.append(id_corrective(fabricated) if not parts else
                     f"Also: no tool returned {_id_list(fabricated)} — leave "
                     f"{_them(fabricated, 'it', 'them')} out.")
        shipped = [token for token in fabricated if token in reply]
        if shipped:
            backstop.append(id_backstop(shipped))
    return Verdict(
        kinds=kinds, claim_lines=lines, refused=refused, calendar_refused=cal_refused,
        narrated=narrated, fabricated=fabricated, corrective=" ".join(parts),
        backstop="\n".join(backstop),
    )


# ── Model-written lane notes ─────────────────────────────────────────

# A lane note's header as Flight Deck writes it, alone on its line:
# "[Autonomous Work — from the AUTO lane]" as stored, "[Note from the AUTO
# lane — …, not your reply]" as the model's history shows it. Only Flight Deck
# writes these into the chat; one inside a reply was written by the model.
_LANE_NOTE_HEADER_RE = re.compile(
    r"^[ \t]*\[(?:[^\[\]\n]{1,80} — from the [^\[\]\n]{1,40} lane"
    r"|Note from the [^\[\]\n]{1,40} lane — [^\[\]\n]{0,160})\][ \t]*$"
)
# The request is about the lanes or what they run ("what did the autonomous
# lane post?", "what did the overnight jobs report?", "što je cron javio
# noćas?"): a reply that quotes a note is the answer.
_ASKS_ABOUT_LANES_RE = re.compile(
    r"(?i)\blanes?\b|autonom|\bautomat|\bcron\w*|\bjobs\b|\bovernight\b|\bdigests?\b"
    r"|\bscheduled\s+(?:jobs?|tasks?|runs?|checks?)\b|\bno[ćc]as\b|\bposlovi\b"
)
# A note's header as stored ("[Autonomous Work — from the AUTO lane]").
_STORED_NOTE_HEADER_RE = re.compile(r"^\[[^\[\]\n]*? — from the [^\[\]\n]*? lane\]\n?")
# A relayed block this short is no evidence of a verbatim relay unless it is
# the whole note ("Sent — verified." could sit inside a longer real note).
_RELAY_MIN_CHARS = 24
# A forged note's block that reports an action ("Sent — verified", "Done").
_LANE_BLOCK_ACTION_RE = re.compile(
    r"(?i)\b(?:sent|verified|confirmed|checked|done|delivered|replied|message\s+id"
    r"|poslan\w*|provjer\w*|potvrđen\w*|gotovo|odgovoren\w*)\b"
)


def _flat(text: str) -> str:
    return " ".join(str(text or "").split())


def _relayed_note(body: str, notes: Sequence[str]) -> bool:
    """*body* (a block under a lane note header in the reply) is a real
    note's text, copied as it stands: the whole of one, or a long enough
    part."""
    flat = _flat(body)
    if not flat:
        return False
    return any(flat == note or (len(flat) >= _RELAY_MIN_CHARS and flat in note) for note in notes)


def strip_forged_lane_notes(text: str, user_text: str = "", notes: Iterable[str] = ()) -> str:
    """Cut a lane note the model wrote into its own reply.

    The model copies the notes it sees in history, appending a made-up block
    ("[Autonomous Work — from the AUTO lane]\\nSent — verified…") that reads
    as another run's confirmation. The header line always goes; the block
    under it (to the next blank line) goes too when it reports an action,
    follows the model's own answer and is not a real note's text (*notes*:
    the session's lane notes) copied as it stands. A reply that opens with a
    header, or introduces it ("Here is what the AUTO lane posted:"), keeps the
    body; the rest of the reply is kept. Untouched when the request asks about
    the lanes or their jobs.
    """
    if not text or "lane" not in text.lower():
        return text
    if _ASKS_ABOUT_LANES_RE.search(strip_automated_prefix(user_text or "")):
        return text
    real = [_flat(_STORED_NOTE_HEADER_RE.sub("", str(note or ""), count=1)) for note in notes]
    lines = text.split("\n")
    kept: list[str] = []
    found = False
    i = 0
    while i < len(lines):
        if not _LANE_NOTE_HEADER_RE.match(lines[i]):
            kept.append(lines[i])
            i += 1
            continue
        found = True
        i += 1
        head = "\n".join(kept).rstrip()
        end = i
        while end < len(lines) and lines[end].strip():
            end += 1
        block = "\n".join(lines[i:end])
        if (head and not head.endswith(":") and _LANE_BLOCK_ACTION_RE.search(block)
                and not _relayed_note(block, real)):
            i = end
    if not found:
        return text
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
