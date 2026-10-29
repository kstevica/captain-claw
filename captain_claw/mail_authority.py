"""Who may write email in this turn — the tool-side guard against unrequested mail.

User rule: the agent doesn't draft, reply to or send email unless it was
specifically told to.

* **Human turns** (a person typing in web chat, WhatsApp, Telegram, the CLI …)
  are never hard-gated. The prompts tell the model to write email only when
  asked, and the soft checks here (:func:`nudge_mail_ok`,
  :func:`stall_mail_ok`) keep the turn-loop nudges from pushing a draft
  nobody asked for — judged over the user's last few messages and a "yes"
  to the agent's own offer (:func:`human_asked_for_mail`). In an FD-spawned
  worker or a being nobody types: :func:`interactive` keeps their deny
  default for plain chat frames.
* **Automated turns** (agent cron, the FD scheduler, flows, Autonomous Work,
  plans, peer consult/delegate, relays, sister sessions, BotPort, FD-spawned
  workers, Iskra beings, MCP tasks) may write email only when the job's own
  human-written text explicitly asks for one (:func:`intent_scope`), or when
  the caller vouches for a human click (``mail_write="allow"``). When that
  text only asks for an email to the user themself ("email me the summary"),
  the write must be one new email to the owner's own address.
* **Mailbox changes** (archive, move, labels / categories, a new label or
  folder — :data:`MAILBOX_CHANGE_ACTIONS`) are a separate class, gated the
  same way in automated turns by :func:`check_mailbox_change`: the job's own
  text must ask to organize email (:func:`organize_intent`). Neither kind of
  intent grants the other.

The decision travels in a :class:`contextvars.ContextVar` (never on the
agent), so concurrent lanes and tasks can't leak authority into each other.
Entry points bind it (:func:`bind` / :func:`bound`); ``google_mail``,
``send_mail`` and Gmail-like MCP proxy tools call :func:`check_mail_write`
before any token fetch or HTTP call.

The wire marker (``automation`` on chat / run_tool frames and the
``/api/tool`` body) is parsed by :func:`from_wire`. Flight Deck matches a
refusal by the literal :data:`MAIL_REFUSAL_TAG`.

Dependency-free on purpose: stdlib only, no config, no I/O.
"""

from __future__ import annotations

import contextlib
import contextvars
import email.utils
import os
import re
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Names shared with Flight Deck — by wire and string (Flight Deck mirrors these
# sets as literals), except that Bat's routes import the pure detectors
# (organize_intent, explicit_mail_intent) and JOB_TEXT_MAX. This module never
# imports Flight Deck, and no context state crosses that import.
# ---------------------------------------------------------------------------

AUTOMATION_KINDS = frozenset({
    "autonomy", "autonomy_tool", "plan", "fd_scheduler", "flow", "flow_tool", "cron",
    "peer", "peer_relay", "sister", "botport", "fd_worker", "being", "mcp_task", "bat", "link",
    "exam", "unknown",
})
MAIL_WRITE_ACTIONS = frozenset({"create_draft", "update_draft", "send", "send_draft"})
# google_mail / microsoft_mail actions that change the mailbox without writing
# an email (archive, move, labels / categories, new label / folder). A separate
# class: email intent never grants them, and they never grant email.
# Flight Deck mirrors this set by string (action_catalog.py, flow_runner.py).
MAILBOX_CHANGE_ACTIONS = frozenset({
    "archive", "move", "add_labels", "remove_labels", "create_label",
    "create_folder", "add_categories", "remove_categories",
})
MAIL_REFUSAL_TAG = "[not-authorized: mail-write]"
JOB_TEXT_MAX = 4000

KIND_LABELS: dict[str, str] = {
    "autonomy": "Autonomous Work",
    "autonomy_tool": "Autonomous Work",
    "plan": "an autonomy plan",
    "fd_scheduler": "a scheduled job",
    "flow": "a flow",
    "flow_tool": "a flow",
    "cron": "a scheduled task",
    "peer": "another agent",
    "peer_relay": "another agent's result",
    "sister": "a background task",
    "botport": "a BotPort task",
    "fd_worker": "a multi-agent run",
    "being": "a being tick",
    "mcp_task": "a task sent over MCP",
    "bat": "a Bat run",
    "link": "a connected system",
    "exam": "an onboarding exam",
    "unknown": "an automation",
}

AUTOMATED_TURN_PREFIX = "[Automated turn — {label}. Not a live message from the user.]"

_MAIL_WRITE_MODES = frozenset({"intent", "allow", "deny"})


# ---------------------------------------------------------------------------
# Authority + binding
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Authority:
    mode: str                   # "human" | "automated"
    kind: str = ""              # one of AUTOMATION_KINDS when automated
    job_text: str = ""          # human-written text the decision is judged on
    mail_write: str = "intent"  # "intent" | "allow" | "deny"


HUMAN = Authority(mode="human")

_CURRENT: contextvars.ContextVar[Authority | None] = contextvars.ContextVar(
    "captain_claw_mail_authority", default=None,
)

_TRUTHY = frozenset({"1", "true", "yes"})
# CLAW_BAT_WORKER is here so a Bat worker fails closed to deny by process
# default: a Bat run's email permission comes ONLY from the per-turn automation
# frame ({kind:'bat', mail_write:'allow'}), never from being a worker — and that
# frame is sent only when the owner approved a mail plan (run-scoped, Phase 5).
_WORKER_ENVS = ("CLAW_BASNA_WORKER", "CLAW_VATRA_WORKER", "CLAW_COUNCIL_WORKER",
                "CLAW_CODE_AGENT", "CLAW_BAT_WORKER")


def automated(kind: str, job_text: str = "", mail_write: str = "intent") -> Authority:
    """An automated-turn authority (unknown kinds / modes fail closed)."""
    if kind not in AUTOMATION_KINDS:
        kind, mail_write = "unknown", "deny"
    if mail_write not in _MAIL_WRITE_MODES:
        mail_write = "deny"
    return Authority("automated", kind, str(job_text or "")[:JOB_TEXT_MAX], mail_write)


def human(text: str) -> Authority:
    """A human turn; *text* is the user's message as received."""
    return Authority("human", "", str(text or "")[:JOB_TEXT_MAX], "intent")


def interactive(text: str) -> Authority:
    """What a chat-style entry point binds for a frame without a marker.

    ``human(text)`` — except in an FD-spawned worker or an Iskra being (J7),
    where nobody types: their plain chat frames are LLM-written prompts, so
    the process default (automated, ``deny``) stays in force.
    """
    d = _process_default()
    return d if d.mode == "automated" else human(text)


def _env_truthy(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in _TRUTHY


def _process_default() -> Authority:
    if _env_truthy("CLAW_BEING_WORKER"):
        return Authority("automated", "being", "", "deny")
    if any(_env_truthy(n) for n in _WORKER_ENVS):
        return Authority("automated", "fd_worker", "", "deny")
    return HUMAN


def current() -> Authority:
    """The bound authority, else the process default (workers/beings deny), else HUMAN."""
    a = _CURRENT.get()
    return a if a is not None else _process_default()


def bound_authority() -> Authority | None:
    """The authority bound for this turn, or None when only the process
    default applies. Read-only; provenance labelling uses it."""
    return _CURRENT.get()


def process_default() -> Authority:
    """The authority an unmarked turn gets in this process."""
    return _process_default()


def bind(a: Authority) -> contextvars.Token:
    return _CURRENT.set(a)


def reset(token: Any) -> None:
    if token is None:
        return
    try:
        _CURRENT.reset(token)
    except (ValueError, RuntimeError):
        # Created in another context (e.g. a generator closed elsewhere).
        pass


@contextlib.contextmanager
def bound(a: Authority | None) -> Iterator[None]:
    """Bind *a* for the block; ``None`` is a no-op."""
    if a is None:
        yield
        return
    tok = bind(a)
    try:
        yield
    finally:
        reset(tok)


def from_wire(raw: Any, *, default_kind: str) -> Authority | None:
    """Parse the ``automation`` wire object. ``None`` only when *raw* is None.

    *default_kind* is accepted for symmetry with the callers (which bind
    ``automated(default_kind, "", "deny")`` themselves when this returns
    None); a present-but-malformed marker always fails closed.
    """
    del default_kind
    if raw is None:
        return None
    if not isinstance(raw, dict):
        return Authority("automated", "unknown", "", "deny")
    kind = raw.get("kind")
    mail_write = raw.get("mail_write")
    if mail_write not in _MAIL_WRITE_MODES:
        mail_write = "deny"
    if kind not in AUTOMATION_KINDS:
        kind, mail_write = "unknown", "deny"
    job_text = str(raw.get("job_text") or "")[:JOB_TEXT_MAX]
    return Authority("automated", kind, job_text, mail_write)


# ---------------------------------------------------------------------------
# Explicit-intent detector (EN + HR)
# ---------------------------------------------------------------------------

_NONE, _SELF, _ANY = "none", "self", "any"
_RANK = {_NONE: 0, _SELF: 1, _ANY: 2}

_MAILW = r"\b(?:e-?mail|mail|mejl|thread)\w*"
_ADDR = r"[\w.+'-]+@[\w-]+(?:\.[\w-]+)+"
_ADDR_RE = re.compile(_ADDR)
_MAILW_RE = re.compile(_MAILW)

# The verb is "imperative" when it starts the text / a clause, or follows one
# of these words.
_IMP_TAIL_RE = re.compile(
    r"(?:^|[.!?;:\n,\-*•>(\"'])\s*$"
    r"|(?:^|[^\w])(?:please|pls|and|then|also|now|just|can you|could you|would you"
    r"|molim(?: te)?|pa|onda|samo"
    # requests phrased around the verb ("I need you to draft …", "help me
    # draft …", "let's reply …", "možeš li napisati …")
    r"|(?:want|need|like|ask|asking) you to|help (?:me|us)(?: to)?|let'?s"
    r"|(?:mozes|mozete) li|trebam da|hocu da|zelim da|daj)\s*$"
)
# HR "i" ("and") — only before an HR verb, so the English "I" ("I reply to
# Ana myself", "i draft replies myself") is never an imperative position.
_HR_I_TAIL_RE = re.compile(r"(?:^|[^\w])i\s*$")
# A scheduled job phrased with its time first ("Every Friday draft …",
# "At 5pm reply to Ana", "svaki petak napiši mail Ani"): the verb after a
# leading time phrase is still in imperative position.
_TIME_TOKEN = (
    r"(?:every|each|daily|weekly|monthly|hourly|nightly|tomorrow|tonight|today|morning|mornings"
    r"|evening|evenings|night|nights|afternoon|noon|midnight|day|days|week|weeks|weekday|weekdays"
    r"|weekend|weekends|month|months|hour|hours|minute|minutes|mins?|am|pm|at|on|in|from|first|next"
    r"|this|the|a|an|monday|tuesday|wednesday|thursday|friday|saturday|sunday|mondays|tuesdays"
    r"|wednesdays|thursdays|fridays|saturdays|sundays|mon|tue|wed|thu|fri|sat|sun"
    r"|svaki|svako|svake|svaku|svakog|dan|dana|jutro|ujutro|vecer|navecer|popodne|tjedan|tjedno"
    r"|mjesec|mjesecno|dnevno|svakodnevno|sat|sata|sati|minuta|minute|u|za|od|sutra|danas"
    r"|svakoga|jutra|veceri|tjedna|mjeseca|radni|radnim|danom"
    r"|ponedjeljka|utorka|srijede|cetvrtka|petka|subote|nedjelje"
    r"|ponedjeljak|utorak|srijeda|srijedu|cetvrtak|petak|subota|subotu|nedjelja|nedjelju"
    r"|ponedjeljkom|utorkom|srijedom|cetvrtkom|petkom|subotom|nedjeljom"
    r"|\d{1,2}(?:[:.]\d{2})?(?:am|pm|h)?)"
)
_TIME_TAIL_RE = re.compile(rf"(?:(?<![\w:.]){_TIME_TOKEN}[\s,]+)+$")
# …which must hold a real time word, not just "the" / "in" / "at".
_TIME_STRONG_RE = re.compile(
    r"\b(?:every|each|daily|weekly|monthly|hourly|nightly|tomorrow|tonight|today|mornings?"
    r"|evenings?|nights?|afternoon|noon|midnight|days?|weeks?|weekdays?|weekends?|months?|hours?"
    r"|minutes?|mins?|\w+days|mon|tue|wed|thu|fri|sat|sun|monday|tuesday|wednesday|thursday"
    r"|friday|saturday|sunday|svak\w*|dan|dana|jutr\w*|ujutro|vecer\w*|navecer|popodne|tjed\w*"
    r"|mjesec\w*|dnevno|svakodnevno|sat[ai]?|minut[ae]|sutra|danas|ponedjelj\w*|utor\w*"
    r"|srijed\w*|cetvrt\w*|pet\w*|subot\w*|nedjelj\w*|\d)"
)
# "Set up a cron job in 5 minutes to draft a reply …" — a job that does it
# (never a reminder to do it yourself).
_JOB_TO_TAIL_RE = re.compile(
    r"\b(?:job|task|cron|cronjob|routine|automation|workflow)\b[^.!?;\n]{0,60}?\bto\s*$"
)
# "don't send", "never email", "nemoj poslati" …
# "Remind me (every Friday) to email Ana" — a reminder, not a job that writes.
_REMIND_TAIL_RE = re.compile(r"\b(?:remind\w*|podsjet\w*)\b[^.!?;\n]{0,60}?\b(?:to|da)\s*$")
# …also with an intensifier between ("do not under any circumstances send",
# "under no circumstances whatsoever archive", "never at any point reply").
_NEG_TAIL_RE = re.compile(
    r"(?:^|[^\w])(?:not|don'?t|do not|never|no|without|nemoj|nemojte|ne|nikad|nikada|bez)"
    r"(?:\s+(?:under|at|in|on|by|for)\s+(?:any|no)\s+\w+)?(?:\s+\w+)?"
    r"(?:\s+(?:whatsoever|ever|at all))?\s*$"
)
# "Do not, under any circumstances, email Ana" — a negation with an aside
_NEG_ASIDE_RE = re.compile(
    r"(?:^|[^\w])(?:don'?t|do not|never|nemoj|nemojte|nikad|nikada)\s*,[^.!?;\n]{0,80},\s*$"
)
# A coordinated list of mail actions ("don't archive, move or label …",
# "never draft or send …", "nemoj arhivirati, premjestiti ni označiti …"): a
# negation or a reminder before its first verb covers every verb in it.
_LIST_VERB = (
    r"(?:(?:un|re)?archive|(?:re|un)?label|(?:un)?tag|(?:re)?categori[sz]e|(?:un)?star|move|file"
    r"|sort|mark|flag|put|add|apply|assign|remove|delete|trash|touch|change|modify|edit|open"
    r"|read|(?:re)?organi[sz]e|clean|draft|compose|write|prepare|create|make|send|forward"
    r"|e-?mail|mail|reply|respond|answer"
    r"|arhivir\w*|premjest\w*|prebac\w*|pomak\w*|makn\w*|oznac\w*|kategoriz\w*|razvrst\w*"
    r"|sortir\w*|sprem\w*|stav\w*|napis\w*|pisa\w*|pisi\w*|sastav\w*|priprem\w*|posal\w*"
    r"|posla\w*|salj\w*|slat\w*|odgovor\w*|odgovar\w*|proslijed\w*|bris\w*|obris\w*|dira\w*"
    r"|otvar\w*|otvori\w*|cita\w*|procita\w*|mijenja\w*|promijen\w*)"
)
# A word of a list member's object ("archive emails or move them"): never a
# coordinator or a word that starts a new instruction — except an "and" /
# "or" inside the object ("any email from Substack or Medium").
_LIST_OBJ_WORD = (
    rf"(?:(?:and|or|i|ili)\s+(?!{_LIST_VERB}\b))?"
    r"(?!(?:and|or|nor|then|just|instead|but|also|please|only|now|rather"
    r"|ili|ni|niti|i|pa|onda|samo|nego|vec|ali|te)\b)[\w'@-]+(?:\.[\w'@-]+)*"
)
# A member's object is at most this many words ("do not archive any of the
# emails, move them or label them").
_LIST_OBJ_MAX = 8
# A negation is never part of a member's object: a member that runs over one
# ("read my emails and under no circumstances archive, move or label them")
# is negated, like one after it.
_LIST_NEG_WORD_RE = re.compile(
    r"(?<![\w'])(?:no|not|never|without|don'?t|nemoj\w*|ne|nikad\w*|nikako|niposto|nista|bez)"
    r"(?![\w'])"
)
# The member from a verb to the next one and what joins them: ","  "and"
# "or"  "/" … (matched at each list verb before it, nearest first).
_LIST_VERB_RE = re.compile(rf"(?<![\w']){_LIST_VERB}\b")
_LIST_TAIL_RE = re.compile(
    rf"(?P<v>{_LIST_VERB})\b(?P<obj>(?:\s+{_LIST_OBJ_WORD}){{0,{_LIST_OBJ_MAX}}})"
    r"\s*(?P<comma>,)?\s*(?P<conj>(?:and/or|and|or|nor|ili|niti|ni|i)\s+|[/&]\s*)?$"
)
_LIST_OR = frozenset({"or", "nor", "and/or", "ili", "ni", "niti"})
# "don't archive emails, move them or label them" — commas between members
# with objects make a list only when it ends with "or" / "nor" ("A, B or C").
_LIST_CLOSE_RE = re.compile(
    rf"\w+(?:\s+{_LIST_OBJ_WORD}){{0,{_LIST_OBJ_MAX}}}\s*"
    rf"(?:,\s*{_LIST_VERB}\b(?:\s+{_LIST_OBJ_WORD}){{0,{_LIST_OBJ_MAX}}}\s*)*"
    rf",?\s*(?:or|nor|ili|niti|ni)\s+{_LIST_VERB}\b"
)
# "don't reply, email Ana the report instead" — a contrast, not a list: only
# in the judged verb's own member (up to the next "," or coordinator, or a
# "just" / "please" / "only" that starts a new instruction), never "instead
# of …" / "rather than …" ("don't archive or label emails instead of reading
# them"), nor in a later clause ("…, just summarize them instead"), and
# "rather" only as the member's last word (not "the rather old newsletters").
_LIST_CONTRAST_RE = re.compile(
    r"\b(?:instead(?!\s+of\b)|rather(?=\s*$)|radije(?!\s+(?:nego|od)\b)"
    r"|umjesto\s+(?:toga|tog|svega)\b)"
)
_LIST_MEMBER_END_RE = re.compile(
    r"[,;.!?:\n()—–]|\s-\s"
    r"|\b(?:and|or|nor|but|then|just|only|please|ili|ni|niti|i|pa|te|ali|nego|onda|zatim"
    r"|samo|molim)\b"
    # …and a verb with an object of its own ("move my emails summarize them instead")
    rf"|\b(?:{_LIST_VERB}|summari[sz]e|list|sazmi\w*|izlistaj\w*)\s+(?:them|it|everything|ih|ga|sve)\b"
)
_WORD_RE = re.compile(r"[\w'-]+")
# An HR infinitive ("nemoj arhivirati mailove i premjestiti ih") is never an
# instruction of its own: it still hangs on the "nemoj" before it.
_HR_INFINITIVE_RE = re.compile(r"\w+(?:ti|ci)\b")
# How far back the tail patterns above look (each one is shorter): a long
# job text stays linear in its verbs.
_TAIL_WINDOW = 160

_NOT_NAMES = frozenset({
    "The", "A", "An", "This", "That", "These", "Those", "I", "Me", "Us", "User", "Users",
    "JSON", "YAML", "Json", "Yaml", "Markdown", "English", "Croatian", "German", "Hrvatskom", "Engleskom",
    "Summary", "Report",
    # (extended) other languages, channels, days and months
    "French", "Spanish", "Italian", "Slovenian", "Serbian", "Bosnian", "Hungarian",
    "Hrvatski", "Engleski", "Njemacki", "Njemackom", "Talijanskom", "Slovenskom",
    "Slack", "Slacku", "Telegram", "Telegramu", "Discord", "Teams", "Signal", "Viber",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
    "Ponedjeljak", "Utorak", "Srijeda", "Srijedu", "Cetvrtak", "Petak", "Subota", "Subotu",
    "Nedjelja", "Nedjelju", "Here", "There", "Only", "All", "Everyone", "Yes", "No",
    "Ok", "Okay", "Today", "Tomorrow", "Danas", "Sutra",
    # (extended) apps and tools — "Respond to Jira tickets" isn't a person
    "Jira", "Github", "Gitlab", "Notion", "Trello", "Asana", "Zendesk", "Confluence",
    "Whatsapp", "Linkedin", "Twitter", "Reddit", "Calendar", "Kalendar", "Drive",
})
_NOT_NAME_PREFIXES = ("Korisnik", "Question", "Pitanj")
_NAME_RE = re.compile(r"([A-Z][a-z]+)(?:'s)?\b")

_USERISH_RE = re.compile(
    r"(?:the user|users?|user's|me|us|korisnik\w*|meni|nama|mi|nam)\b"
)

_DET = (
    r"(?:the|that|this|these|those|his|her|their|my|our|all|each|every|any"
    r"|taj|ovaj|tu|ovu|sve|svaki|svaku|njegov\w*|njezin\w*|njen\w*)"
)
_CHANNEL_RE = re.compile(
    r"\b(?:whats\s?app\w*|slack\w*|telegram\w*|sms|teams|discord\w*|viber\w*|signal\w*"
    r"|messenger\w*|jira|github|gitlab|linkedin|twitter)\b"
)
_REAL_MAIL_RE = re.compile(r"\b(?:e-?mail|mail|mejl)\w*")

# A flow step's template placeholder ("{{trigger.from}}"). It is the user's
# own text (flows are judged on the RAW template), so it counts as an
# addressee when it names one ("from", "sender", "client_email" …) or when
# the sentence is about email anyway ("Email {{x}} …", "… by email").
_TPL = r"\{\{[^{}\n]{1,80}\}\}"
_TPL_RE = re.compile(_TPL)
_TPL_ADDR_WORDS = frozenset({
    "from", "sender", "to", "email", "mail", "address", "recipient", "recipients", "client",
    "customer", "contact", "author", "klijent", "kupac", "posiljatelj", "primatelj",
})

_MAILOBJ_PREFIX_RE = re.compile(r"(?:(?:to|on|na|za)\s+)?")
_MAILOBJ_MAIL_RE = re.compile(rf"(?:{_DET}\s+)?(?:\w+\s+)?{_MAILW}")
_MAILOBJ_MSG_RE = re.compile(r"(?:(?:the|that|this)\s+)?(?:message|poruk\w*)\s+(?:from|od)\s+")

# Nouns after "<compose verb> … email" that make it something else ("write
# the email parser", "do the email triage", "the drafts list").
_EN_MAIL_DENY_TAIL = (
    r"(?!\s+(?:list|lists|folder|folders|summary|count|report|overview|digest|filter|filters"
    r"|rule|rules|label|labels|template|templates|signature|address|addresses|triage|subjects?"
    r"|lines|headers?|ids?|body|bodies|contents?|stats|statistics|analysis|data|exports?"
    r"|backups?|settings|config|configuration|notifications?|parser|client|server|handler"
    r"|integration|module|function|script|code|tool|logic|validation|validator|regex|fields?"
    r"|columns?|format|formatting|tests?|copy|copies|copywriting|ideas?|designs?|mockups?"
    r"|wireframes?|samples?|examples?|outlines?)\b)"
)
# EN-1 compose
_EN_COMPOSE_RE = re.compile(
    r"\b(?:draft|compose|write|prepare|create|do|make)\s+"
    r"(?:(?:a|an|the|my|our|some|short|quick|brief|polite|formal|friendly|follow-up"
    r"|followup|new|thank-you|reply|gmail)\s+){0,2}"
    r"(?:(?P<mail>e-?mails?|mails?|drafts?(?!\s+of\b))|(?P<reply>repl(?:y|ies)|responses?|messages?))\b"
    + _EN_MAIL_DENY_TAIL
)
# EN-1 compose with free words before the noun ("draft a status update
# email", "write the monthly investor update email"): never a preposition
# or pronoun (so "draft notes from the email" isn't one), and only the real
# mail nouns (a "project draft" or a Slack "message" isn't an email).
_EN_FREE_WORD = (
    r"(?:(?!(?:from|about|of|in|on|for|to|with|into|at|by|via|and|or|that|which|than|sure"
    r"|if|whether|me|us|him|her|them|it|up|down|out|off|over|back|all)\b)[\w'-]+\s+)"
)
# …and the noun ends the phrase: "the onboarding email copy" or "the email
# newsletter design" is a document, not an email.
_EN_FREE_FOLLOW = (
    r"(?=\s*(?:$|[.,;:!?)\n]|\{\{|(?:to|for|with|about|and|then|that|saying|asking|telling"
    r"|thanking|informing|letting|confirming|re|regarding|on|in|at|from|of|including|containing"
    r"|listing|every|each|daily|weekly|monthly|tomorrow|today|tonight|this|next|by|so|which|who"
    r"|if|when|once|after|before|but|or|cc|bcc|using|via|through)\b))"
)
_EN_COMPOSE_FREE_RE = re.compile(
    r"\b(?:draft|compose|write|prepare)\s+(?:(?:a|an|the|my|our|some|this|that|these|those"
    rf"|his|her|their)\s+)?{_EN_FREE_WORD}{{1,3}}"
    r"(?:(?P<mail>e-?mails?|mails?|gmail\s+drafts?)|(?P<reply>repl(?:y|ies)|responses?))\b"
    + _EN_MAIL_DENY_TAIL + _EN_FREE_FOLLOW
)
# EN-1 "write me an email" (self unless the sentence names someone else)
_EN_COMPOSE_ME_RE = re.compile(
    r"\b(?:draft|compose|write|prepare)\s+(?:me|us)\s+(?:(?:a|an|the|some)\s+)?"
    rf"{_EN_FREE_WORD}{{0,2}}?(?:e-?mails?|mails?)\b" + _EN_MAIL_DENY_TAIL
)
# EN-1b "write Ana an email", "draft Ana a reply" (NAME checked on the cased text)
_EN_COMPOSE_TO_RE = re.compile(
    r"\b(?:write|draft)\s+(?P<n>[a-z]+)\s+(?:a|an)\s+(?:\w+\s+)?(?:e-?mail|mail|reply|draft)\b"
)
# EN-2 reply
_EN_REPLY_RE = re.compile(r"\b(?:reply(?:[\s-]+all)?|respond|answer)\b\s*")
# "reply to her" — a pronoun object; an email only when the sentence is about email
_EN_PRONOUN_OBJ_RE = re.compile(
    r"(?:to\s+)?(?:him|her|them|it|each of them|all of them|everyone)\b"
    r"(?=\s*(?:$|[.,;:!?)\n]|(?:and|with|saying|say|telling|tell|that|to|about|re|regarding|by"
    r"|via|asap|now|today|tomorrow|immediately|promptly|politely|briefly|quickly|kindly"
    r"|confirming|thanking|asking|letting|informing|if|when|once|from|using|through|within"
    r"|in)\b))"
)
_HR_PRONOUN_OBJ_RE = re.compile(r"(?:mu|joj|im|njemu|njoj|njima|svima)\b")
# EN-3 send
_EN_SEND_NOUN_RE = re.compile(
    r"\bsend\s+(?P<words>(?:(?!(?:from|about|of|in|on|for|with|into|at|by|via|over|per|without"
    rf"|to)\b)(?:[\w'@.-]+|{_TPL})\s+){{0,3}}?)(?:(?:a|an|the)\s+)?(?:[\w-]+\s+)?"
    r"(?:e-?mails?|mails?|repl(?:y|ies)|drafts?)\b"
)
_EN_SEND_BY_RE = re.compile(
    r"\bsend\b(?P<mid>.{0,60}?)\b(?:by|via|over|per|through|using)\s+"
    r"(?:e-?mail|mail|gmail|google_mail|send_mail|outlook|microsoft_mail)\b"
)
# "send / forward / mail … to ana@x.co" (an address, an address-like
# placeholder, or "to my inbox")
_EN_SEND_TO_RE = re.compile(
    r"\b(?:send|forward|mail|e-?mail)\b(?P<mid>(?:(?![.!?;](?:\s|$))[^\n]){0,80}?)\bto\s+"
    rf"(?:(?P<addr>{_ADDR})|(?P<tpl>{_TPL})"
    r"|(?P<my>(?:my|our)\s+(?:own\s+)?(?:e-?mail|mail|inbox|gmail|address|e-?mail address)\b))"
)
# "use google_mail to send …", "via gmail create_draft …" — naming the mail
# tool together with a write verb
_EN_TOOL_WRITE_RE = re.compile(
    r"\b(?:use|using|call|via|with|through)\s+(?:the\s+)?"
    r"(?:google_mail|microsoft_mail|send_mail|gmail|outlook)\b"
    r"(?:\s+tool)?(?:\s+action)?[\s=:'\",]*(?:to\s+)?"
    r"(?:create_draft|send_draft|send|draft|reply|compose|write)\b"
)
# EN-4 email-as-verb
_EN_EMAIL_VERB_RE = re.compile(
    rf"\be-?mail\s+(?:(?P<self>me|us)\b|(?:it|them|him|her|the|this|that)\b|{_ADDR})"
)
_EN_MAIL_ME_RE = re.compile(r"\bmail\s+(?:me|us)\b")
_EN_EMAIL_NAME_RE = re.compile(r"\be-?mail\s+(?=[a-z{])")
_AFTER_NAME_OK_RE = re.compile(
    r"\s*(?:$|[.,;:!?]|(?:the|a|an|that|this|my|our|about|re|with|to|and)\b)"
)
_DETERMINER_TAIL_RE = re.compile(
    r"(?:^|[^\w])(?:the|a|an|this|that|my|your|his|her|their|our|each|every|any|its)\s*$"
)
# EN-4 "email" is the noun after a quantifier or a verb that takes mail as
# its object ("Archive email that is older …", "Summarize email the boss
# sent", "all email from Ana") …
_EMAIL_NOUN_TAIL_RE = re.compile(
    r"(?:^|[^\w'])(?:all|some|these|those|such|other|more|most|both|several"
    r"|(?:un)?archive|(?:re|un)?label|(?:un)?tag|move|file|sort|(?:un)?star|mark|flag"
    r"|(?:re)?categori[sz]e|summari[sz]e|read|reread|check|search|scan|review|find|process"
    r"|handle|open|delete|trash|remove|ignore|skip|list|count|collect|fetch|get|pull|filter"
    r"|triage|clean|organi[sz]e|monitor|watch|track|analy[sz]e|parse|extract|forward|save"
    r"|export|print|translate|keep|show|display|block|mute|snooze|prioriti[sz]e|clear|purge"
    r"|go through)\s+$"
)
# … or before a relative clause ("email that came today", "email which is …")
_EMAIL_RELATIVE_RE = re.compile(
    r"e-?mail\s+(?:that|which)(?:'s\b|\s+(?:is|are|was|were|has|have|had|came|comes|come|arrived"
    r"|arrives|arrive|got|gets|landed|lands|mentions?|mentioned|contains?|contained|needs?"
    r"|needed|looks?|looked|seems?|seemed|matches|matched|includes?|included|refers?|referred"
    r"|sent|received|bounced|failed|asks?|asked|says|said|i|we|you|they|he|she|isn't|aren't"
    r"|wasn't|weren't|hasn't|haven't|doesn't|didn't|don't|will|would|should|could|can|might"
    r"|may|must)\b)"
)
# EN-4 in a statement is no instruction: a subject before "email" ("I'll
# email them myself", "who I should email", "Ana will email them") …
_EMAIL_SUBJECT_TAIL_RE = re.compile(
    r"(?:^|[^\w'])(?P<s>i|we|he|she|they|who|someone|somebody|everyone|everybody|nobody|people"
    r"|[a-z]+)(?:'ll|'d|'ve|'m|'re)?\s+(?P<mod>(?:(?:will|would|should|could|can|must|might|may"
    r"|shall|usually|always|often|never|also|just|then|still|already|normally|typically"
    r"|sometimes|generally|ever|need to|needs to|have to|has to|had to|want to|wants to"
    r"|wanted to|used to|plan to|plans to|going to|gonna|wanna|tried to|try to|forgot to|did|do"
    r"|does|didn't|don't|doesn't)\s+){0,3})$"
)
_EMAIL_SUBJECTS = frozenset({
    "i", "we", "he", "she", "they", "who", "someone", "somebody", "everyone", "everybody",
    "nobody", "people",
})
# …after a NAME only with a modal ("Ana will email …"; "Please always email" is a request)
_EMAIL_NAME_MOD_RE = re.compile(
    r"(?:'ll|'d|will|would|should|could|can|must|might|may|usually|normally|typically|often"
    r"|sometimes|already|still|needs to|has to|wants to|plans to)\b"
)
# … or asked of someone else ("Ana asked me to email her the deck")
_EMAIL_ASKED_TAIL_RE = re.compile(
    r"(?:^|[^\w])(?:ask\w*|told|tell\w*|want\w*|expect\w*|need\w*|requir\w*|like|wish\w*|get)"
    r"\s+(?:me|us|him|her|them)\s+to\s+$"
)
# "email me and Ana …", "send me, ana@x.co …" — the user and someone else
_ALSO_RE = re.compile(r"\s*(?:,|and|&|\+)\s*")

_HR_STOP = (
    r"(?:sazetak|pregled|popis|listu|lista|izvjestaj|summary|list|overview|digest"
    r"|statistiku|iz|od|o|u|na|s|sa|po|prema|svih|sve|tablic\w*|analiz\w*|sazet\w*"
    r"|pregled\w*|popis\w*|izvjestaj\w*|list\w*)"
)
_HR_FILLERS = rf"(?:(?!{_HR_STOP}\b)(?:\w+|{_TPL})\s+){{0,2}}"
_HR_MAIL_NOUN = r"(?:mail\w*|mejl\w*|e-?mail\w*|draft\w*|nacrt\w*)"
# HR-1 compose (imperfective "piši" is the natural form for a recurring job)
_HR_COMPOSE_RE = re.compile(
    r"\b(?:napisi|napisite|napisati|sastavi|sastavite|pripremi|pripremite|napravi|napravite"
    rf"|kreiraj|slozi|pisi|pisite)\s+{_HR_FILLERS}"
    rf"(?:(?P<mail>{_HR_MAIL_NOUN})|(?P<reply>odgovor\w*))\b"
)
_HR_SEND_VERB = r"(?:posalji|posaljite|poslati|salji|saljite)"
# HR-2 send
_HR_SEND_NOUN_RE = re.compile(
    rf"\b{_HR_SEND_VERB}\s+(?P<words>{_HR_FILLERS}){_HR_MAIL_NOUN}\b"
)
_HR_SEND_CHANNEL_RE = re.compile(
    rf"\b(?:{_HR_SEND_VERB}|javi|pisi)\b(?P<mid>.{{0,60}}?)"
    r"\b(?:mailom|mejlom|e-?mailom|na (?:e-?)?mail\b|putem (?:e-?)?maila)"
)
# "pošalji … na ana@x.co" (or an address-like placeholder)
_HR_SEND_TO_RE = re.compile(
    rf"\b(?:{_HR_SEND_VERB}|proslijedi|proslijedite)\b"
    r"(?P<mid>(?:(?![.!?;](?:\s|$))[^\n]){0,80}?)\b(?:na|za)\s+"
    rf"(?:(?P<addr>{_ADDR})|(?P<tpl>{_TPL}))"
)
_HR_SELF_RE = re.compile(r"\s*(?:mi|nam)\b")
_HR_SELF_BEFORE_RE = re.compile(r"(?:^|[^\w])(?:mi|nam)\s+$")
# HR-3 reply (never the infinitive "odgovoriti")
_HR_REPLY_RE = re.compile(r"\b(?:odgovori(?:te)?|odgovaraj(?:te)?)\b\s*")
# A clitic or a dative NAME between a leading time phrase and an HR verb
# ("Svako jutro mi napiši mail", "Svaki petak Ani odgovori na mail").
_HR_CLITIC_TAIL_RE = re.compile(r"(?:^|[^\w])(?P<w>mi|nam|mu|joj|im|ti|vam)\s+$")
_HR_WORD_TAIL_RE = re.compile(r"(?:^|[^\w])(?P<w>\w+)\s+$")

_EN_SELF_AFTER_RE = re.compile(r"\s*(?:me|us)\b")
_TO_SELF_RE = re.compile(r"\bto\s+(?:me|us)\b")
_SENT_START_RE = re.compile(r"[.!?;](?=\s|$)|\n")
# People a reply goes to ("draft replies to the investors who wrote this week")
_EN_PEOPLE_RE = re.compile(
    r"\b(?:to|for)\s+(?:(?:the|all|my|our|these|those|each|every|any|new)\s+){0,2}"
    r"(?:\w+\s+)?(?:investors?|clients?|customers?|candidates?|applicants?|vendors?|suppliers?"
    r"|partners?|leads?|senders?|contacts?|guests?|attendees?|recruiters?|subscribers?"
    r"|colleagues?|everyone|everybody)\b"
    r"(?!['\u2019]|\s+(?!(?:who|that|which|from|about|re|regarding|with|and|or|this|today"
    r"|tonight|tomorrow|each|every|on|in|at|by|saying|asking|thanking|letting|telling|to|if"
    r"|when|once|after|before|but|so|confirming|informing)\b)[a-z])"
    r"|\bwho\s+(?:wrote|emailed|e-mailed|mailed|sent|contacted|asked|reached out)\b"
)
# Where a reply goes that isn't email ("… in the doc", "… on the blog", "in
# Zendesk") — only for the looser reply forms (people, pronoun, a sentence
# about email before the verb).
_VENUE_RE = re.compile(
    r"\b(?:docs?|documents?|google doc|sheets?|spreadsheets?|page|pages|forms?|forum|blog|posts?"
    r"|comments?|reviews?|tickets?|portal|chats?|zendesk|intercom|hubspot|greenhouse|trustpilot"
    r"|notion|faq|survey|website|web|site|crm|app store|pr|issues?|wiki|dokument\w*"
    r"|komentar\w*|recenzij\w*|chatu|webu|stranic\w*|forumu|tiket\w*|portalu)\b"
)


_APOSTROPHES = str.maketrans({c: "'" for c in "’‘ʼ′＇`´"})


def _normalize(text: str) -> tuple[str, str]:
    """``(cased, low)`` — diacritics folded, whitespace collapsed, same length."""
    s = str(text or "").replace("đ", "dj").replace("Đ", "Dj")
    # Typographic apostrophes (phones and macOS type "Don’t") read as "'".
    s = s.translate(_APOSTROPHES)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"\s+", lambda m: "\n" if "\n" in m.group(0) else " ", s).strip()
    low = "".join(c.lower() if len(c.lower()) == 1 else c for c in s)
    return s, low


def _imperative_at(low: str, start: int, *, hr: bool = False, cased: str = "") -> bool:
    """Is the verb at *start* in imperative position (clause start, a request
    phrase, a leading time phrase, or "a job … to")? *hr*: an HR verb, which
    may also follow "i" ("and"), and may follow a clitic ("mi napiši") — or,
    after a leading time phrase, a dative NAME ("Svaki petak Ani napiši")."""
    if _NEG_ASIDE_RE.search(low[:start]):
        return False
    starts = [start]
    if hr:
        cm = _HR_CLITIC_TAIL_RE.search(low, 0, start)
        if cm:
            starts.append(cm.start("w"))
        elif cased:
            wm = _HR_WORD_TAIL_RE.search(low, 0, start)
            if wm and _name_at(cased, wm.start("w")) >= 0:
                tm = _TIME_TAIL_RE.search(low, 0, wm.start("w"))
                if tm and tm.start() < wm.start("w") and _TIME_STRONG_RE.search(
                        low, tm.start(), wm.start("w")):
                    starts.append(wm.start("w"))
    for s in starts:
        ends = [s]
        tm = _TIME_TAIL_RE.search(low, 0, s)
        if tm and tm.start() < s and _TIME_STRONG_RE.search(low, tm.start(), s):
            ends.append(tm.start())
        for end in ends:
            if _IMP_TAIL_RE.search(low[:end]) or (hr and _HR_I_TAIL_RE.search(low[:end])):
                return True
    jm = _JOB_TO_TAIL_RE.search(low[:start])
    return bool(jm) and "remind" not in jm.group(0) and "podsjet" not in jm.group(0)


def _negated_at(low: str, start: int) -> bool:
    """A negation ("don't send") or a reminder ("remind me to send") before
    *start* — or before the first verb of a list the verb at *start* is part
    of ("don't draft, send or reply …", "never archive, move my emails")."""
    return _negated_here(low, start) or _list_negated(low, start)


def _negated_here(low: str, start: int) -> bool:
    lo = max(0, start - _TAIL_WINDOW)
    return bool(_NEG_TAIL_RE.search(low, lo, start) or _REMIND_TAIL_RE.search(low, lo, start)
                or _NEG_ASIDE_RE.search(low, lo, start))


def _list_negated(low: str, start: int) -> bool:
    """Walk back from the verb at *start* over a coordinated list of mail
    actions to its first verb; True when that one is negated or a reminder.

    A bare verb joins with anything (", ", "and", "/"); one with an object
    only with "or" / "nor" ("don't archive emails or move them"), in a serial
    "A, B or C", or before an HR infinitive. After an object, ", " or "and"
    starts a new instruction ("don't archive anything, label the invoices"),
    and so does a contrast in the verb's own member right after a negated
    one ("don't reply, archive them instead").

    Every verb before it that can start its member is judged, nearest
    first, so an earlier verb whose object runs over the negation ("read my
    emails and under no circumstances archive, move …") never hides it; a
    member whose object holds a negation is negated itself."""
    infinitive = bool(_HR_INFINITIVE_RE.match(low, start))
    contrast = not infinitive and _member_contrast(low, start)
    todo, seen = [start], {start}
    for step in range(8):
        later = []
        for pos in todo:
            for m in _list_members_before(low, pos):
                conj = (m.group("conj") or "").strip()
                if not (m.group("comma") or conj):
                    continue
                if m.group("obj").strip() and conj not in _LIST_OR and not infinitive and not (
                        m.group("comma") and not conj and _LIST_CLOSE_RE.match(low, pos)):
                    continue
                v = m.start("v")
                if _LIST_NEG_WORD_RE.search(m.group("obj")) or _negated_here(low, v):
                    if not (contrast and step == 0 and not conj):
                        return True
                elif v not in seen:
                    seen.add(v)
                    later.append(v)
        todo = later
    return False


def _list_members_before(low: str, pos: int) -> Iterator[re.Match]:
    """Each list member that ends at *pos* (a verb, its object and what joins
    it to the next verb), the one whose verb is nearest *pos* first."""
    verbs = list(_LIST_VERB_RE.finditer(low, max(0, pos - _TAIL_WINDOW), pos))
    for vm in reversed(verbs):
        m = _LIST_TAIL_RE.match(low, vm.start(), pos)
        if m:
            yield m


def _member_contrast(low: str, start: int) -> bool:
    """A contrast word ("instead", "rather", "radije") in the list member of
    the verb at *start* — from the verb to the next "," or coordinator (an
    address is one word: its "." ends nothing, "send it to ana@x.co instead")."""
    vm = _WORD_RE.match(low, start)
    a = vm.end() if vm else start
    member = _ADDR_RE.sub(lambda m: "@" * len(m.group(0)), low[a:a + 80])
    em = _LIST_MEMBER_END_RE.search(member)
    return bool(_LIST_CONTRAST_RE.search(member, 0, em.start() if em else len(member)))


def _email_verb_at(cased: str, low: str, start: int) -> bool:
    """EN-4: is the "email" at *start* the verb of an instruction — wherever
    it stands ("When it's ready email it to Ana", "Always email …", "— email
    the top 5 to …") — and not the noun ("Archive email that is older …") or
    a statement ("I'll email them myself", "Ana will email them")?"""
    lo = max(0, start - _TAIL_WINDOW)
    if (_DETERMINER_TAIL_RE.search(low, lo, start) or _EMAIL_NOUN_TAIL_RE.search(low, lo, start)
            or _EMAIL_RELATIVE_RE.match(low, start)
            or _EMAIL_ASKED_TAIL_RE.search(low, lo, start)):
        return False
    sm = _EMAIL_SUBJECT_TAIL_RE.search(low, lo, start)
    if not sm or sm.group("s") == "you":
        return True  # "You should email the report to Ana" asks the agent
    if sm.group("s") in _EMAIL_SUBJECTS:
        return False
    after_name = low[sm.end("s"):start].lstrip()
    return not (_EMAIL_NAME_MOD_RE.match(after_name) and _name_at(cased, sm.start("s")) >= 0)


def _name_at(cased: str, pos: int) -> int:
    """End index of a NAME (a capitalised, non-stoplisted word) at *pos*, else -1."""
    if cased[pos:pos + 2] == "{{":
        return -1
    m = _NAME_RE.match(cased, pos)
    if not m:
        return -1
    word = m.group(1)
    if word in _NOT_NAMES or word.startswith(_NOT_NAME_PREFIXES):
        return -1
    return m.end()


def _template_addressish(tpl: str) -> bool:
    parts = re.split(r"[^a-z0-9]+", tpl.lower().replace("_", " ").replace("-", " "))
    return any(p in _TPL_ADDR_WORDS or "mail" in p for p in parts if p)


def _template_at(cased: str, pos: int, *, mail_ctx: bool) -> int:
    """End index of a ``{{…}}`` placeholder at *pos* that can be an addressee, else -1."""
    m = _TPL_RE.match(cased, pos)
    if not m:
        return -1
    return m.end() if (mail_ctx or _template_addressish(m.group(0))) else -1


def _addressee_at(cased: str, low: str, pos: int, *, mail_ctx: bool) -> int:
    """End index of a NAME, an address or an addressee placeholder at *pos*, else -1."""
    m = _ADDR_RE.match(low, pos)
    if m:
        return m.end()
    end = _name_at(cased, pos)
    if end >= 0:
        return end
    return _template_at(cased, pos, mail_ctx=mail_ctx)


def _sentence(low: str, pos: int) -> tuple[int, int]:
    """The sentence around *pos*: from the last boundary before it to the
    clause window after it."""
    a = 0
    for m in _SENT_START_RE.finditer(low, 0, pos):
        a = m.end()
    return a, _clause_window(low, pos)[1]


def _mail_in_sentence(low: str, pos: int) -> bool:
    a, b = _sentence(low, pos)
    seg = low[a:b]
    return bool(_REAL_MAIL_RE.search(seg) or _ADDR_RE.search(seg))


def _mailobj_at(cased: str, low: str, pos: int) -> bool:
    """Is a mail-shaped object (mail word, address, NAME, addressee
    placeholder, message-from-NAME) at *pos*?"""
    pre = _MAILOBJ_PREFIX_RE.match(low, pos)
    p = pre.end() if pre else pos
    if _MAILOBJ_MAIL_RE.match(low, p) or _ADDR_RE.match(low, p):
        return True
    if _name_at(cased, p) >= 0:
        return True
    if _template_at(cased, p, mail_ctx=_mail_in_sentence(low, p)) >= 0:
        return True
    msg = _MAILOBJ_MSG_RE.match(low, p)
    if msg:
        q = msg.end()
        return _addressee_at(cased, low, q, mail_ctx=False) >= 0
    return False


def _clause_window(low: str, pos: int) -> tuple[int, int]:
    """The rest of the sentence after *pos*, at most 80 chars."""
    end = min(len(low), pos + 80)
    m = re.search(r"[.!?;](?=\s|$)|\n", low[pos:end])
    return pos, (pos + m.start() if m else end)


def _userish_addressee(low: str, pos: int) -> bool:
    m = re.match(r"\s*(?:(?:to|for|na|za)\s+)?", low[pos:])
    q = pos + (m.end() if m else 0)
    return bool(_USERISH_RE.match(low, q))


def _reply_noun_ok(cased: str, low: str, pos: int, *, hr: bool, verb: int = -1) -> bool:
    """A reply/response/message/odgovor noun counts only with a mail-shaped
    addressee in the same sentence, never the user. *verb*: where the
    compose verb starts — a sentence already about email before it ("When
    Ana emails, draft a reply") counts too."""
    if _userish_addressee(low, pos):
        return False
    a, b = _clause_window(low, pos)
    seg = low[a:b]
    if _MAILW_RE.search(seg) or _ADDR_RE.search(seg):
        return True
    if verb >= 0 and _mail_in_sentence(low, verb) and not _VENUE_RE.search(seg):
        return True
    if hr:
        for m in re.finditer(r"\b\w|\{\{", cased[a:b]):
            if _addressee_at(cased, low, a + m.start(), mail_ctx=False) >= 0:
                return True
        return False
    for m in re.finditer(r"\b(?:to|for)\s+", seg):
        if _addressee_at(cased, low, a + m.end(), mail_ctx=False) >= 0:
            return True
    return bool(_EN_PEOPLE_RE.search(seg)) and not _VENUE_RE.search(seg)


def _other_channel(low: str, pos: int) -> bool:
    """The clause after *pos* names a non-mail channel (WhatsApp, Slack …) and
    no mail word / address — a reply there isn't an email."""
    a, b = _clause_window(low, pos)
    seg = low[a:b]
    return bool(_CHANNEL_RE.search(seg)) and not (_REAL_MAIL_RE.search(seg) or _ADDR_RE.search(seg))


def _also_someone(cased: str, low: str, pos: int) -> bool:
    """After "me" at *pos*: "… and Ana", ", ana@x.co" — someone besides the user."""
    m = _ALSO_RE.match(low, pos)
    return bool(m) and m.end() > pos and _addressee_at(cased, low, m.end(), mail_ctx=True) >= 0


def _other_addressee(cased: str, low: str, pos: int, *, hr: bool) -> bool:
    """Does the clause after *pos* name an addressee besides the user?"""
    a, b = _clause_window(low, pos)
    if _ADDR_RE.search(low[a:b]):
        return True
    rx = r"\b\w|\{\{" if hr else r"\b(?:to|for)\s+"
    for m in re.finditer(rx, low[a:b]):
        q = a + (m.start() if hr else m.end())
        if _addressee_at(cased, low, q, mail_ctx=False) >= 0:
            return True
    return False


def _scopes(text: str) -> list[str]:
    cased, low = _normalize(text)
    if not low:
        return []
    out: list[str] = []

    # EN-1 compose (any)
    for rx in (_EN_COMPOSE_RE, _EN_COMPOSE_FREE_RE):
        for m in rx.finditer(low):
            if not _imperative_at(low, m.start()) or _negated_at(low, m.start()):
                continue
            if rx is _EN_COMPOSE_FREE_RE and _CHANNEL_RE.search(m.group(0)):
                continue  # "write a WhatsApp reply to Ana"
            if m.group("mail") or (
                _reply_noun_ok(cased, low, m.end(), hr=False, verb=m.start())
                and not _other_channel(low, m.end())
            ):
                out.append(_ANY)
    for m in _EN_COMPOSE_ME_RE.finditer(low):
        if _imperative_at(low, m.start()) and not _negated_at(low, m.start()):
            out.append(_ANY if _other_addressee(cased, low, m.end(), hr=False) else _SELF)
    for m in _EN_COMPOSE_TO_RE.finditer(low):
        if (_imperative_at(low, m.start()) and not _negated_at(low, m.start())
                and _name_at(cased, m.start("n")) >= 0):
            out.append(_ANY)

    # EN-2 reply (any)
    for m in _EN_REPLY_RE.finditer(low):
        if (not _imperative_at(low, m.start()) or _negated_at(low, m.start())
                or _other_channel(low, m.end())):
            continue
        if _mailobj_at(cased, low, m.end()) or (
            _EN_PRONOUN_OBJ_RE.match(low, m.end()) and _mail_in_sentence(low, m.start())
            and not _VENUE_RE.search(low, *_clause_window(low, m.end()))
        ):
            out.append(_ANY)

    # EN-3 send
    for m in _EN_SEND_NOUN_RE.finditer(low):
        if _negated_at(low, m.start()):
            continue
        after = low[m.start() + 4:]
        me = _EN_SELF_AFTER_RE.match(after)
        selfish = bool(me) or bool(_TO_SELF_RE.search(low[m.start():m.end()]))
        if me and _also_someone(cased, low, m.start() + 4 + me.end()):
            selfish = False
        out.append(_SELF if selfish else _ANY)
    for m in _EN_SEND_BY_RE.finditer(low):
        if _negated_at(low, m.start()):
            continue
        after = low[m.start() + 4:]
        me = _EN_SELF_AFTER_RE.match(after)
        selfish = bool(me) or bool(_TO_SELF_RE.search(m.group("mid")))
        if me and _also_someone(cased, low, m.start() + 4 + me.end()):
            selfish = False
        out.append(_SELF if selfish else _ANY)
    for m in _EN_SEND_TO_RE.finditer(low):
        if not _imperative_at(low, m.start()) or _negated_at(low, m.start()):
            continue
        if m.group("tpl") and not _template_addressish(m.group("tpl")):
            continue
        selfish = bool(m.group("my")) or bool(_EN_SELF_AFTER_RE.match(m.group("mid")))
        out.append(_SELF if selfish and not m.group("addr") else _ANY)
    for m in _EN_TOOL_WRITE_RE.finditer(low):
        # an instruction ("Use gmail to send …"), not a description of one
        # ("how often we use gmail to send newsletters")
        if _imperative_at(low, m.start()) and not _negated_at(low, m.start()):
            out.append(_ANY)

    # EN-4 email-as-verb
    for m in _EN_EMAIL_VERB_RE.finditer(low):
        if _negated_at(low, m.start()) or not _email_verb_at(cased, low, m.start()):
            continue
        if m.group("self") and not _also_someone(cased, low, m.end()):
            out.append(_SELF)
        else:
            out.append(_ANY)
    for m in _EN_MAIL_ME_RE.finditer(low):
        if _negated_at(low, m.start()) or _DETERMINER_TAIL_RE.search(low[:m.start()]):
            continue
        out.append(_ANY if _also_someone(cased, low, m.end()) else _SELF)
    for m in _EN_EMAIL_NAME_RE.finditer(low):
        if not _imperative_at(low, m.start()) or _negated_at(low, m.start()):
            continue
        end = _addressee_at(cased, low, m.end(), mail_ctx=True)
        if end >= 0 and _AFTER_NAME_OK_RE.match(low, end):
            out.append(_ANY)

    # HR-1 compose — "napiši mi mail" is self unless someone else is named
    for m in _HR_COMPOSE_RE.finditer(low):
        if (not _imperative_at(low, m.start(), hr=True, cased=cased)
                or _negated_at(low, m.start())):
            continue
        if m.group("mail") or (
            _reply_noun_ok(cased, low, m.end(), hr=True, verb=m.start())
            and not _other_channel(low, m.end())
        ):
            verb_end = re.match(r"\w+", low[m.start():]).end() + m.start()
            selfish = bool(_HR_SELF_RE.match(low, verb_end)) or bool(
                _HR_SELF_BEFORE_RE.search(low[:m.start()]))
            if selfish and not _other_addressee(cased, low, verb_end, hr=True):
                out.append(_SELF)
            else:
                out.append(_ANY)

    # HR-2 send
    for rx in (_HR_SEND_NOUN_RE, _HR_SEND_CHANNEL_RE):
        for m in rx.finditer(low):
            if _negated_at(low, m.start()):
                continue
            verb_end = re.match(r"\w+", low[m.start():]).end() + m.start()
            selfish = bool(_HR_SELF_RE.match(low, verb_end)) or bool(
                _HR_SELF_BEFORE_RE.search(low[:m.start()]))
            out.append(_SELF if selfish else _ANY)
    for m in _HR_SEND_TO_RE.finditer(low):
        if _negated_at(low, m.start()):
            continue
        if m.group("tpl") and not _template_addressish(m.group("tpl")):
            continue
        out.append(_ANY)

    # HR-3 reply (any)
    for m in _HR_REPLY_RE.finditer(low):
        if (not _imperative_at(low, m.start(), hr=True, cased=cased)
                or _negated_at(low, m.start()) or _other_channel(low, m.end())):
            continue
        if _mailobj_at(cased, low, m.end()) or (
            _HR_PRONOUN_OBJ_RE.match(low, m.end()) and _mail_in_sentence(low, m.start())
            and not _VENUE_RE.search(low, *_clause_window(low, m.end()))
        ):
            out.append(_ANY)

    return out


def intent_scope(text: str) -> str:
    """``"none"`` | ``"self"`` | ``"any"`` — does *text* explicitly ask for an email?"""
    found = _scopes(text)
    if not found:
        return _NONE
    return _ANY if _ANY in found else _SELF


def explicit_mail_intent(text: str) -> bool:
    return intent_scope(text) != _NONE


def narrower_intent(a: str, b: str) -> str:
    """``""`` unless both texts ask for an email; else the narrower-scoped one (*a* on a tie).

    A self-scoped result keeps only the addresses both texts name: a self
    job may also reach an address its text names (:func:`check_mail_write`),
    so an agent-written task ("email me at x@evil.co …") can't add one the
    user never wrote.

    Organizing mail (:func:`organize_intent`) needs both texts too. When
    they don't both ask for an email but both ask to organize, the result is
    the one without email intent (*a* first), so it never carries email
    authority; a result that asks to organize when the other text doesn't
    has its organize verbs blanked.
    """
    sa, sb = intent_scope(a), intent_scope(b)
    if sa == _NONE or sb == _NONE:
        if organize_intent(a) and organize_intent(b):
            return a if sa == _NONE else b
        return ""
    out, other = (b, a) if _RANK[sb] < _RANK[sa] else (a, b)
    if organize_intent(out) and not organize_intent(other):
        blanked = _without_organize(out)
        if organize_intent(blanked) or intent_scope(blanked) != intent_scope(out):
            return ""
        out = blanked
    return _keep_shared_addresses(out, other) if intent_scope(out) == _SELF else out


def _keep_shared_addresses(text: str, other: str) -> str:
    keep = job_addresses(other)
    return _ADDR_RE.sub(
        lambda m: m.group(0) if m.group(0).lower() in keep else "[address]", text,
    )


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------


# Mailbox tools with create_draft / update_draft / send / send_draft actions
# (and the MAILBOX_CHANGE_ACTIONS).
DRAFTING_MAIL_TOOLS = frozenset({"google_mail", "microsoft_mail"})


def is_mail_write(tool: str, action: str | None) -> bool:
    if tool in ("send_mail", "mcp_mail"):
        return True
    return tool in DRAFTING_MAIL_TOOLS and action in MAIL_WRITE_ACTIONS


def _verb(action: str | None) -> str:
    if action == "create_draft":
        return "created"
    if action == "update_draft":
        return "updated"
    return "sent"


def _label(kind: str) -> str:
    return KIND_LABELS.get(kind, KIND_LABELS["unknown"])


# Kinds whose text is a job the user wrote (or can edit): a refusal tells
# them how to make it write email, not who is waiting for a reply.
_JOB_KINDS = frozenset({"cron", "fd_scheduler", "flow", "flow_tool", "plan", "mcp_task"})
_PEER_KINDS = frozenset({"peer", "peer_relay"})
_TRIAGE_KINDS = frozenset({"autonomy", "autonomy_tool"})
# A connected system (agent link): its owner wrote the link's instructions in
# Flight Deck; what the system sends never counts as theirs.
_LINK_KINDS = frozenset({"link"})


def refusal_text(kind: str, verb: str) -> str:
    """The refusal for an automated turn that may not write email; what the
    model should tell the user depends on what started the turn."""
    head = (
        f"{MAIL_REFUSAL_TAG} Not {verb}: this turn was started by {_label(kind)}, not by "
        "the user, and its instructions don't ask for an email to be written or sent. "
        "Nothing was drafted or sent. Don't retry or work around this (no other tool, no "
        "send_mail) — "
    )
    if kind in _TRIAGE_KINDS:
        return head + "tell the user who is waiting for a reply and offer to draft it when they ask."
    if kind in _JOB_KINDS:
        return head + (
            "tell the user that this job's text doesn't ask to email anyone, so no email was "
            "written. If they want it to write email, they can edit the job to say so "
            "explicitly, e.g. 'email Ana the report' or 'pošalji Ani izvještaj mailom'."
        )
    if kind in _PEER_KINDS:
        return head + (
            "tell the user no email was written because the request didn't explicitly ask for "
            "one. They can ask for it in their own words, e.g. 'draft a reply to Ana'."
        )
    if kind in _LINK_KINDS:
        return head + (
            "say in your answer that no email was written: what the connected system sends "
            "can't ask for email, only the owner's instructions for this link in Flight Deck can."
        )
    return head + "say in your answer that no email was written because this run wasn't asked to write one."


def refusal_text_self(kind: str, verb: str) -> str:
    return (
        f"{MAIL_REFUSAL_TAG} Not {verb}: this turn was started by {_label(kind)}, and its "
        "instructions only ask for an email to the user themself. Write one new email "
        "addressed only to the user's own address or an address the instructions name (no "
        "other recipients, not a reply to someone else's message). Nothing was drafted or sent."
    )


def is_refusal(text: str) -> bool:
    return MAIL_REFUSAL_TAG in str(text or "")


def check_mail_write(
    tool: str,
    action: str | None,
    *,
    recipients: list[str] | None = None,
    own_addresses: set[str] | None = None,
) -> str | None:
    """``None`` to proceed, else the refusal text for this mail write."""
    if not is_mail_write(tool, action):
        return None
    a = current()
    if a.mode == "human":
        return None
    if a.mail_write == "allow":
        return None
    verb = _verb(action if tool in DRAFTING_MAIL_TOOLS else None)
    if a.mail_write != "intent":
        return refusal_text(a.kind, verb)
    sc = intent_scope(a.job_text)
    if sc == _NONE:
        return refusal_text(a.kind, verb)
    if sc == _ANY:
        return None
    # self scope: one new email to the owner — the mailbox's own address, or
    # an address the job's own text names ("email me at stevica@firma.hr").
    ok = (
        recipients is not None
        and _self_recipients_ok(recipients, own_addresses, job_addresses(a.job_text))
        and tool != "mcp_mail"
        and (tool not in DRAFTING_MAIL_TOOLS or action in ("create_draft", "send"))
    )
    return None if ok else refusal_text_self(a.kind, verb)


def check_notify_others(what: str) -> str | None:
    """``None`` to proceed, else the refusal for a calendar change that emails
    other people — Outlook sends invitations, updates, cancellations and
    replies to the organizer itself, the moment the event changes.

    Stricter than :func:`check_mail_write`: an automated turn needs an explicit
    allow (a human-approved action, a literal flow tool step) — a job's text
    asking for an email is not a request to invite anyone. *what* names the
    effect ("invitations to the attendees").
    """
    a = current()
    if a.mode == "human" or a.mail_write == "allow":
        return None
    return (
        f"{MAIL_REFUSAL_TAG} Not done: this change would email {what}, and this turn "
        f"was started by {_label(a.kind)}, not by the user. Nothing was changed. Don't "
        "retry or work around this — tell the user what you would do and let them ask for it."
    )


# ---------------------------------------------------------------------------
# Mailbox changes (archive / move / labels / categories / new label or folder)
# ---------------------------------------------------------------------------
# Not email writes: nothing is sent, so there is no recipient check. An
# automated turn may change the mailbox only when the job's own text asks to
# organize email (:func:`organize_intent`) or a human vouched for it
# ("allow"). Email intent never grants it, and organize intent never grants
# email (:func:`intent_scope` knows none of these verbs).

# EN verbs. Before ":" or a place noun it is a noun ("Archive folder: …"), and
# "move on / forward" moves nothing.
_ORG_EN_RE = re.compile(
    r"\b(?:(?:un)?archive|(?:re|un)?label|(?:un)?tag|(?:re)?categori[sz]e|(?:un)?star"
    r"|move(?!\s+(?:on|forward|ahead|along|over|up)\b))\b"
    r"(?!\s*:|\s+(?:folders?|labels?|categor(?:y|ies)|tags?)\b)"
)
# …and verbs that organize only with somewhere to put it ("file / sort … into
# X", "put / add … to the X folder", "mark … as read").
_ORG_EN_DEST_RE = re.compile(
    r"\b(?:(?:file|sort)(?=[^.!?;\n]{0,60}?\b(?:into|in|to|under)\s)"
    r"|(?:put|add)(?=[^.!?;\n]{0,60}?\b(?:in|into|under|to)\s+(?:[\w'-]+\s+){0,3}?"
    r"(?:folder|label|categor)\w*)"
    r"|mark(?=[^.!?;\n]{0,60}?\bas\s+(?:un)?(?:read|important|starred)\b))\b"
)
# Creating, applying or removing a label / folder / category ("create a label
# Invoices", "apply the Clients label to …"); *mod* is the word before the noun
# ("a Gmail label").
_ORG_EN_LABEL_RE = re.compile(
    r"\b(?:create|make|add|set up|apply|assign|remove)\s+(?:(?:a|an|the|new|my|our)\s+){0,2}"
    r"(?:(?P<mod>[\w'-]+)\s+)?(?:labels?|folders?|categor(?:y|ies)|tags?)\b"
)
# The mailbox as a whole ("organize my inbox") — not into a summary or a list.
_ORG_WHOLE_TAIL = (
    r"(?!\s+(?:into|in|as)\s+(?:(?:a|an|the|my)\s+)?(?:summary|list|report|digest|overview|table"
    r"|doc\w*|sazet\w*|pregled\w*|popis\w*|izvjestaj\w*|tablic\w*|dokument\w*)\b)"
)
_ORG_EN_WHOLE_RE = re.compile(
    r"\b(?:organi[sz]e|clean\s+up|tidy\s+up|declutter)\s+(?:(?:my|the|our)\s+)?"
    r"(?:e-?mails?|mails?|mailbox|inbox|gmail|outlook)\b" + _ORG_WHOLE_TAIL
)
_ORG_HR_RE = re.compile(
    r"\b(?:arhiviraj(?:te)?|arhivirati|premjesti(?:te)?|premjestaj(?:te)?|premjestiti"
    r"|prebaci(?:te)?|prebacuj(?:te)?|prebaciti|pomakni(?:te)?|makni(?:te)?"
    r"|kategoriziraj(?:te)?|kategorizirati)\b"
    # "spremi / stavi … u mapu X" — only with the place it goes
    r"|\b(?:spremi(?:te)?|stavi(?:te)?)\b(?=[^.!?;\n]{0,60}?\bu\s+(?:[\w'-]+\s+){0,2}?"
    r"(?:map[aeiu]\b|folder\w*|label\w*|oznak\w*|kategorij\w*|inbox\w*|arhiv\w*))"
)
# …and, like EN sort / file / mark, verbs that organize only with somewhere
# in the mailbox to put it: "razvrstaj / sortiraj … u mape / pod labelu X /
# po mapama" (not "po hitnosti", "u 8 sati", "u tri skupine"), "označi … kao
# pročitano / labelom X" (not "označi najvažnije").
_ORG_HR_BOX = (
    r"(?:map(?:a|e|i|u|ama)\b|folder\w*|label\w*|oznak\w*|arhiv\w*|inbox\w*|kategorij\w*"
    r"|sanduc\w*|direktorij\w*|spam\w*)"
)
_ORG_HR_DEST_RE = re.compile(
    r"\b(?:razvrstaj(?:te)?|razvrstavaj(?:te)?|razvrstati|sortiraj(?:te)?|sortirati)\b"
    r"(?=[^.!?;\n]{0,60}?\b(?:(?:u|pod)\s+(?:[\w'-]+\s+){0,2}?" + _ORG_HR_BOX
    + r"|po\s+(?:map|folder|label|oznak|kategorij)\w*))"
    r"|\b(?:oznaci(?:te)?|oznacavaj(?:te)?|oznaciti)\b"
    r"(?=[^.!?;\n]{0,60}?(?:\bkao\s+(?:ne)?(?:procitan\w*|vazn\w*|spam\w*|zvjezdic\w*)"
    r"|\b(?:label|oznak|kategorij|zvjezdic|zastavic)\w*))"
)
_ORG_HR_LABEL_RE = re.compile(
    r"\b(?:napravi(?:te)?|kreiraj(?:te)?|stvori(?:te)?|dodaj(?:te)?|ukloni(?:te)?|makni(?:te)?)\s+"
    r"(?:(?P<mod>[\w'-]+)\s+)?(?:label\w*|oznak\w*|map[aeu]\b|folder\w*|kategorij\w*)"
)
_ORG_HR_WHOLE_RE = re.compile(
    r"\b(?:organiziraj(?:te)?|pospremi(?:te)?|pocisti(?:te)?|sredi(?:te)?)\s+"
    r"(?:(?:mi|moj|moje|svoj|svoje)\s+)?"
    r"(?:inbox\w*|e-?mail\w*|mail\w*|mejl\w*|sanduc\w*|post[aeiu]\b)"
    + _ORG_WHOLE_TAIL
)
# (regex, HR verb, kind): "verb" needs mail context and a mail-like object,
# "label" mail context only, "whole" names the mailbox itself.
_ORG_VERBS = (
    (_ORG_EN_RE, False, "verb"), (_ORG_EN_DEST_RE, False, "verb"),
    (_ORG_EN_LABEL_RE, False, "label"), (_ORG_EN_WHOLE_RE, False, "whole"),
    (_ORG_HR_RE, True, "verb"), (_ORG_HR_DEST_RE, True, "verb"),
    (_ORG_HR_LABEL_RE, True, "label"), (_ORG_HR_WHOLE_RE, True, "whole"),
)
# "use google_mail to archive …", "via outlook action=move" — naming the tool
_ORG_TOOL_RE = re.compile(
    r"\b(?:use|using|call|via|with|through)\s+(?:the\s+)?"
    r"(?:google_mail|microsoft_mail|gmail|outlook)\b"
    r"(?:\s+tool)?(?:\s+action)?[\s=:'\",]*(?:to\s+)?"
    r"(?:(?:un)?archive|label|tag|move|file|sort|star|mark|categori[sz]e|organi[sz]e"
    r"|create_label|create_folder|add_labels|remove_labels|add_categories|remove_categories)\b"
)
# "make sure to archive …", "always label …", "uvijek arhiviraj …" — still a request
_ORG_ASK_TAIL_RE = re.compile(
    r"(?:^|[^\w])(?P<w>make sure (?:to|you)|be sure to|remember to|don'?t forget to|always"
    r"|uvijek|obavezno)\s+$"
)
# Mail context anywhere in the sentence ("newsletters", "in Gmail", an address)…
_ORG_MAIL_RE = re.compile(
    r"\b(?:e-?mail\w*|mail\w*|mejl\w*|gmail\w*|outlook\w*|inbox\w*|newsletter\w*|sanduc\w*)|"
    + _ADDR
)
# …or, after the verb, a place mail goes or another mail word ("move receipts
# into the Receipts folder", "arhiviraj poruke od Ane").
_ORG_PLACE_RE = re.compile(
    r"\b(?:folder\w*|label\w*|categor\w*|kategorij\w*|oznak\w*|map[aeiu]\b|messages?\b|poruk\w*"
    r"|threads?\b|spam\b|junk\b|promotions\b)"
)
_ORG_FROM_RE = re.compile(r"\b(?:from|od)\s+")
# What the verb acts on: up to four words before a preposition or the next clause.
_ORG_OBJ_RE = re.compile(
    r"(?:\s+(?!(?:from|to|into|in|under|on|as|with|by|for|at|of|than|that|which|who|and|or|then"
    r"|but|od|u|na|s|sa|iz|za|kao|i|pa|ali)\b)[\w'@.-]*\w){0,4}"
)
_ORG_WORD_RE = re.compile(r"[\w'@.-]*\w")
# Objects that aren't mail ("move the meeting", "archive the project files",
# "label the chart axes", "pomakni sastanak", "spremi izvještaj u mapu").
_ORG_NOT_MAIL_RE = re.compile(
    r"(?:files?|documents?|docs?|attachments?|photos?|pictures?|images?|videos?|notes?|tasks?"
    r"|todos?|cards?|tickets?|issues?|bugs?|prs?|commits?|branch(?:es)?|releases?|repos?"
    r"|repositor(?:y|ies)|projects?|meetings?|events?|appointments?|calls?|deadlines?|slides?"
    r"|sheets?|spreadsheets?|rows?|columns?|cells?|results?|data|charts?|axes|axis|points?"
    r"|summary|summaries|reports?|expenses?|transactions?|contacts?|code|cursor|pages?|posts?"
    r"|comments?|datotek\w*|dokument\w*|fajl\w*|privit\w*|slik\w*|fotografij\w*|biljesk\w*"
    r"|zadat\w*|zadac\w*|sastan\w*|termin\w*|dogadja\w*|izvjestaj\w*|sazet\w*|tablic\w*"
    r"|projekt\w*|prezentacij\w*)"
)
# Where an organize verb puts things that isn't the mailbox: another app
# ("file them in Notion", "tag @ana in the #support Slack channel", "na
# Slacku") or the answer itself ("label each in your reply", "move the urgent
# ones to the top of the summary", "označi važne u sažetku") — even when mail
# came earlier in the sentence. A mailbox place right after the noun keeps it
# ("to the Reports folder", "u mapu Izvještaji").
_ORG_APPS = (
    r"(?:slack\w*|notion\w*|(?:google\s+)?drive\w*|dropbox\w*|onedrive\w*|sharepoint\w*"
    r"|quickbooks\w*|xero\w*|zendesk\w*|intercom\w*|hubspot\w*|salesforce\w*|crm\w*|jira\w*"
    r"|trello\w*|asana\w*|confluence\w*|airtable\w*|excel\b|(?:google\s+)?sheets?\b"
    r"|spreadsheets?\b|todoist\w*|evernote\w*|obsidian\w*|clickup\w*|basecamp\w*|github\w*"
    r"|gitlab\w*|whats\s?app\w*|telegram\w*|teams\b|discord\w*|viber\w*|signal\b|messenger\w*"
    r"|sms\b|linkedin\w*|twitter\w*|calendar\w*|kalendar\w*|wiki\w*)"
)
_ORG_APP_WORD_RE = re.compile(_ORG_APPS)
# What lives in another app, not in the mailbox ("the Slack thread")
_ORG_APP_THING_RE = re.compile(
    r"(?:threads?|channels?|messages?|conversations?|chats?|dms?|boards?|workspaces?)"
)
_ORG_ANSWER_EN = (
    r"(?:summary|summaries|digest|recap|briefing|brief|report|reply|response|answer|list"
    r"|table|overview|output|doc|document|notes|chat)"
)
_ORG_ANSWER_HR = (
    r"(?:sazet\w*|izvjestaj\w*|pregled\w*|popis\w*|tablic\w*|odgovor\w*|dokument\w*"
    r"|biljesk\w*|list[iu]\b)"
)
_ORG_PLACE_WORD = (
    r"(?:folder\w*|label\w*|categor\w*|kategorij\w*|oznak\w*|map[aeiu]\b|inbox\w*|tags?\b"
    r"|arhiv\w*)"
)
_ORG_DEST_DET = (
    r"(?:the|a|an|my|our|your|their|his|her|its|this|that|these|those|each|every|any|some"
    r"|today'?s|[\w-]+'s|(?:moj|tvoj|nas|vas|svoj|njegov|njezin|njihov)"
    r"(?:a|e|i|u|o|oj|om|im|ima|eg|ega|em|emu)?|mom|tvom|svom|tom|ovom|taj|ovaj|tu|ovu|ta|ova)"
)
# A mailbox place right after the app / the answer noun keeps it in the
# mailbox ("to the Slack alerts folder", "to the top of my inbox", "u mapu
# Izvještaji") — not one in another phrase ("in the summary with a label",
# "in Slack and the inbox").
_ORG_NOT_PLACE_NEXT = (
    rf"(?!\s+(?:of\s+)?(?:{_ORG_DEST_DET}\s+)?(?:(?!(?:{_ORG_PLACE_WORD}|and|or|but|then|with|as"
    r"|of|in|into|on|to|at|by|for|from|not|i|ili|ali|pa|s|sa|kao|u|na|za|od|iz)\b)[\w'#@-]+\s+)?"
    rf"{_ORG_PLACE_WORD})"
)
# Between the preposition that puts it there (in / into / to / onto / on /
# under / u / na, or an app it goes through: "tag @ana via Slack") and the
# app or the answer noun, any words but a mailbox place ("in our team's main
# Slack channel", "in which Slack channel fits", "under a ticket with
# Zendesk", "in a column of the summary", "to the end of the list"). Never a
# source: what the mail came through ("emails that came through Zendesk",
# "sent via HubSpot"), a list of the mailbox's own ("from people in my
# contacts list"), or the mailbox the object sits in ("archive LinkedIn
# messages in my inbox").
_ORG_DEST_SKIP = (
    rf"(?:(?!(?:{_ORG_PLACE_WORD}|(?:via|through|kroz|preko)\b))[\w'#@-]+(?:\.[\w-]+)*\s+)"
)
# Only an explicit list of the mailbox's own ("people in my contacts list",
# "senders on my block list", "popis kontakata") isn't the answer — never an
# "email / mail / recipients list".
_ORG_OWN_LIST = (
    r"(?!(?:[\w'-]+\s+)?(?:contacts?|(?:block(?:ed)?|safe|allow(?:ed)?|trusted)(?:\s+senders?)?"
    rf"|mailing|distribution|vip|black|white)\s+lists?\b(?!\s+{_ORG_ANSWER_EN}\b)"
    r"|(?:popis\w*|list[aeiu])\s+(?:kontak\w*|blokiran\w*|sigurn\w*|vip)\b)"
)
_ORG_ELSEWHERE_RE = re.compile(
    r"\b(?P<prep>in|into|to|onto|on|under|inside|within|via|through|u|na|pod|unutar|kroz"
    rf"|preko)\s+{_ORG_DEST_SKIP}{{0,4}}?{_ORG_APPS}{_ORG_NOT_PLACE_NEXT}"
    rf"|\b(?:in|into|to|onto|on|at|under|inside|within)\s+{_ORG_DEST_SKIP}{{0,4}}?{_ORG_DEST_DET}"
    rf"\s+{_ORG_OWN_LIST}(?:[\w'#@-]+\s+){{0,2}}?{_ORG_ANSWER_EN}\b{_ORG_NOT_PLACE_NEXT}"
    rf"|\b(?:u|na|pod)\s+{_ORG_DEST_SKIP}{{0,3}}?{_ORG_OWN_LIST}{_ORG_ANSWER_HR}{_ORG_NOT_PLACE_NEXT}"
    rf"|\b(?:to|at|on)\s+(?:the\s+)?(?:very\s+)?(?:top|bottom)\b{_ORG_NOT_PLACE_NEXT}"
    r"|\bna\s+(?:sam\s+)?(?:vrh|dno|pocet)\w*"
)
# "emails that came through Zendesk", "sent via HubSpot": what the mail
# arrived through describes it — it isn't where the verb puts it. Only an
# arrival verb (not "posted / synced via …") right after the mail it
# describes, with no preposition that puts it somewhere before it ("tag @ana
# in an email sent via Slack", "tag them and have them sent via Slack").
_ORG_SOURCE_PREPS = frozenset({"via", "through", "kroz", "preko"})
_ORG_SOURCE_TAIL_RE = re.compile(
    r"\b(?:e-?mail|mail|mejl|newsletter)\w*\s+(?:(?:that|which|koji|koje|koja|sto)\s+)?"
    r"(?:(?:were|was|are|is|have\s+been|has\s+been|had\s+been|i|we|you|they|su|je|se)\s+)?"
    r"(?:(?:sent|came|come|comes|coming|arrived|arrive|arrives|arriving|received|got|get|gets"
    r"|delivered|stigl\w*|stize|stizu|poslan\w*|primljen\w*|dosl\w*|dolaz\w*)"
    r"(?:\s+(?:in|out|over|automatically|directly|here|izravno|automatski))?"
    r"|forwarded(?=\s+to\s+(?:me|us)\b))(?:\s+to\s+(?:me|us))?\s+$"
)
_ORG_PUT_PREP_RE = re.compile(r"\b(?:in|into|to|onto|on|under|inside|within|u|na|pod|unutar)\b")
# The mailbox the verb's object sits in or comes from, right after it
# ("LinkedIn messages in my inbox", "the Slack digests from my inbox"): mail,
# not another app's ("the Slack channel, not the emails in my inbox").
_ORG_IN_MAILBOX_RE = re.compile(
    r"\s+(?:in|from|out\s+of|inside|within|u|iz)\s+(?:[\w'-]+\s+){0,2}?"
    rf"(?:inbox\w*|mailbox\w*|gmail\w*|outlook\w*|sanduc\w*)(?!\s+{_ORG_ANSWER_EN}\b)"
)
# …looked for only up to the next instruction ("archive the newsletters and
# list the rest in the summary": the summary is the list's).
_ORG_NEXT_ACTION_RE = re.compile(
    r"(?:,|\b(?:and|then|but|pa|i|te|onda|zatim)\b)\s+(?:(?:then|also|just|please|onda|zatim"
    rf"|jos|i)\s+)?(?:{_LIST_VERB}|summari[sz]e|list|report|tell|notify|post|share|include"
    r"|show|give|check|review|highlight|mention|note|compile|let|ping|text|message|log|record"
    r"|save|export|copy|attach|upload|print|sazmi\w*|javi\w*|provjeri\w*|pregledaj\w*"
    r"|napravi\w*|dodaj\w*|ukljuci\w*|navedi\w*|izlistaj\w*|reci\w*|obavijesti\w*)\b"
)


def _org_imperative(cased: str, low: str, start: int, *, hr: bool) -> bool:
    """:func:`_imperative_at`, or after "make sure to" / "always" in one."""
    if _imperative_at(low, start, hr=hr, cased=cased if hr else ""):
        return True
    t = _ORG_ASK_TAIL_RE.search(low, 0, start)
    return bool(t) and _imperative_at(low, t.start("w"), hr=hr, cased=cased if hr else "")


def _org_not_mail(after: str, *, hr: bool) -> bool:
    """Is the verb's object something other than mail? The head noun decides
    (EN last word, HR first); a non-mail word with no mail word counts too,
    and so does another app's ("archive the Slack thread") — unless the
    mailbox comes right after it ("archive LinkedIn messages in my inbox")."""
    om = _ORG_OBJ_RE.match(after)
    words = _ORG_WORD_RE.findall(om.group(0)) if om else []
    if any(_ORG_APP_WORD_RE.fullmatch(w) and _ORG_APP_THING_RE.fullmatch(nxt)
           for w, nxt in zip(words, words[1:])) and not any(_ORG_MAIL_RE.search(w) for w in words):
        if not (om and _ORG_IN_MAILBOX_RE.match(after, om.end())):
            return True  # "the Slack thread" (but "the GitHub notifications" are mail)
    bad = [w for w in words if _ORG_NOT_MAIL_RE.fullmatch(w)]
    if not bad:
        return False
    head = words[0] if hr else words[-1]
    return head in bad or not any(_ORG_MAIL_RE.search(w) for w in words)


def _org_elsewhere(low: str, pos: int) -> bool:
    """Does the organize verb ending at *pos* put things somewhere that isn't
    the mailbox — another app or the answer ("file them in Notion", "label
    each in your reply")? Looked for up to the next instruction."""
    _, b = _clause_window(low, pos)
    nm = _ORG_NEXT_ACTION_RE.search(low, pos, b)
    p, end = pos, (nm.start() if nm else b)
    while True:
        em = _ORG_ELSEWHERE_RE.search(low, p, end)
        if not em:
            return False
        if em.group("prep") not in _ORG_SOURCE_PREPS:
            return True
        sm = _ORG_SOURCE_TAIL_RE.search(low, max(pos, em.start() - 80), em.start())
        if not sm or _ORG_PUT_PREP_RE.search(low, pos, sm.start()):
            return True
        # "emails that came through Zendesk" — but "… sent via Stripe into
        # Xero" still goes somewhere
        p = em.end("prep")


def _org_mail_context(cased: str, low: str, m: re.Match, *, hr: bool, kind: str) -> bool:
    """Is the organize verb *m* about email? Mail context in its sentence (the
    verb and the noun it creates not counted), no other channel, and — for a
    plain verb — an object that isn't a file, a meeting, a report …"""
    a, _ = _sentence(low, m.start())
    _, b = _clause_window(low, m.end())
    before, after = low[a:m.start()], low[m.end():b]
    mod = m.groupdict().get("mod") or ""
    mail = bool(_ORG_MAIL_RE.search(before) or _ORG_MAIL_RE.search(mod)
                or _ORG_MAIL_RE.search(after))
    if not mail and _CHANNEL_RE.search(low, a, b):
        return False  # "archive the Slack thread"
    if kind == "verb" and _org_not_mail(after, hr=hr):
        return False
    if mail or _ORG_PLACE_RE.search(after):
        return True
    return any(
        _addressee_at(cased, low, m.end() + f.end(), mail_ctx=False) >= 0
        for f in _ORG_FROM_RE.finditer(after)
    )


def organize_intent(text: str) -> bool:
    """Does *text* itself ask to archive, move, label / tag / categorize email,
    or to create a mail label or folder (EN + HR)? The verb must be in
    imperative position, not negated or a reminder (nor in a negated list,
    "don't archive, move or label …"), with mail in the same sentence and
    nothing put outside the mailbox ("… in Notion", "… in the summary") — or
    name the mail tool ("use gmail to archive …").

    Independent of :func:`intent_scope`: asking to organize mail never asks
    for an email, and asking for an email never asks to organize."""
    cased, low = _normalize(text)
    if not low:
        return False
    for m in _ORG_TOOL_RE.finditer(low):
        if _org_imperative(cased, low, m.start(), hr=False) and not _negated_at(low, m.start()):
            return True
    for rx, hr, kind in _ORG_VERBS:
        for m in rx.finditer(low):
            if not _org_imperative(cased, low, m.start(), hr=hr) or _negated_at(low, m.start()):
                continue
            if _org_elsewhere(low, m.end()):
                continue  # "… in Notion", "… in the summary", "to the top"
            if kind == "whole" or _org_mail_context(cased, low, m, hr=hr, kind=kind):
                return True
    return False


def _without_organize(text: str) -> str:
    """*text* (diacritics folded) with its organize verbs blanked: a judged
    text that keeps its email intent but asks to organize nothing."""
    cased, low = _normalize(text)
    rxs = [rx for rx, _hr, _kind in _ORG_VERBS] + [_ORG_TOOL_RE]
    out, last = [], 0
    for s, e in sorted(m.span() for rx in rxs for m in rx.finditer(low)):
        if s < last:
            continue
        out.append(cased[last:s] + "[-]")
        last = e
    return "".join(out) + cased[last:]


def is_mailbox_change(tool: str, action: str | None) -> bool:
    return tool in DRAFTING_MAIL_TOOLS and action in MAILBOX_CHANGE_ACTIONS


def _mailbox_verb(action: str | None) -> str:
    return {
        "archive": "archived", "move": "moved",
        "add_labels": "labeled", "add_categories": "categorized",
        "create_label": "created", "create_folder": "created",
    }.get(str(action or ""), "changed")


def mailbox_refusal_text(kind: str, verb: str) -> str:
    """The refusal for an automated turn that may not change the mailbox;
    like :func:`refusal_text`, what to tell the user depends on the kind."""
    head = (
        f"{MAIL_REFUSAL_TAG} Not {verb}: this turn was started by {_label(kind)}, not by "
        "the user, and its instructions don't ask to organize email (archive, move, label "
        "or categorize). Nothing in the mailbox was changed. Don't retry or work around this "
        "(no other action or tool) — "
    )
    if kind in _TRIAGE_KINDS:
        return head + (
            "tell the user which emails you would archive, move or label, and offer to do it "
            "when they ask."
        )
    if kind in _JOB_KINDS:
        return head + (
            "tell the user that this job's text doesn't ask to organize email, so nothing was "
            "changed. If they want it to, they can edit the job to say so explicitly, e.g. "
            "'archive newsletters from Substack' or 'premjesti račune u mapu Računi'."
        )
    if kind in _PEER_KINDS:
        return head + (
            "tell the user nothing was changed because the request didn't explicitly ask for "
            "it. They can ask for it in their own words, e.g. 'archive the newsletters from "
            "Substack'."
        )
    if kind in _LINK_KINDS:
        return head + (
            "say in your answer that nothing in the mailbox was changed: what the connected "
            "system sends can't ask for that, only the owner's instructions for this link in "
            "Flight Deck can."
        )
    return head + (
        "say in your answer that nothing in the mailbox was changed because this run wasn't "
        "asked to change it."
    )


def check_mailbox_change(tool: str, action: str | None) -> str | None:
    """``None`` to proceed, else the refusal for this mailbox change.

    Like :func:`check_mail_write` without recipients: a human turn and an
    ``allow`` turn proceed, ``deny`` refuses, ``intent`` needs the job's own
    text to ask to organize email (:func:`organize_intent`) — email intent
    alone (even "email me the summary") does not count."""
    if not is_mailbox_change(tool, action):
        return None
    a = current()
    if a.mode == "human" or a.mail_write == "allow":
        return None
    if a.mail_write == "intent" and organize_intent(a.job_text):
        return None
    return mailbox_refusal_text(a.kind, _mailbox_verb(action))


def job_addresses(text: str) -> set[str]:
    """Email addresses written literally in *text* (lower-cased)."""
    return {m.group(0).lower() for m in _ADDR_RE.finditer(_normalize(text)[1])}


def _self_recipients_ok(
    recipients: list[str], own_addresses: set[str] | None, named: set[str],
) -> bool:
    """Self scope: every recipient is the owner or an address the job names.

    With *own_addresses* None (send_mail: the owner's address is unknown) at
    most one recipient may be outside *named* — it is taken to be the owner.
    """
    rc = [str(r or "").strip().lower() for r in recipients]
    if not rc or any(not r for r in rc):
        return False
    others = [r for r in rc if r not in named]
    if own_addresses is None:
        return len(others) <= 1
    return len(others) <= 1 and all(r in own_addresses for r in others)


def needs_recipient_check() -> bool:
    """True only for an automated ``intent`` turn whose text asks for an email to the user."""
    a = current()
    return a.mode == "automated" and a.mail_write == "intent" and intent_scope(a.job_text) == _SELF


def parse_recipients(*fields: Any) -> list[str]:
    """Addresses from to/cc/bcc-style fields (strings or lists), lower-cased."""
    flat: list[str] = []
    for f in fields:
        if f is None:
            continue
        if isinstance(f, (list, tuple, set)):
            flat.extend(str(x) for x in f if x)
        elif str(f).strip():
            flat.append(str(f))
    out: list[str] = []
    for _name, addr in email.utils.getaddresses(flat):
        addr = (addr or "").strip().lower()
        if addr:
            out.append(addr)
    return out


# ---------------------------------------------------------------------------
# Soft checks + stored / forwarded intent
# ---------------------------------------------------------------------------


def human_turn_text(agent: Any) -> str:
    """The human's message for soft checks (bound text, else the agent's fallback)."""
    a = current()
    if a.mode != "human":
        return ""
    return a.job_text or str(getattr(agent, "_turn_user_text", "") or "")


# How far back "the user asked for this email" looks in a conversation: the
# request is often a turn or two before its "yes" ("Every Friday email Ana
# the report" → "Should I set it up?" → "yes").
RECENT_HUMAN_MESSAGES = 3
_RECENT_TEXT_MAX = JOB_TEXT_MAX
# member_privacy.TURN_INPUT: the first user message of each complete() /
# stream() carries it (later user-role messages of the call are nudges).
_TURN_INPUT_KEY = "turn_input"

# A short "yes" to the agent's own offer ("Shall I draft a reply to Ana?" →
# "yes do it", "da, napravi", "odgovori joj da može").
_AFFIRM_RE = re.compile(
    r"^\W*(?:yes|yeah|yep|yup|sure|ok|okay|k|please|pls|go ahead|go for it|do it|do that|do so"
    r"|sounds good|perfect|great|absolutely|of course|definitely|da|moze|mozes|naravno|svakako"
    r"|hajde|ajde|vazi|dogovoreno|super|odlicno|slazem se|tako je|izvoli"
    r"|draft|write|send|reply|respond|answer|email|mail|napravi|napisi|posalji|salji|odgovori"
    r"|sastavi|pripremi|kreiraj)\b"
)
_AFFIRM_NEG_RE = re.compile(
    r"^\W*(?:no|nope|nah|ne|nemoj\w*|not now|wait|cekaj|stani|stop|kasnije|later)\b"
    r"|\b(?:don'?t|do not|not|never|nemoj\w*|ne)\s+(?:\w+\s+)?(?:draft|write|send|reply|respond"
    r"|answer|e-?mail|mail|napravi|napisi|posalji|salji|saljes|odgovori|odgovaraj|sastavi)\b"
)
_AFFIRM_MAX_WORDS = 15
_OFFER_MAIL_RE = re.compile(
    r"\b(?:drafts?|e-?mails?|mails?|repl(?:y|ies)|respond|odgovor\w*|odgovori\w*|nacrt\w*"
    r"|mejl\w*|mail\w*)\b"
)
# The offer itself: an offer phrase followed closely by a write verb ("Shall
# I draft a reply?", "Want me to send it?", "Mogu li napisati odgovor?"), or
# a question that starts with one ("Draft a reply to Ana?"). A message that
# merely mentions email and ends with "Anything else?" isn't an offer.
_OFFER_WRITE = (
    r"(?:draft\w*|reply|respond|compose|write|send|create|e-?mail|forward|napis\w*|posal\w*"
    r"|posla\w*|odgovor\w*|sastav\w*|nacrt\w*|proslijed\w*)\b"
)
_OFFER_ASK_RE = re.compile(
    r"\b(?:shall i|should i|want me to|would you like(?: me to)?|do you want(?: me to)?"
    r"|let me know if you(?:'d)? (?:like|want)(?: me to)?|i can|i could|(?:i'?d be )?happy to"
    r"|if you(?:'d)? like,? i(?: can|'ll| will)?|mogu(?: li)?|mogao bih|mogla bih"
    r"|(?:zelis|zelite|hoces|hocete|trebam) li(?: da)?|da li da|ako zelis(?: da)?"
    r"|ako zelite(?: da)?|javi ako|javite ako)\s+(?:[\w'-]+\s+){0,2}?" + _OFFER_WRITE
    + r"|(?:^|[.!?\n]\s*)" + _OFFER_WRITE + r"[^.!?\n]{0,80}\?"
)


def _session_messages(agent: Any) -> list:
    msgs = getattr(getattr(agent, "session", None), "messages", None)
    return msgs if isinstance(msgs, list) else []


def _human_input_text(msg: Any) -> str | None:
    """A user message that is a person's own words: a turn's first input,
    not an automated turn (provenance prefix) or an FD envelope."""
    if not isinstance(msg, dict) or msg.get("role") != "user" or msg.get(_TURN_INPUT_KEY) is not True:
        return None
    text = str(msg.get("content") or "")
    if not text.strip() or text.lstrip().startswith("["):
        return None
    return text


def _current_input_index(msgs: list, current_text: str) -> int:
    """Where this turn's own input sits in the session; -1 when it isn't there."""
    cur = current_text.strip()
    if not cur:
        return -1
    for i in range(len(msgs) - 1, -1, -1):
        m = msgs[i]
        if (isinstance(m, dict) and m.get("role") == "user" and m.get(_TURN_INPUT_KEY) is True
                and str(m.get("content") or "").rstrip().endswith(cur)):
            return i
    return -1


def _recent_human(agent: Any, current_text: str, n: int) -> tuple[list[str], int]:
    """``(texts newest first, index of this turn's input or -1)``.

    Earlier messages are read only when this turn's own message is found in
    the agent's session — the history it belongs to.
    """
    out = [current_text] if current_text.strip() else []
    msgs = _session_messages(agent)
    idx = _current_input_index(msgs, current_text)
    if idx < 0:
        return out[:n], -1
    for m in reversed(msgs[:idx]):
        if len(out) >= n:
            break
        t = _human_input_text(m)
        if t is not None:
            out.append(t)
    return out[:n], idx


def recent_human_texts(agent: Any, n: int = RECENT_HUMAN_MESSAGES) -> list[str]:
    """The human's latest messages in this conversation, newest first (soft
    checks only; empty in an automated turn)."""
    if current().mode != "human":
        return []
    return _recent_human(agent, human_turn_text(agent), n)[0]


def _affirms_mail_offer(agent: Any, current_text: str) -> bool:
    """Is *current_text* a short yes to the agent's preceding offer of an email?"""
    _cased, low = _normalize(current_text)
    if not low or len(low.split()) > _AFFIRM_MAX_WORDS:
        return False
    if not _AFFIRM_RE.search(low) or _AFFIRM_NEG_RE.search(low):
        return False
    msgs = _session_messages(agent)
    idx = _current_input_index(msgs, current_text)
    if idx < 0:
        return False
    for m in reversed(msgs[:idx]):
        if not isinstance(m, dict):
            continue
        if m.get("role") == "user":
            return False
        if m.get("role") != "assistant" or not str(m.get("content") or "").strip():
            continue
        _c, offer = _normalize(str(m.get("content") or "")[-1500:])
        return bool(_OFFER_MAIL_RE.search(offer) and _OFFER_ASK_RE.search(offer))
    return False


def human_asked_for_mail(agent: Any) -> bool:
    """Soft check for a human turn: did the user ask for an email in this
    conversation — in one of their last messages, or by saying yes to the
    agent's offer to write one? False in an automated turn."""
    if current().mode != "human":
        return False
    texts = recent_human_texts(agent)
    if any(explicit_mail_intent(t) for t in texts):
        return True
    return bool(texts) and _affirms_mail_offer(agent, texts[0])


def nudge_mail_ok(agent: Any) -> bool:
    """May the tool-avoidance nudge push a draft? Never in an automated turn."""
    return human_asked_for_mail(agent)


def stall_mail_ok(agent: Any) -> bool:
    """May the stall nag push the model to act on an email?"""
    a = current()
    if a.mode == "human":
        return human_asked_for_mail(agent)
    if a.mail_write == "allow":
        return True
    return a.mail_write == "intent" and intent_scope(a.job_text) != _NONE


def intent_source_text(agent: Any) -> str:
    """Text whose email intent may be stored (cron) or forwarded (peers).

    Only the ContextVar — never ``agent._turn_user_text`` (concurrent turns and
    orchestrator workers would leak through it).
    """
    del agent
    a = current()
    if a.mode == "human":
        return a.job_text
    if a.mail_write == "deny":
        return ""
    return a.job_text


def recent_intent_text(agent: Any) -> str:
    """Text whose email intent an agent-created cron job stores: in a human
    turn the user's last few messages (newest first — the request and its
    confirmation, "Every Friday email Ana the report" … "yes, set it up"),
    else :func:`intent_source_text`.

    Only the bound message and the session it belongs to — never
    ``agent._turn_user_text``.
    """
    a = current()
    if a.mode != "human":
        return intent_source_text(agent)
    texts = _recent_human(agent, a.job_text, RECENT_HUMAN_MESSAGES)[0]
    return "\n".join(texts)[:_RECENT_TEXT_MAX]


def cron_job_text(payload: dict | None) -> str:
    """The text a fired agent cron prompt job is judged on."""
    if not isinstance(payload, dict):
        return ""
    text = str(payload.get("text") or "")
    if payload.get("author") == "agent":
        return narrower_intent(text, str(payload.get("mail_intent_text") or ""))
    return text


def automated_prefix() -> str:
    """The provenance prefix for the bound automated turn; ``""`` when human."""
    a = current()
    if a.mode == "human":
        return ""
    return AUTOMATED_TURN_PREFIX.format(label=_label(a.kind))


_PREFIX_LEAD = AUTOMATED_TURN_PREFIX.split("{label}", 1)[0]


def strip_automated_prefix(text: str) -> str:
    """*text* without a leading :data:`AUTOMATED_TURN_PREFIX` line (for
    detectors that look at how the job's own text starts)."""
    s = str(text or "")
    if s.startswith(_PREFIX_LEAD):
        head, sep, rest = s.partition("\n")
        if sep and head.endswith("]"):
            return rest
    return s
