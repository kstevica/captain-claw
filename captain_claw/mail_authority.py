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
# Names shared with Flight Deck (by wire and string, never by import)
# ---------------------------------------------------------------------------

AUTOMATION_KINDS = frozenset({
    "autonomy", "autonomy_tool", "plan", "fd_scheduler", "flow", "flow_tool", "cron",
    "peer", "peer_relay", "sister", "botport", "fd_worker", "being", "mcp_task", "unknown",
})
MAIL_WRITE_ACTIONS = frozenset({"create_draft", "update_draft", "send", "send_draft"})
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
_WORKER_ENVS = ("CLAW_BASNA_WORKER", "CLAW_VATRA_WORKER", "CLAW_COUNCIL_WORKER", "CLAW_CODE_AGENT")


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
_NEG_TAIL_RE = re.compile(
    r"(?:^|[^\w])(?:not|don'?t|do not|never|no|without|nemoj|nemojte|ne|nikad|nikada|bez)"
    r"(?:\s+\w+)?\s*$"
)
# "Do not, under any circumstances, email Ana" — a negation with an aside
_NEG_ASIDE_RE = re.compile(
    r"(?:^|[^\w])(?:don'?t|do not|never|nemoj|nemojte|nikad|nikada)\s*,[^.!?;\n]{0,80},\s*$"
)

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
    r"(?:e-?mail|mail|gmail|google_mail|send_mail|outlook)\b"
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
    r"\b(?:use|using|call|via|with|through)\s+(?:the\s+)?(?:google_mail|send_mail|gmail)\b"
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


def _normalize(text: str) -> tuple[str, str]:
    """``(cased, low)`` — diacritics folded, whitespace collapsed, same length."""
    s = str(text or "").replace("đ", "dj").replace("Đ", "Dj")
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
    """A negation ("don't send") or a reminder ("remind me to send") before *start*."""
    head = low[:start]
    return bool(_NEG_TAIL_RE.search(head) or _REMIND_TAIL_RE.search(head)
                or _NEG_ASIDE_RE.search(head))


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
            if not _imperative_at(low, m.start()):
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
        if _imperative_at(low, m.start()) and _name_at(cased, m.start("n")) >= 0:
            out.append(_ANY)

    # EN-2 reply (any)
    for m in _EN_REPLY_RE.finditer(low):
        if not _imperative_at(low, m.start()) or _other_channel(low, m.end()):
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
        if _negated_at(low, m.start()) or _DETERMINER_TAIL_RE.search(low[:m.start()]):
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
        if not _imperative_at(low, m.start(), hr=True, cased=cased):
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
                or _other_channel(low, m.end())):
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
    """
    sa, sb = intent_scope(a), intent_scope(b)
    if sa == _NONE or sb == _NONE:
        return ""
    out, other = (b, a) if _RANK[sb] < _RANK[sa] else (a, b)
    return _keep_shared_addresses(out, other) if intent_scope(out) == _SELF else out


def _keep_shared_addresses(text: str, other: str) -> str:
    keep = job_addresses(other)
    return _ADDR_RE.sub(
        lambda m: m.group(0) if m.group(0).lower() in keep else "[address]", text,
    )


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------


def is_mail_write(tool: str, action: str | None) -> bool:
    if tool in ("send_mail", "mcp_mail"):
        return True
    return tool == "google_mail" and action in MAIL_WRITE_ACTIONS


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
    verb = _verb(action if tool == "google_mail" else None)
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
        and (tool != "google_mail" or action in ("create_draft", "send"))
    )
    return None if ok else refusal_text_self(a.kind, verb)


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
