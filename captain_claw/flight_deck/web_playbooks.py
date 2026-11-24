"""Web playbooks — known ways to get something done on a site, in the person's Chrome.

A playbook is a recording Captain ships: "How long from A to B?" → open Google
Maps directions with A and B in the link, read the fastest route's time, say
it. The person's own recordings join the same list (browser_recordings.as_playbook);
this module holds the shipped seed, the rules every playbook keeps, and what
the agent's ``web_playbook`` tool (tools/web_playbook.py) needs to run one.

Seed files: ``flight_deck/playbooks/<category>.yaml``, each
``{"category": <file stem>, "playbooks": [...]}``. One playbook::

    id: maps.travel_time                 # <site>.<task>, unique across files
    title: How long from A to B
    says:                                # how people ask for it (2–8)
      - How long does it take to get from Zagreb to Split?
      - How far is the airport from here by car?
    kind: look                           # look = reads and answers; act = fills
                                         #   in and stops before `stop`
    where: any                           # any = a headless browser will do;
                                         #   prefer_chrome = public, but sites wall
                                         #   bots; chrome = needs the person signed in
    region: global                       # global | eu | two-letter country code
    sites: [google.com/maps]             # hosts (and path) it works on
    inputs:                              # name → spec, in the order to ask
      origin: {label: From, kind: place}
      destination: {label: To, kind: place}
      mode: {label: How, kind: choice, options: [driving, transit, walking, bicycling],
             default: driving}
      # a number takes min (0 unless set) and max: {label: Nights, kind: number, min: 0, max: 60}
    open: https://www.google.com/maps/dir/?api=1&origin={origin}&destination={destination}&travelmode={mode}
    steps:                               # optional, the recordings' step vocabulary
      - kind: do
        text: If a departure time was given, set "Depart at" to it first.
    read:                                # what to take off the page (field → what it is)
      duration: The fastest route's time with traffic now
      distance: Its distance
    answer: "{duration} by {mode} ({distance})."     # optional
    stop: ""                             # act only: the button never pressed
    needs: []                            # other tools it leans on
    tags: [directions]

Values go into ``open`` URL-encoded; a date is ``YYYY-MM-DD``, a time
``HH:MM`` (24 h). The one exception is a site the person names (their bank,
carrier, utility): ``open: "{site}"`` with ``site`` a ``url`` input, put in
as given and checked to be https when it runs; such a playbook needs no
``sites`` — never a web search for a sign-in page, where an ad or a look-alike
can come first. An optional input used in ``open`` has a default (``""``
leaves its parameter empty). An input's ``ask`` is how the value is got, as on
a recording step: ``before`` (asked up front — the default for one without a
default), ``during`` (asked when the run gets there) or ``agent`` (the agent's
call — the default for one with a default).

Rules no playbook breaks: an ``act`` playbook names its ``stop`` and the run
ends with that page in front of the person; no step types a password or other
secret; nothing buys, pays, trades, sends or submits on its own.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

import yaml

SEED_DIR = Path(__file__).with_name("playbooks")

# The recordings' vocabulary (flight_deck/browser_recordings.py), kept in step
# by a test rather than an import: that module pulls the deck's web stack in.
STEP_KINDS = ("open", "switch", "click", "type", "select", "check", "upload", "key", "do")
ASK_MODES = ("before", "during", "agent")
STEP_KEYS = {"Enter", "Escape", "Ctrl+Enter", "⌘+Enter"}

KINDS = ("look", "act")
WHERE = ("any", "prefer_chrome", "chrome")
INPUT_KINDS = ("text", "place", "date", "time", "number", "choice", "url", "file", "flag")
NEEDS = ("calendar", "mail", "files", "memory", "contacts", "web_search", "location")
CATEGORIES = ("travel", "local", "shopping", "accounts", "work", "research",
              "money", "home", "media", "regional")

_ID = re.compile(r"^[a-z0-9]+(?:\.[a-z0-9_]+)+$")
_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_SLOT = re.compile(r"\{([a-z][a-z0-9_]*)\}")
_REGION = re.compile(r"^(?:global|eu|[a-z]{2})$")
_SITE = re.compile(r"^[a-z0-9.-]+\.[a-z]{2,}(?:/[^\s]*)?$")
_TAG = re.compile(r"^[a-z0-9][a-z0-9-]{0,30}$")


class PlaybookError(ValueError):
    """What is wrong with one playbook, every problem at once."""

    def __init__(self, pid: str, problems: list[str]):
        self.pid = pid
        self.problems = problems
        super().__init__(f"{pid or '(no id)'}: " + "; ".join(problems))


def _s(value: Any) -> str:
    return " ".join(str(value).split()) if isinstance(value, (str, int, float)) else ""


def _slots(text: str) -> set[str]:
    return set(_SLOT.findall(text or ""))


def _site_host(site: str) -> str:
    return site.split("/", 1)[0].removeprefix("www.")


def _named_site(url: str, inputs: dict[str, dict[str, Any]]) -> bool:
    """``{site}`` alone, where site is a url input: the person names the site
    (their bank, carrier, utility). Put in as given, never URL-encoded, and
    the runner checks it is https before opening it."""
    m = _SLOT.fullmatch(url or "")
    return bool(m and inputs.get(m.group(1), {}).get("kind") == "url")


def _clean_inputs(raw: Any, problems: list[str]) -> dict[str, dict[str, Any]]:
    if raw in (None, {}):
        return {}
    if not isinstance(raw, dict):
        problems.append("inputs must be a mapping of name → spec")
        return {}
    out: dict[str, dict[str, Any]] = {}
    for name, spec in raw.items():
        name = str(name)
        if not _NAME.match(name):
            problems.append(f"input name {name!r} must be snake_case")
            continue
        if not isinstance(spec, dict):
            problems.append(f"input {name}: spec must be a mapping")
            continue
        label = _s(spec.get("label"))
        kind = _s(spec.get("kind")) or "text"
        item: dict[str, Any] = {"label": label, "kind": kind}
        if not label or len(label) > 60:
            problems.append(f"input {name}: label is required (≤ 60 chars)")
        if kind not in INPUT_KINDS:
            problems.append(f"input {name}: kind {kind!r} not in {INPUT_KINDS}")
        has_default = "default" in spec
        if has_default:
            item["default"] = spec["default"] if isinstance(spec["default"], (bool, int, float)) else _s(spec["default"])
        required = spec.get("required", not has_default)
        if not isinstance(required, bool):
            problems.append(f"input {name}: required must be true/false")
            required = True
        item["required"] = required
        ask = _s(spec.get("ask")) or ("agent" if has_default else "before")
        if ask not in ASK_MODES:
            problems.append(f"input {name}: ask {ask!r} not in {ASK_MODES}")
        item["ask"] = ask
        if kind == "choice":
            options = spec.get("options")
            if not isinstance(options, list) or len(options) < 2:
                problems.append(f"input {name}: a choice needs at least two options")
            else:
                item["options"] = [_s(o) for o in options]
                if has_default and item["default"] not in item["options"]:
                    problems.append(f"input {name}: default {item['default']!r} is not one of its options")
        elif "options" in spec:
            problems.append(f"input {name}: only a choice has options")
        if kind == "number":
            # No number a playbook takes is below 0 unless it says so (a form's
            # number box once scrolled itself to -51 nights).
            for key, fallback in (("min", 0), ("max", None)):
                bound = spec.get(key, fallback)
                if bound is None:
                    continue
                if isinstance(bound, bool) or not isinstance(bound, (int, float)):
                    problems.append(f"input {name}: {key} must be a number")
                else:
                    item[key] = bound
            if "min" in item and "max" in item and item["min"] > item["max"]:
                problems.append(f"input {name}: min is above max")
            if has_default and isinstance(item.get("default"), (int, float)) and range_problem(name, item, item["default"]):
                problems.append(f"input {name}: default {item['default']!r} is out of its range")
        elif "min" in spec or "max" in spec:
            problems.append(f"input {name}: only a number has min and max")
        hint = _s(spec.get("hint"))
        if hint:
            item["hint"] = hint[:200]
        out[name] = item
    return out


def _clean_steps(raw: Any, inputs: dict[str, dict[str, Any]], problems: list[str]) -> list[dict[str, Any]]:
    names = set(inputs)
    if raw in (None, []):
        return []
    if not isinstance(raw, list):
        problems.append("steps must be a list")
        return []
    out: list[dict[str, Any]] = []
    for n, step in enumerate(raw, 1):
        if not isinstance(step, dict):
            problems.append(f"step {n}: must be a mapping")
            continue
        kind = _s(step.get("kind"))
        if kind not in STEP_KINDS:
            problems.append(f"step {n}: kind {kind!r} not in {STEP_KINDS}")
            continue
        if step.get("secret"):
            problems.append(f"step {n}: a playbook never types a secret")
        text_fields = [str(step.get(k) or "") for k in ("text", "note", "value", "url")]
        target = step.get("target")
        if target is not None and not isinstance(target, dict):
            problems.append(f"step {n}: target must be a mapping (role, name, near, selector)")
            target = None
        if target:
            text_fields += [str(v) for v in target.values()]
        for slot in set().union(*(_slots(t) for t in text_fields)) - names:
            problems.append(f"step {n}: {{{slot}}} is not an input")
        if kind == "do" and not _s(step.get("text")):
            problems.append(f"step {n}: a do step needs text")
        if kind in ("open", "switch"):
            url = _s(step.get("url"))
            if not url.startswith("https://") and not _named_site(url, inputs):
                problems.append(f"step {n}: {kind} needs an https url (or a url input on its own)")
        if kind in ("click", "type", "select", "check", "upload") and not (target or _s(step.get("note"))):
            problems.append(f"step {n}: {kind} needs a target or a note saying what to {kind}")
        if kind in ("type", "select") and "value" not in step:
            problems.append(f"step {n}: {kind} needs a value")
        if kind == "key" and _s(step.get("key")) not in STEP_KEYS:
            problems.append(f"step {n}: key must be one of {sorted(STEP_KEYS)}")
        out.append({k: v for k, v in step.items() if v not in (None, "", [], {})})
    return out


def clean_playbook(raw: Any, category: str) -> dict[str, Any]:
    """One playbook as the deck keeps it. Raises PlaybookError with every problem."""
    if not isinstance(raw, dict):
        raise PlaybookError("", ["a playbook is a mapping"])
    problems: list[str] = []
    pid = _s(raw.get("id"))
    if not _ID.match(pid):
        problems.append("id must look like site.task (lowercase, dots, snake_case)")
    known = {"id", "title", "says", "kind", "where", "region", "sites", "inputs", "open", "steps",
             "read", "answer", "stop", "needs", "tags"}
    for extra in sorted(set(raw) - known):
        problems.append(f"unknown field {extra!r}")

    title = _s(raw.get("title"))
    if not 3 <= len(title) <= 80:
        problems.append("title is required (3–80 chars)")

    says = raw.get("says")
    if not isinstance(says, list) or not 2 <= len(says) <= 8 or not all(_s(x) for x in says):
        problems.append("says needs 2–8 phrasings")
        says = []
    says = [_s(x)[:160] for x in says]

    kind = _s(raw.get("kind"))
    if kind not in KINDS:
        problems.append(f"kind must be one of {KINDS}")
    where = _s(raw.get("where"))
    if where not in WHERE:
        problems.append(f"where must be one of {WHERE}")
    region = _s(raw.get("region")) or "global"
    if not _REGION.match(region):
        problems.append("region is global, eu or a two-letter country code")

    inputs = _clean_inputs(raw.get("inputs"), problems)
    names = set(inputs)
    url = _s(raw.get("open"))
    named_site = _named_site(url, inputs)

    sites = raw.get("sites") or []
    if not isinstance(sites, list) or (not sites and not named_site):
        problems.append("sites needs at least one host")
        sites = []
    sites = [_s(x).lower().removeprefix("https://").removeprefix("http://") for x in sites]
    for site in sites:
        if not _SITE.match(site):
            problems.append(f"site {site!r} is not a host (no scheme)")

    if named_site:
        pass                       # the person's own site: checked when it runs
    elif not url.startswith("https://"):
        problems.append("open must be an https link, or a url input on its own ({site})")
    else:
        for slot in _slots(url) - names:
            problems.append(f"open uses {{{slot}}}, which is not an input")
        host = (urlsplit(_SLOT.sub("x", url)).hostname or "").removeprefix("www.")
        if sites and not any(host == _site_host(s) or host.endswith("." + _site_host(s)) for s in sites):
            problems.append(f"open's host {host!r} is not one of sites")
        for slot in _slots(url):
            spec = inputs.get(slot)
            if spec and not spec["required"] and "default" not in spec:
                problems.append(f"optional input {slot} is used in open, so it needs a default")

    steps = _clean_steps(raw.get("steps"), inputs, problems)

    read = raw.get("read") or {}
    if not isinstance(read, dict):
        problems.append("read must be a mapping of field → what it is")
        read = {}
    read = {str(k): _s(v) for k, v in read.items()}
    for field, about in read.items():
        if not _NAME.match(field) or not about:
            problems.append(f"read field {field!r} needs a snake_case name and a description")
    if kind == "look" and not read:
        problems.append("a look playbook says what to read")

    answer = _s(raw.get("answer"))
    for slot in _slots(answer) - names - set(read):
        problems.append(f"answer uses {{{slot}}}, which is neither an input nor a read field")

    stop = _s(raw.get("stop"))
    if kind == "act" and not stop:
        problems.append("an act playbook names its stop (the button never pressed)")
    if kind == "look" and stop:
        problems.append("a look playbook has no stop")

    needs = raw.get("needs") or []
    if not isinstance(needs, list) or any(_s(x) not in NEEDS for x in needs):
        problems.append(f"needs are among {NEEDS}")
        needs = []
    tags = raw.get("tags") or []
    if not isinstance(tags, list) or any(not _TAG.match(_s(x)) for x in tags):
        problems.append("tags are short lowercase words")
        tags = []

    if problems:
        raise PlaybookError(pid, problems)
    return {
        "id": pid, "title": title, "category": category, "says": says, "kind": kind,
        "where": where, "region": region, "sites": sites, "inputs": inputs, "open": url,
        "steps": steps, "read": read, "answer": answer, "stop": stop,
        "needs": [_s(x) for x in needs], "tags": [_s(x) for x in tags],
    }


def check_seed(seed_dir: Path = SEED_DIR) -> tuple[list[dict[str, Any]], list[str]]:
    """Every seed playbook, cleaned, and every problem found (file: id: what)."""
    good: list[dict[str, Any]] = []
    problems: list[str] = []
    seen: dict[str, str] = {}
    for path in sorted(seed_dir.glob("*.yaml")):
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            problems.append(f"{path.name}: not YAML: {exc}")
            continue
        category = path.stem
        if category not in CATEGORIES:
            problems.append(f"{path.name}: category {category!r} not in {CATEGORIES}")
        if doc.get("category") != category:
            problems.append(f"{path.name}: category must say {category!r}")
        items = doc.get("playbooks")
        if not isinstance(items, list):
            problems.append(f"{path.name}: needs a playbooks list")
            continue
        for raw in items:
            try:
                book = clean_playbook(raw, category)
            except PlaybookError as exc:
                problems.extend(f"{path.name}: {exc.pid or '(no id)'}: {p}" for p in exc.problems)
                continue
            if book["id"] in seen:
                problems.append(f"{path.name}: {book['id']}: id already used in {seen[book['id']]}")
                continue
            seen[book["id"]] = path.name
            good.append(book)
    return good, problems


@lru_cache(maxsize=1)
def seed() -> tuple[dict[str, Any], ...]:
    """The shipped playbooks that pass the rules (a bad one is left out, not fatal)."""
    good, _ = check_seed()
    return tuple(good)


def by_id(pid: str) -> dict[str, Any] | None:
    return next((b for b in seed() if b["id"] == pid), None)


# ── Running one: finding it, its values, its link, its steps in words ─────
#
# Pure functions, used by the agent's web_playbook tool (tools/web_playbook.py)
# — nothing here opens a page.

_WORD = re.compile(r"\w+")
_QUIET_WORDS = frozenset(
    "a an the to of in on for from at by my me i is it how what when which who do does did can could "
    "get got show find much many with and or this that there be are was will would should please tell "
    "check see look up out want need".split())
HERE_WORDS = frozenset({"here", "me", "my location", "current location", "where i am", "my place", "home"})
MAX_VALUE_CHARS = 2000
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TIME = re.compile(r"^(\d{1,2}):(\d{2})$")
_QUOTED = re.compile(r"[\"“„]([^\"“”„]{1,80})[\"”“]")
_ROLE_WORDS = {"button": "button", "link": "link", "textbox": "field", "searchbox": "search box",
               "combobox": "field", "checkbox": "checkbox", "radio": "option", "tab": "tab",
               "menuitem": "menu item", "option": "option", "switch": "switch"}


# Words people use for the same thing, folded into one.
_SAME = {
    "parcel": "package", "shipment": "package", "packet": "package",
    "tracking": "track", "where's": "track", "cost": "price", "costs": "price", "prices": "price",
    "cheapest": "cheap", "cheaper": "cheap", "fly": "flight", "flying": "flight", "plane": "flight",
    "rail": "train", "trains": "train", "bill": "invoice", "receipt": "invoice", "statement": "invoice",
    "subscription": "subscription", "membership": "subscription", "renewal": "subscription",
    "far": "distance", "drive": "driving", "commute": "travel", "trip": "travel", "journey": "travel",
    "hours": "open", "opening": "open", "booking": "book", "reserve": "book", "reservation": "book",
    "weather": "forecast", "rain": "forecast", "temperature": "forecast",
}


def words(text: Any) -> list[str]:
    """Lowercase words without the filler ("how", "my", "the"), plurals cut,
    the usual synonyms folded into one ("parcel" → "package")."""
    out: list[str] = []
    for w in _WORD.findall(str(text or "").casefold().replace("_", " ")):
        if w in _QUIET_WORDS or len(w) < 2:
            continue
        w = _SAME.get(w, w)
        if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(_SAME.get(w, w))
    return out


def _book_words(book: dict[str, Any]) -> list[tuple[set[str], int]]:
    hosts = " ".join(s.split("/", 1)[0].removeprefix("www.").rsplit(".", 1)[0] for s in book["sites"])
    return [
        (set(words(book["id"].replace(".", " "))) | set(words(hosts)), 3),
        (set(words(book["title"])), 3),
        (set(words(" ".join(book["tags"]))), 2),
        # the examples name places and things ("Berlin"): a hint, not the job
        (set(words(" ".join(book["says"]))) | set(words(" ".join(f"{k} {v}" for k, v in book["read"].items()))), 1),
    ]


def search(query: str, *, region: str = "", category: str = "", kind: str = "", limit: int = 6,
           books: Any = None) -> list[dict[str, Any]]:
    """The playbooks that best fit *query* (what the person asked), best first.
    *region*: the person's country (two letters) — that country's playbooks
    and the global and EU ones; empty = all. *books*: another list to search
    (the person's recordings joined in); the seed by default."""
    wanted = list(dict.fromkeys(words(query)))
    if not wanted:
        return []
    region = (region or "").strip().lower()
    scored: list[tuple[float, str, dict[str, Any]]] = []
    for book in (seed() if books is None else books):
        if category and book.get("category") != category:
            continue
        if kind and book.get("kind") != kind:
            continue
        if region and book.get("region") not in ("global", "eu", region):
            continue
        fields = _book_words(book)
        score = 0.0
        hit = 0
        for w in wanted:
            best = max((weight for bag, weight in fields if w in bag), default=0)
            if not best:   # a word's start: "flight" ~ "flights", "invoice" ~ "invoices"
                best = max((weight - 1 for bag, weight in fields
                            if len(w) >= 4 and any(b.startswith(w) or w.startswith(b) for b in bag if len(b) >= 4)),
                           default=0)
            score += best
            hit += 1 if best else 0
        if not hit:
            continue
        score += hit / len(wanted)            # the share of the ask it covers breaks ties
        if region and book.get("region") == region:
            score += 1
        scored.append((score, book["id"], book))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [b for _s, _i, b in scored[:max(1, limit)]]


def check_values(book: dict[str, Any], given: dict[str, Any] | None) -> dict[str, Any]:
    """This run's values for *book*'s inputs: what the agent gave, checked and
    put in shape, defaults for the rest. ``{"values", "problems", "ask_before",
    "decide", "ask_during"}`` — the last three are input names still without a
    value: asked of the person before the run, the agent's own call, asked when
    a step gets there (only for inputs the link doesn't carry). ``out_of_range``:
    inputs given a number outside their range (a problem too) — the person's
    to fix, not the agent's."""
    given = dict(given or {})
    inputs = book["inputs"]
    in_open = _slots(book["open"])
    out: dict[str, Any] = {"values": {}, "problems": [], "ask_before": [], "decide": [], "ask_during": [],
                           "out_of_range": []}
    for name in given:
        if name not in inputs:
            known = ", ".join(inputs) or "none"
            out["problems"].append(f"{name!r} isn't one of its inputs ({known})")
    for name, spec in inputs.items():
        raw = given.get(name)
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            if "default" in spec:
                out["values"][name] = spec["default"]
            elif spec["required"]:
                if spec["ask"] == "during" and name not in in_open:
                    out["ask_during"].append(name)
                elif spec["ask"] == "agent":
                    out["decide"].append(name)
                else:
                    out["ask_before"].append(name)
            continue
        value, problem = _shape(name, spec, raw)
        if not problem and spec["kind"] == "number":
            problem = range_problem(name, spec, value)
            if problem:
                out["out_of_range"].append(name)
        if problem:
            out["problems"].append(problem)
        else:
            out["values"][name] = value
    return out


def range_words(spec: dict[str, Any]) -> str:
    """A number input's range in words ("at least 0", "0 to 100"), or ""."""
    low, high = spec.get("min"), spec.get("max")
    num = lambda n: str(int(n)) if float(n).is_integer() else str(n)  # noqa: E731
    if low is not None and high is not None:
        return f"{num(low)} to {num(high)}"
    if low is not None:
        return f"at least {num(low)}"
    return f"at most {num(high)}" if high is not None else ""


def range_problem(name: str, spec: dict[str, Any], value: Any) -> str:
    """What's wrong with a number outside *spec*'s min/max, or ""."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return ""
    low, high = spec.get("min"), spec.get("max")
    if (low is not None and num < low) or (high is not None and num > high):
        return f"{name} ({spec['label']}) is {range_words(spec)} (got {value})"
    return ""


def _shape(name: str, spec: dict[str, Any], raw: Any) -> tuple[Any, str]:
    kind = spec["kind"]
    label = spec["label"]
    if kind == "flag":
        if isinstance(raw, bool):
            return raw, ""
        low = str(raw).strip().lower()
        if low in ("true", "yes", "1", "on"):
            return True, ""
        if low in ("false", "no", "0", "off"):
            return False, ""
        return None, f"{name} ({label}) is true or false"
    if isinstance(raw, (dict, list)):
        return None, f"{name} ({label}) is one value, not a list"
    text = " ".join(str(raw).split())
    if len(text) > MAX_VALUE_CHARS:
        return None, f"{name} ({label}) is over {MAX_VALUE_CHARS} characters"
    if kind == "choice":
        options = spec.get("options") or []
        pick = next((o for o in options if o.casefold() == text.casefold()
                     or o.casefold() == text.casefold().replace(" ", "_")), None)
        if pick is None:
            return None, f"{name} ({label}) is one of: {', '.join(options)}"
        return pick, ""
    if kind == "date":
        if not _DATE.match(text):
            return None, f"{name} ({label}) is a date as YYYY-MM-DD (got {text!r})"
        return text, ""
    if kind == "time":
        m = _TIME.match(text)
        if not m or int(m.group(1)) > 23 or int(m.group(2)) > 59:
            return None, f"{name} ({label}) is a 24-hour time as HH:MM (got {text!r})"
        return f"{int(m.group(1)):02d}:{m.group(2)}", ""
    if kind == "number":
        try:
            num = float(text.replace(",", "."))
        except ValueError:
            return None, f"{name} ({label}) is a number (got {text!r})"
        return (str(int(num)) if num.is_integer() else str(num)), ""
    if kind == "url":
        low = text.lower()
        if "://" not in low:
            text = "https://" + text
        elif not low.startswith("https://"):
            return None, f"{name} ({label}) is an https:// address (got {text!r})"
        host = urlsplit(text).hostname or ""
        if " " in text or "." not in host:
            return None, f"{name} ({label}) is a web address like https://example.com (got {text!r})"
        return text, ""
    if kind == "place" and text.casefold().strip(" .!") in HERE_WORDS:
        return None, (f"{name} ({label}): “{text}” can't go in a link — give the town or address "
                      "(from what you know of where they are, or ask them)")
    return text, ""


def open_link(book: dict[str, Any], values: dict[str, Any]) -> str:
    """The link to open, with this run's values in it (URL-encoded; a site the
    person names goes in as given — check_values made it https)."""
    url = book["open"]
    m = _SLOT.fullmatch(url)
    if m and _named_site(url, book["inputs"]):
        return str(values.get(m.group(1)) or "")

    def put(slot: re.Match) -> str:
        value = values.get(slot.group(1), "")
        if isinstance(value, bool):
            value = "true" if value else "false"
        return quote(str(value), safe="")

    return _SLOT.sub(put, url)


def filled(text: Any, book: dict[str, Any], values: dict[str, Any]) -> str:
    """*text* with this run's values in its slots (‹label› for one not given)."""
    def put(slot: re.Match) -> str:
        name = slot.group(1)
        value = values.get(name)
        if value is True:
            return "yes"
        if value is False:
            return "no"
        if value not in (None, ""):
            return str(value)
        label = book["inputs"].get(name, {}).get("label", name)
        return f"‹{label}: not given›" if name in values else f"‹{label}›"

    return _SLOT.sub(put, str(text or ""))


def step_sentence(step: dict[str, Any], book: dict[str, Any], values: dict[str, Any]) -> str:
    """One step as an instruction, with this run's values in it."""
    fill = lambda v: filled(v, book, values)  # noqa: E731
    kind = step["kind"]
    target = step.get("target") or {}
    name, role = fill(target.get("name") or ""), _ROLE_WORDS.get(str(target.get("role") or ""), "")
    what = (f"the {role} “{name}”" if role else f"“{name}”") if name else (f"the {role}" if role else "")
    note = fill(step.get("note") or "")
    if not what and note:
        what, note = note[:1].lower() + note[1:], ""
    value = fill(step.get("value") or "")
    if kind == "do":
        line = fill(step.get("text"))
    elif kind == "open":
        line = f"Open {fill(step.get('url'))}"
    elif kind == "switch":
        line = f"Switch to the tab {fill(step.get('url') or step.get('title'))}"
    elif kind == "click":
        line = f"Click {what or 'it'}"
    elif kind == "type":
        line = f"Type “{value}” into {what or 'the field'}"
    elif kind == "select":
        line = f"Choose “{value}” in {what or 'the list'}"
    elif kind == "check":
        line = f"Tick {what or 'the box'}"
    elif kind == "upload":
        line = f"Upload {value} with {what or 'the file field'}"
    elif kind == "key":
        line = f"Press {step.get('key')}" + (f" in {what}" if what else "")
    else:
        line = ""
    return f"{line} — {note}" if note and line else line


def step_lines(book: dict[str, Any], values: dict[str, Any]) -> list[str]:
    return [f"{n}. {step_sentence(s, book, values)}" for n, s in enumerate(book["steps"], 1)]


# A first step starting with one of these has to be done before the page can answer.
_ACTION_WORDS = frozenset(
    "open type search for pick look find set go choose in scroll filter enter click fill apply hand pause select "
    "sign log press tap expand switch sort add upload".split())
_IF_WORDS = ("if ", "only ", "unless ", "when ", "where ")


def answers_as_opened(book: dict[str, Any]) -> bool:
    """Whether the page a look playbook's link opens may already hold the
    answer: no steps, or a first one that only applies sometimes ("If a time
    was given …") or says what to read — not one to do first ("Open …",
    "Type …"). Only then is a read of the page worth a call."""
    if book.get("kind") != "look" or not book.get("read"):
        return False
    steps = book.get("steps") or []
    if not steps:
        return True
    step = steps[0]
    said = " ".join(str(step.get("text") or step.get("note") or "").split()).lower()
    if said.startswith(_IF_WORDS):
        return True
    if step["kind"] != "do":
        return False
    return (said.split(" ", 1)[0].strip(",:;") if said else "") not in _ACTION_WORDS


def label_words(label: Any) -> str:
    return " ".join(_WORD.findall(str(label or "").casefold()))


def is_stop(label: Any, stop: Any) -> bool:
    """Whether a button named *label* is the playbook's *stop*: the same words,
    give or take a count after them ("Check in (2)") and punctuation."""
    a, b = label_words(label), label_words(stop)
    if not a or not b:
        return False
    if a == b:
        return True
    return a.startswith(b + " ") and all(w.isdigit() for w in a[len(b) + 1:].split())


def stop_clicks_before(book: dict[str, Any]) -> int:
    """How many times the steps click a button with the stop's own name before
    the end ("“Cancel subscription” only opens the reason dialog"): those clicks
    go through, the next one is the stop."""
    stop = book.get("stop") or ""
    if not stop:
        return 0
    count = 0
    for step in book["steps"][:-1]:
        name = (step.get("target") or {}).get("name")
        if name and is_stop(name, stop):
            count += 1
            continue
        said = " ".join(str(step.get(k) or "") for k in ("text", "note"))
        if any(is_stop(q, stop) for q in _QUOTED.findall(said)):
            count += 1
    return count


def answer_text(book: dict[str, Any], values: dict[str, Any], fields: dict[str, Any]) -> str:
    """The playbook's answer line with what was read, or "" when it has none
    or a part of it wasn't found."""
    template = book.get("answer") or ""
    if not template:
        return ""
    known = {k: v for k, v in {**{k: v for k, v in values.items() if not isinstance(v, bool)},
                               **fields}.items() if v not in (None, "")}
    if not _slots(template) <= set(known):
        return ""
    return _SLOT.sub(lambda m: str(known[m.group(1)]), template)


if __name__ == "__main__":  # python -m captain_claw.flight_deck.web_playbooks [file.yaml ...]
    import sys
    from collections import Counter

    books, found = check_seed()
    wanted = set(sys.argv[1:])
    if wanted:
        found = [p for p in found if p.split(":", 1)[0] in wanted]
        books = [b for b in books if f"{b['category']}.yaml" in wanted]
    for line in found:
        print("PROBLEM", line)
    counts = Counter(b["category"] for b in books)
    print(f"{len(books)} good playbooks, {len(found)} problems  " + "  ".join(f"{k}:{v}" for k, v in sorted(counts.items())))
    sys.exit(1 if found else 0)
