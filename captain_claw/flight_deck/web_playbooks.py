"""Web playbooks — known ways to get something done on a site, in the person's Chrome.

A playbook is a recording Captain ships: "How long from A to B?" → open Google
Maps directions with A and B in the link, read the fastest route's time, say
it. The person's own recordings join the same list later ("my playbooks");
this module holds the shipped seed and the rules every playbook keeps.

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
from urllib.parse import urlsplit

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
