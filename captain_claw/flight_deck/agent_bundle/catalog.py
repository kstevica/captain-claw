"""What a process agent is made of, part by part — what an agent bundle may carry.

Only what is listed here ever goes into a bundle or comes out of one: the
owner marker (``.fd-owner``), ``.env``, ``config.yaml`` (Flight Deck writes a
new one from ``fd/agent.json``), logs, caches, spools, the owner's profile and
indexes that are rebuilt from the rest (``data/home-config/memory.db``) are not
listed, so they never travel. An import writes only the paths listed here.

Each part has a **category**; the categories are what an export picks from:

* ``identity``   — always: who it is (the model settings and instructions travel in ``fd/agent.json``)
* ``onboarding`` — its documents, lessons, skills, practice and exams
* ``memory``     — what it learned: insights, intentions, reflections, intuitions, playbooks…
* ``data``       — its datastore tables
* ``conversations`` — its sessions, topics and Flight Deck's chat log
* ``files``      — its workspace files
* ``schedules``  — its scheduled tasks (Flight Deck's, and its own cron jobs)

``secrets`` is not a category one picks: tables and columns marked so hold
keys or passwords and are emptied in every bundle that isn't an encrypted
backup.
"""

from __future__ import annotations

from dataclasses import dataclass, field

HC = "data/home-config-parent/.captain-claw"   # a process agent's own ~/.captain-claw
WORKSPACE = "data/workspace"

CATEGORIES = ("identity", "onboarding", "memory", "data", "conversations", "files", "schedules")
SECRETS = "secrets"

CATEGORY_LABELS = {
    "identity": "Identity",
    "onboarding": "Onboarding",
    "memory": "What it learned",
    "data": "Data tables",
    "conversations": "Conversations",
    "files": "Files",
    "schedules": "Scheduled tasks",
}


@dataclass(frozen=True)
class Part:
    key: str          # stable id; a store's file in the bundle is stores/<key>.db
    category: str
    kind: str         # "file" | "dir" | "sqlite"
    path: str         # relative to the agent's folder
    label: str
    # dir: sub-folders (relative to ``path``) that are another part's, or never travel
    skip: tuple[str, ...] = ()
    # sqlite: tables that belong to another category than the part's own
    tables: dict[str, str] = field(default_factory=dict)
    # sqlite: columns that hold secrets, per table (emptied unless the bundle carries secrets)
    secret_columns: dict[str, tuple[str, ...]] = field(default_factory=dict)


PARTS: tuple[Part, ...] = (
    # identity
    Part("personality", "identity", "file", f"{HC}/personality.md", "Personality"),
    Part("cognitive_mode", "identity", "file", f"{HC}/cognitive_mode.txt", "How it thinks"),
    Part("mrav_mode", "identity", "file", f"{HC}/mrav_mode.txt", "Small-model runtime switch"),
    Part("nano_mode", "identity", "file", f"{HC}/nano_mode.txt", "Nano mode switch"),
    # onboarding
    Part("setup_record", "onboarding", "file", ".owner-setup.json", "Onboarding record"),
    Part("lessons", "onboarding", "file", f"{HC}/lessons.json", "Lessons"),
    Part("skills", "onboarding", "dir", f"{HC}/skills", "Skills"),
    Part("legacy_skills", "onboarding", "dir", "data/skills", "Skills (older folder)"),
    Part("knowledge", "onboarding", "dir", f"{WORKSPACE}/knowledge", "Documents"),
    Part("practice", "onboarding", "file", "practice.json", "Practice tasks"),
    Part("exams", "onboarding", "dir", "exams", "Exams"),
    # memory
    Part("insights", "memory", "sqlite", f"{HC}/insights.db", "Insights"),
    Part("intentions", "memory", "sqlite", "data/sessions/intentions.db", "Intentions"),
    Part("reflections", "memory", "dir", f"{HC}/reflections", "Reflections"),
    Part("intuitions", "memory", "sqlite", f"{HC}/intuitions.db", "Intuitions"),
    # data
    Part("datastore", "data", "sqlite", f"{HC}/datastore.db", "Datastore tables"),
    # conversations — sessions.db also keeps what the agent learned to do
    # (playbooks, scripts, APIs, browser workflows), its contacts and its own cron jobs.
    Part("sessions", "conversations", "sqlite", "data/sessions/sessions.db", "Sessions",
         tables={
             "playbooks": "memory", "scripts": "memory", "apis": "memory",
             "browser_workflows": "memory", "contacts": "memory", "contact_mentions": "memory",
             "cron_jobs": "schedules",
             "browser_credentials": SECRETS,
         },
         secret_columns={"apis": ("credentials",), "direct_api_calls": ("auth_token",)}),
    Part("topics", "conversations", "sqlite", f"{HC}/conversation_topics.db", "Topics"),
    Part("sister_sessions", "conversations", "sqlite", f"{HC}/sister_sessions.db", "Side sessions"),
    Part("saved_attribution", "conversations", "sqlite", "data/sessions/saved_attribution.db",
         "Who saved which file"),
    # files
    Part("workspace", "files", "dir", WORKSPACE, "Workspace files", skip=("knowledge",)),
)

BY_KEY = {p.key: p for p in PARTS}

# Never inside a copied folder, whatever it is called there.
SKIP_NAMES = frozenset({".DS_Store", "__pycache__", ".fd-owner"})


def part_for_path(rel: str) -> Part | None:
    """The file or folder part a bundle path (``agent/<rel>``) belongs to —
    None for anything else, which an import never writes."""
    rel = rel.strip("/")
    best: Part | None = None
    for p in PARTS:
        if p.kind == "sqlite":
            continue
        if rel == p.path and p.kind == "file":
            return p
        if p.kind == "dir" and rel.startswith(p.path + "/"):
            inner = rel[len(p.path) + 1:]
            if any(inner == s or inner.startswith(s + "/") for s in p.skip):
                continue
            if best is None or len(p.path) > len(best.path):   # the most specific folder
                best = p
    return best


def table_category(part: Part, table: str) -> str:
    return part.tables.get(table, part.category)


def wanted_tables(part: Part, tables: list[str], categories: set[str], *, secrets: bool = False) -> list[str]:
    """The tables of a store an export with *categories* keeps."""
    keep = set(categories) | ({SECRETS} if secrets else set())
    return [t for t in tables if table_category(part, t) in keep]


def normalize_categories(raw) -> set[str]:
    """The categories asked for; identity always."""
    got = {str(c) for c in (raw or []) if str(c) in CATEGORIES}
    return got | {"identity"}
