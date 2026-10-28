"""An agent's secrets reach a non-admin owner masked — the team's key above all.

An agent on the team tier set holds the deck's own model key in its .env
(`_build_env` resolves "@system" at spawn; `heal_keyless_agents` writes it
later), and the kiosk Settings dialog reads ``GET /fd/agent-config`` every time
it opens. Org keys never reach a browser, so for anyone but an admin:

* every secret-named .env value (``*_API_KEY``, ``*_TOKEN``, ``*_SECRET`` …)
  and every value holding a team key is a placeholder — dots and the last four
  characters — and a team key anywhere else (a comment, config.yaml) too;
* what the kiosk dialog needs from config.yaml (provider, model, endpoint) is
  still there;
* a save that sends a placeholder back keeps the value behind it — the mask is
  never written over the key — and a placeholder with nothing behind it is
  refused, with nothing saved;
* an admin (also one acting for the owner) and a deck without accounts read
  the files as they are.

Real FlightDeckDB + process registry in tmp dirs; ``DATA_DIR`` is never the
real ``./fd-data``.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import yaml

from captain_claw.flight_deck import auth as fd_auth
from captain_claw.flight_deck import rate_limiter, server
from captain_claw.flight_deck.auth import ACT_AS_HEADER, create_access_token, set_auth_db
from captain_claw.flight_deck.db import FlightDeckDB

ADMIN = "user-admin"
BOB = "user-bob"
REMOTE = ("203.0.113.9", 40001)

TEAM_KEY = "sk-ant-team-0123456789abcd"        # fd:provider-keys → anthropic
ENDPOINT_KEY = "gw-endpoint-key-5555XYZW"      # fd:endpoint-keys → a gateway
OWN_KEY = "brave-own-key-1234wxyz"             # the owner's own tool key

BOB_ENV = (
    "# written by Flight Deck\n"
    f"ANTHROPIC_API_KEY={TEAM_KEY}\n"
    f'BRAVE_API_KEY="{OWN_KEY}"\n'
    "OLLAMA_BASE_URL=http://127.0.0.1:11434\n"
    "CLAW_VFS_USER=user-bob\n"
    f"CUSTOM_HEADER=Bearer {TEAM_KEY}\n"
    f"# old one: {TEAM_KEY}\n"
    "SHORT_TOKEN=abc\n"
    "EMPTY_API_KEY=\n"
)
BOB_CONFIG = (
    "model:\n"
    "  provider: openai\n"
    "  model: gpt-4.1\n"
    f"  api_key: {ENDPOINT_KEY}\n"
    "  base_url: https://gw.example/v1\n"
    "  vision: true\n"
)


@pytest.fixture
async def deck(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Auth-enabled deck with team keys: Bob (a user) and the admin own one agent each."""
    prev = fd_auth._db
    db = FlightDeckDB(tmp_path / "fd.db")
    await db.init()
    set_auth_db(db)
    for uid, role in ((ADMIN, "admin"), (BOB, "user")):
        await db._db.execute(
            "INSERT INTO users (id, email, password_hash, display_name, role, created_at, updated_at)"
            " VALUES (?, ?, 'h', ?, ?, '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
            (uid, f"{uid}@x.co", uid, role))
    await db._db.commit()
    await db.set_system_setting("fd:provider-keys", json.dumps({"anthropic": TEAM_KEY}))
    await db.set_system_setting("fd:endpoint-keys", json.dumps({"openai@gw.example/v1": ENDPOINT_KEY}))

    data = tmp_path / "fd-data"
    data.mkdir()
    monkeypatch.setattr(server, "DATA_DIR", data)
    monkeypatch.setattr(server, "PROCESS_REGISTRY_FILE", data / ".processes.json")
    monkeypatch.setattr(server, "_processes", {})
    monkeypatch.setattr(server, "AUTH_ENABLED", True)
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    monkeypatch.setenv("FD_ALLOWED_HOSTS", "fd.test")
    monkeypatch.delenv("FD_LOCKDOWN", raising=False)
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
                 "XAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(rate_limiter, "_api_limiter", rate_limiter._SlidingWindow())

    server._save_process_registry({
        "bob-agent": {"slug": "bob-agent", "name": "Bob's agent", "web_port": 24101,
                      "web_auth": "bob-tok", "owner": BOB, "pid": None},
        "admin-agent": {"slug": "admin-agent", "name": "Admin's agent", "web_port": 24102,
                        "web_auth": "admin-tok", "owner": ADMIN, "pid": None},
    })
    for slug in ("bob-agent", "admin-agent"):
        (data / slug).mkdir()
        (data / slug / "config.yaml").write_text(BOB_CONFIG)
        (data / slug / ".env").write_text(BOB_ENV)
    try:
        yield db
    finally:
        await db.close()
        fd_auth._db = prev


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app, client=REMOTE), base_url="http://fd.test")


def _hdr(caller: str, act_as: str | None = None) -> dict:
    h = {"Authorization": f"Bearer {create_access_token(caller)}"}
    if act_as is not None:
        h[ACT_AS_HEADER] = act_as
    return h


def _files(slug: str) -> tuple[str, str]:
    d = server.DATA_DIR / slug
    return (d / "config.yaml").read_text(), (d / ".env").read_text()


async def _read(caller: str, slug: str = "bob-agent", act_as: str | None = None) -> dict:
    async with _client() as c:
        r = await c.get(f"/fd/agent-config/process/{slug}", headers=_hdr(caller, act_as))
    assert r.status_code == 200, r.text
    return r.json()


async def _save(caller: str, body: dict, slug: str = "bob-agent") -> httpx.Response:
    async with _client() as c:
        return await c.put(f"/fd/agent-config/process/{slug}", headers=_hdr(caller), json=body)


# ── what the owner reads ────────────────────────────────────────────────────

class TestOwnerReadsMasked:
    async def test_no_secret_reaches_the_owner(self, deck):
        got = await _read(BOB)
        for secret in (TEAM_KEY, ENDPOINT_KEY, OWN_KEY):
            assert secret not in got["env"]
            assert secret not in got["config_yaml"]

    async def test_env_lines_masked_with_last_four(self, deck):
        lines = (await _read(BOB))["env"].splitlines()
        assert lines == [
            "# written by Flight Deck",
            "ANTHROPIC_API_KEY=····abcd",
            'BRAVE_API_KEY="····wxyz"',              # quotes kept, value masked
            "OLLAMA_BASE_URL=http://127.0.0.1:11434",  # not a secret
            "CLAW_VFS_USER=user-bob",
            "CUSTOM_HEADER=····abcd",                # holds the team key
            "# old one: ····abcd",                   # the team key in a comment
            "SHORT_TOKEN=····",                      # too short to hint at
            "EMPTY_API_KEY=",                        # nothing to hide
        ]

    async def test_config_keeps_what_the_kiosk_dialog_reads(self, deck):
        cfg = yaml.safe_load((await _read(BOB))["config_yaml"])
        assert cfg["model"]["provider"] == "openai"
        assert cfg["model"]["model"] == "gpt-4.1"
        assert cfg["model"]["base_url"] == "https://gw.example/v1"
        assert cfg["model"]["vision"] is True
        assert cfg["model"]["api_key"] == "····XYZW"

    async def test_flight_decks_own_env_key_is_masked_by_value(self, deck, monkeypatch):
        fd_key = "sk-fd-environment-key-9876"
        monkeypatch.setenv("OPENAI_API_KEY", fd_key)
        (server.DATA_DIR / "bob-agent" / ".env").write_text(f"SOME_URL=https://x/?k={fd_key}\n")
        assert (await _read(BOB))["env"] == "SOME_URL=····9876\n"

    async def test_a_short_team_key_is_masked_only_as_a_whole_value(self, deck):
        await deck.set_system_setting("fd:endpoint-keys", json.dumps({"ollama@box:11434": "x"}))
        (server.DATA_DIR / "bob-agent" / ".env").write_text("PLACEHOLDER=x\nMODE=xylophone\n")
        assert (await _read(BOB))["env"] == "PLACEHOLDER=····\nMODE=xylophone\n"


# ── who reads the files as they are ─────────────────────────────────────────

class TestAdminsReadRaw:
    async def test_admin_reads_own_agent_raw(self, deck):
        assert await _read(ADMIN, "admin-agent") == {"config_yaml": BOB_CONFIG, "env": BOB_ENV}

    async def test_admin_acting_for_the_owner_reads_raw(self, deck):
        assert await _read(ADMIN, "bob-agent", act_as=BOB) == {"config_yaml": BOB_CONFIG, "env": BOB_ENV}

    def test_a_deck_without_accounts_reads_raw(self, monkeypatch):
        monkeypatch.setenv("FD_AUTH_ENABLED", "false")
        req = SimpleNamespace(state=SimpleNamespace())
        assert server._sees_agent_secrets(req, None)
        assert server._sees_agent_secrets(req, {"id": BOB, "role": "user"})

    def test_with_accounts_only_admins_do(self, monkeypatch):
        monkeypatch.setenv("FD_AUTH_ENABLED", "true")
        plain = SimpleNamespace(state=SimpleNamespace())
        assert not server._sees_agent_secrets(plain, {"id": BOB, "role": "user"})
        assert not server._sees_agent_secrets(plain, None)
        assert server._sees_agent_secrets(plain, {"id": ADMIN, "role": "admin"})
        acting = SimpleNamespace(state=SimpleNamespace(acting_admin={"id": ADMIN}))
        assert server._sees_agent_secrets(acting, {"id": BOB, "role": "user"})


# ── saving what was read ────────────────────────────────────────────────────

class TestSaveKeepsTheRealValues:
    async def test_saving_the_masked_read_unchanged_changes_nothing(self, deck):
        got = await _read(BOB)
        r = await _save(BOB, got)
        assert r.status_code == 200, r.text
        assert _files("bob-agent") == (BOB_CONFIG, BOB_ENV)

    async def test_an_edit_elsewhere_keeps_every_masked_value(self, deck):
        got = await _read(BOB)
        env = got["env"].replace("OLLAMA_BASE_URL=http://127.0.0.1:11434", "OLLAMA_BASE_URL=http://box:11434")
        env = env.replace("ANTHROPIC_API_KEY=····abcd", "export ANTHROPIC_API_KEY = '····abcd'")
        config = got["config_yaml"].replace("gpt-4.1", "gpt-4.1-mini")
        r = await _save(BOB, {"config_yaml": config, "env": env + "NEW_VAR=1\n"})
        assert r.status_code == 200, r.text
        cfg_text, env_text = _files("bob-agent")
        assert f"api_key: {ENDPOINT_KEY}" in cfg_text and "gpt-4.1-mini" in cfg_text
        assert "····" not in cfg_text and "····" not in env_text
        assert f"export ANTHROPIC_API_KEY = '{TEAM_KEY}'" in env_text
        assert f'BRAVE_API_KEY="{OWN_KEY}"' in env_text
        assert f"CUSTOM_HEADER=Bearer {TEAM_KEY}" in env_text
        assert f"# old one: {TEAM_KEY}" in env_text
        assert "SHORT_TOKEN=abc" in env_text
        assert "OLLAMA_BASE_URL=http://box:11434" in env_text
        assert env_text.endswith("NEW_VAR=1\n")

    async def test_a_new_value_replaces_the_masked_one(self, deck):
        got = await _read(BOB)
        env = got["env"].replace('BRAVE_API_KEY="····wxyz"', "BRAVE_API_KEY=brave-new-key")
        assert (await _save(BOB, {"env": env})).status_code == 200
        env_text = _files("bob-agent")[1]
        assert "BRAVE_API_KEY=brave-new-key" in env_text and OWN_KEY not in env_text
        assert f"ANTHROPIC_API_KEY={TEAM_KEY}" in env_text

    async def test_a_placeholder_with_nothing_behind_it_is_refused(self, deck):
        got = await _read(BOB)
        env = got["env"].replace("ANTHROPIC_API_KEY=····abcd", "OPENAI_API_KEY=····abcd")
        r = await _save(BOB, {"config_yaml": "model: changed\n", "env": env})
        assert r.status_code == 400
        assert "OPENAI_API_KEY" in r.json()["detail"]
        assert _files("bob-agent") == (BOB_CONFIG, BOB_ENV)  # neither file written

    async def test_a_mask_for_another_value_is_refused(self, deck):
        got = await _read(BOB)
        env = got["env"].replace("ANTHROPIC_API_KEY=····abcd", "ANTHROPIC_API_KEY=····zzzz")
        assert (await _save(BOB, {"env": env})).status_code == 400
        assert _files("bob-agent")[1] == BOB_ENV

    async def test_admin_saves_as_before(self, deck):
        r = await _save(ADMIN, {"env": "KEY=2\n"}, slug="admin-agent")
        assert r.status_code == 200
        assert _files("admin-agent")[1] == "KEY=2\n"


# ── the helpers on their own ────────────────────────────────────────────────

def test_mask_then_unmask_is_the_original_text():
    keys = {TEAM_KEY}
    text = f"A_TOKEN=tok-123456789012\r\nB=1\r\n# {TEAM_KEY}\r\nexport C_SECRET='s3cr3t-value-long'\n"
    masked = server._mask_agent_env(text, keys)
    assert "tok-123456789012" not in masked and TEAM_KEY not in masked and "s3cr3t" not in masked
    assert masked.count("\r\n") == 3  # line endings as they were
    assert server._unmask_agent_env(masked, text, keys) == text


def test_duplicate_names_restore_their_own_values():
    text = "API_KEY=first-value-aaaa1111\nAPI_KEY=second-value-bbbb2222\n"
    masked = server._mask_agent_env(text, set())
    assert masked == "API_KEY=····1111\nAPI_KEY=····2222\n"
    swapped = "API_KEY=····2222\nAPI_KEY=····1111\n"
    assert server._unmask_agent_env(swapped, text, set()) == (
        "API_KEY=second-value-bbbb2222\nAPI_KEY=first-value-aaaa1111\n")


def test_secret_names():
    secret = ("OPENAI_API_KEY", "APIKEY", "api_key", "SLACK_BOT_TOKEN", "CLIENT_SECRET",
              "DB_PASSWORD", "BOTPORT_KEY", "TOKEN")
    plain = ("OLLAMA_BASE_URL", "CLAW_VFS_USER", "KEYBOARD", "MONKEY", "TOKENS_MAX", "SECRETARY")
    assert all(server._SECRET_ENV_NAME.search(n) for n in secret)
    assert not any(server._SECRET_ENV_NAME.search(n) for n in plain)
