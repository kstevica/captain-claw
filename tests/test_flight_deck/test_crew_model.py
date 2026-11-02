"""Crew: a worker on one of the person's agents' models, a tier of the deck's, or one typed in for
the task. The key stays in Flight Deck: all a worker ever holds is a key for one of its turns."""

# ruff: noqa: F811
from __future__ import annotations

import json
import shutil
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, HTTPException

from captain_claw.flight_deck import (
    crew_access,
    crew_harness,
    crew_logins,
    crew_model,
    crew_routes,
    crew_runner,
    crew_store,
    server,
    usage_quota,
)
from captain_claw.flight_deck.crew_access import Grant
from captain_claw.flight_deck.crew_harness import Route, RunSpec

from .test_crew_runner import ADMIN_GRANT, _dispatch, _run, _seen, _settle, deck  # noqa: F401

NOT_ADMIN = Grant(True, "approved", ("claude", "codex", "opencode", "pi"), 4)
TEAM_KEY = "sk-team-key-that-nobody-may-read"
THEIR_KEY = "sk-their-own-key-for-their-server"
ANTHROPIC_TIER = {"provider": "anthropic", "model": "claude-test", "base_url": "", "api_key": "@system", "input_ctx": 200000, "output_ctx": 8000}
GEMINI_TIER = {"provider": "gemini", "model": "gemini-test", "base_url": "", "api_key": "@system", "input_ctx": 100000}
TYPED = {"kind": "typed", "url": "http://10.0.0.9:8000/v1", "model": "local-model", "context": "32000", "key": THEIR_KEY}


@pytest.fixture(autouse=True)
def _their_files_and_approval(deck, monkeypatch):
    monkeypatch.setenv("FD_DATA_DIR", str(deck["tmp"] / "fd-data"))
    monkeypatch.setattr(crew_model, "_TRY_AGAIN_AFTER", 0.0)
    crew_model._CALLS.clear()
    monkeypatch.setattr(crew_access, "grant_of", lambda row: ADMIN_GRANT if row and row.get("role") == "admin" else NOT_ADMIN)


async def _as_them(deck, person: dict, brief: str, **over) -> dict:
    return await crew_runner.dispatch(person, NOT_ADMIN, {"harness": "claude", "brief": brief, "project": "p", **over})


async def _a_person(deck, name: str = "u") -> dict:
    return await deck["db"].create_user(f"{name}@example.com", "x", role="user")


async def _team_tiers(deck, **tiers) -> None:
    db = deck["db"]
    await db.set_system_setting("fd:shared-tier-sets", json.dumps({"sets": [{"id": "team", "name": "Team", "tiers": tiers}], "defaultSetId": "team"}))
    await db.set_system_setting("fd:provider-keys", json.dumps({"anthropic": TEAM_KEY, "gemini": TEAM_KEY}))


@pytest.fixture
def agents(deck, monkeypatch):
    home = deck["tmp"] / "fd-data"
    home.mkdir(exist_ok=True)
    registry = home / ".processes.json"
    monkeypatch.setattr(server, "PROCESS_REGISTRY_FILE", registry)

    def make(slug: str, owner: str, *, env: str = "", config: dict | None = None, **entry) -> None:
        folder = home / slug
        folder.mkdir()
        (folder / "config.yaml").write_text(json.dumps(config or {"model": {"provider": "anthropic", "model": "claude-agent"}, "context": {"max_tokens": 90000}}))
        (folder / ".env").write_text(env)
        current = json.loads(registry.read_text()) if registry.exists() else {}
        registry.write_text(json.dumps({**current, slug: {"name": slug.title(), "owner": owner, "slug": slug, **entry}}))

    return make


def _env_of(task: dict) -> dict[str, str]:
    text = (Path(task["worktree"]) / "env.txt").read_text()
    return dict(line.split("=", 1) for line in text.splitlines() if "=" in line)


def _stand_in(deck, monkeypatch, harness: str) -> None:
    fake = deck["tmp"] / "bin" / harness
    if not fake.exists():
        shutil.copy(deck["tmp"] / "bin" / "claude", fake)
    monkeypatch.setenv(f"FD_CREW_{harness.upper()}_BIN", str(fake))


# ── Who it's for, and what's offered ─────────────────────────────────────


async def test_what_can_be_chosen_says_which_clis_can_use_it_and_never_a_key(deck, agents):
    person = await _a_person(deck)
    await _team_tiers(deck, balanced=ANTHROPIC_TIER, vision=GEMINI_TIER, micro={"provider": "ollama", "model": "tiny"})
    agents("mine", person["id"], env=f"ANTHROPIC_API_KEY={TEAM_KEY}\n")
    agents("theirs", "someone-else")
    got = await crew_model.choices(deck["db"], person, NOT_ADMIN)
    assert TEAM_KEY not in json.dumps(got)
    tiers = {t["id"]: t for t in got["tiers"]}
    assert set(tiers) == {"balanced", "vision"}, "a tier for tiny models isn't one a coding CLI can work on"
    assert set(tiers["balanced"]["works_with"]) == {"claude", "opencode", "crush", "pi"} and tiers["balanced"]["context"] == 200000
    assert set(tiers["vision"]["works_with"]) == {"opencode", "crush", "pi"}
    assert [a["id"] for a in got["agents"]] == ["mine"], "only their own: a shared one is on its owner's account"
    assert got["agents"][0]["model"] == "claude-agent" and got["agents"][0]["context"] == 90000


async def test_a_cli_only_gets_a_model_it_can_talk_to(deck):
    await _team_tiers(deck, vision=GEMINI_TIER)
    with pytest.raises(HTTPException) as refused:
        await _dispatch(deck, "SCENARIO:work", model_source={"kind": "tier", "tier": "vision"})
    assert refused.value.status_code == 400 and "Claude Code can't talk to" in refused.value.detail and "OpenCode, Crush or Pi can" in refused.value.detail


async def test_someone_elses_agent_cant_be_named(deck, agents):
    person = await _a_person(deck)
    agents("theirs", "someone-else", env=f"ANTHROPIC_API_KEY={TEAM_KEY}\n")
    with pytest.raises(HTTPException) as refused:
        await _as_them(deck, person, "SCENARIO:work", model_source={"kind": "agent", "agent": "theirs"})
    assert "no agent called theirs" in refused.value.detail


async def test_an_agents_team_key_going_where_the_team_doesnt_send_it_is_refused(deck, agents):
    person = await _a_person(deck)
    await _team_tiers(deck, balanced=ANTHROPIC_TIER)
    agents("away", person["id"], env=f"ANTHROPIC_API_KEY={TEAM_KEY}\n",
           config={"model": {"provider": "anthropic", "model": "m", "base_url": "https://their.example.com"}})
    with pytest.raises(HTTPException) as refused:
        await crew_model.resolve(deck["db"], person, NOT_ADMIN, {"kind": "agent", "agent": "away"})
    assert "the team" in refused.value.detail


async def test_naming_no_key_of_its_own_doesnt_send_the_teams_from_its_env_somewhere_else(deck, agents):
    """Its config points at their own server and says "@system", with the team's key in its .env."""
    person = await _a_person(deck)
    await _team_tiers(deck, balanced=ANTHROPIC_TIER)
    agents("away", person["id"], env=f"ANTHROPIC_API_KEY={TEAM_KEY}\n",
           config={"model": {"provider": "anthropic", "model": "m", "api_key": "@system", "base_url": "https://their.example.com"}})
    with pytest.raises(HTTPException) as refused:
        await crew_model.resolve(deck["db"], person, NOT_ADMIN, {"kind": "agent", "agent": "away"})
    assert "the team" in refused.value.detail
    # Nor by its environment naming the place, nor by a key in its config's list of them.
    agents("named", person["id"], env=f"ANTHROPIC_API_KEY={TEAM_KEY}\nANTHROPIC_BASE_URL=https://their.example.com\n")
    agents("listed", person["id"], config={"model": {"provider": "anthropic", "model": "m", "base_url": "https://their.example.com"}, "provider_keys": {"anthropic": TEAM_KEY}})
    for slug in ("named", "listed"):
        with pytest.raises(HTTPException):
            await crew_model.resolve(deck["db"], person, NOT_ADMIN, {"kind": "agent", "agent": slug})
    choices = await crew_model.choices(deck["db"], person, NOT_ADMIN)
    assert all(a["works_with"] == [] and a["why_not"] for a in choices["agents"]), "and the page says so, not a dead end at the start"


async def test_the_teams_key_on_the_teams_endpoint_is_fine_and_counts_as_the_teams(deck, agents):
    person = await _a_person(deck)
    await _team_tiers(deck, balanced=ANTHROPIC_TIER)
    agents("mine", person["id"], env=f"ANTHROPIC_API_KEY={TEAM_KEY}\n", key_source="byok")
    got = await crew_model.resolve(deck["db"], person, NOT_ADMIN, {"kind": "agent", "agent": "mine"})
    assert got.key == TEAM_KEY and not got.own_keys, "whatever the registry remembers of it, this one's the team's to count"


async def test_a_providers_other_names_and_those_flight_deck_has_no_entry_for_find_their_key(deck, agents):
    person = await _a_person(deck)
    agents("alias", person["id"], env="ANTHROPIC_API_KEY=sk-under-its-name\n", config={"model": {"provider": "claude", "model": "m"}})
    assert (await crew_model.resolve(deck["db"], person, NOT_ADMIN, {"kind": "agent", "agent": "alias"})).key == "sk-under-its-name"
    agents("deep", person["id"], env="DEEPSEEK_API_KEY=sk-deep\n", config={"model": {"provider": "deepseek", "model": "deepseek-chat"}})
    got = await crew_model.resolve(deck["db"], person, NOT_ADMIN, {"kind": "agent", "agent": "deep"})
    assert got.key == "sk-deep" and got.url == "https://api.deepseek.com/v1"


async def test_a_tiers_team_key_doesnt_go_to_an_address_someone_named_in_their_own_set(deck):
    person = await _a_person(deck)
    await _team_tiers(deck, balanced=ANTHROPIC_TIER)
    mine = {**ANTHROPIC_TIER, "base_url": "https://their.example.com"}
    await deck["db"].set_settings(person["id"], {"fd:forge-tiers": json.dumps({"sets": [{"id": "s", "tiers": {"balanced": mine}}], "activeSetId": "s"})})
    got = await crew_model.resolve(deck["db"], person, NOT_ADMIN, {"kind": "tier", "tier": "balanced"})
    assert got.url == "https://their.example.com/v1" and got.key == ""


async def test_the_decks_own_key_isnt_lent_to_someone_who_isnt_an_admin(deck, monkeypatch):
    person = await _a_person(deck)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-the-decks-own")
    tier = {**ANTHROPIC_TIER, "api_key": ""}
    await deck["db"].set_settings(person["id"], {"fd:forge-tiers": json.dumps({"sets": [{"id": "s", "tiers": {"balanced": tier}}], "activeSetId": "s"})})
    source = {"kind": "tier", "tier": "balanced"}
    assert (await crew_model.resolve(deck["db"], person, NOT_ADMIN, source)).key == ""
    assert (await crew_model.resolve(deck["db"], person, ADMIN_GRANT, source)).key == "sk-the-decks-own"


def test_where_each_provider_is():
    def where(provider: str, base: str = "") -> tuple[str, tuple[str, ...]]:
        got = crew_model._upstream(provider, "m", base, "k")
        return got.url, got.apis

    assert where("anthropic") == ("https://api.anthropic.com/v1", ("anthropic",))
    assert where("claude") == ("https://api.anthropic.com/v1", ("anthropic",)), "by any name an agent takes"
    assert where("anthropic", "https://gw.example.com")[0] == "https://gw.example.com/v1"
    assert where("openai") == ("https://api.openai.com/v1", ("openai", "responses"))
    assert where("openai", "http://lm:1234/v1")[0] == "http://lm:1234/v1", "a server of someone's own gets the benefit of the doubt"
    assert len(where("openai", "http://lm:1234/v1")[1]) == 3
    assert where("gemini")[1] == ("openai",)
    assert where("ollama", "http://box:11434")[0] == "http://box:11434/v1" and len(where("ollama")[1]) == 3
    assert crew_model._upstream("anthropic", "m", "", "k").native_key and not crew_model._upstream("anthropic", "m", "https://gw.example.com", "k").native_key
    assert crew_model._upstream("openrouter", "openrouter/anthropic/claude", "", "k").model == "anthropic/claude"
    for nowhere in ("claude-cli", "browser", "litert", ""):
        with pytest.raises(HTTPException):
            crew_model._upstream(nowhere, "m", "", "k")
    with pytest.raises(HTTPException, match="ChatGPT"):
        crew_model._upstream("openai", "gpt-5-codex", "", "")
    assert crew_model._upstream("openai", "gpt-5-codex", "", "k").key == "k"


# ── What a worker holds ──────────────────────────────────────────────────


async def test_a_workers_environment_has_no_real_key_only_one_for_its_turn(deck, monkeypatch):
    person = await _a_person(deck)
    await _team_tiers(deck, balanced=ANTHROPIC_TIER)
    monkeypatch.setenv("FD_PORT", "26999")
    done = await _settle(await _as_them(deck, person, "SCENARIO:envdump", model_source={"kind": "tier", "tier": "balanced"}))
    env = _env_of(done)
    assert TEAM_KEY not in (Path(done["worktree"]) / "env.txt").read_text()
    assert env["ANTHROPIC_API_KEY"] == env["CREW_MODEL_TOKEN"] == done["model_token"] and env["ANTHROPIC_API_KEY"].startswith("crew-")
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:26999/fd/crew/llm"
    assert env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "200000" and env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "8000"
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in env
    assert "--model claude-test" in _seen(done)
    assert done["model"] == "claude-test" and done["model_source"] == {"kind": "tier", "tier": "balanced", "label": "Balanced tier"}


async def test_no_login_is_needed_and_none_is_given_on_a_chosen_model(deck):
    person = await _a_person(deck)
    await crew_logins.save_login(deck["db"], person["id"], "claude", "sk-ant-oat01-their-login")
    await crew_logins.save_login(deck["db"], person["id"], "openrouter", "sk-or-their-other-key")
    done = await _settle(await _as_them(deck, person, "SCENARIO:envdump", model_source=TYPED))
    text = (Path(done["worktree"]) / "env.txt").read_text()
    assert "their-login" not in text and "their-other-key" not in text and THEIR_KEY not in text
    other = await _a_person(deck, "no-login")
    assert (await _settle(await _as_them(deck, other, "SCENARIO:work", model_source=TYPED)))["status"] == "ready"


async def test_a_turn_gets_a_key_of_its_own(deck):
    done = await _settle(await _dispatch(deck, "SCENARIO:work", model_source=TYPED))
    first = done["model_token"]
    again = await _settle(await crew_runner.steer(done["user_id"], done["id"], "go on"))
    assert again["model_token"] and again["model_token"] != first


def test_how_each_cli_is_pointed_at_it(tmp_path):
    route = Route(url="http://127.0.0.1:9/fd/crew/llm/v1", model="big:9b", api="responses", context=32000)
    argv = crew_harness._codex_argv(RunSpec(binary="codex", prompt="p", cwd="/w", model="big:9b", route=route))
    said = " ".join(argv)
    assert 'model_provider="crew"' in said and 'model_providers.crew.base_url="http://127.0.0.1:9/fd/crew/llm/v1"' in said
    assert 'model_providers.crew.env_key="CREW_MODEL_TOKEN"' in said and "model_context_window=32000" in said and "-m big:9b" in said
    assert crew_harness.route_model("opencode", route) == "crew/big:9b" and crew_harness.route_model("claude", route) == "big:9b"

    plain = Route(url=route.url, model="big:9b", api="openai", context=32000)
    config = crew_logins.opencode_route_config(plain)
    provider = config["provider"]["crew"]
    assert provider["npm"] == "@ai-sdk/openai-compatible" and provider["options"] == {"baseURL": route.url, "apiKey": "{env:CREW_MODEL_TOKEN}"}
    assert provider["models"]["big:9b"]["limit"]["context"] == 32000 and config["small_model"] == "crew/big:9b"
    native = Route(url=route.url, model="claude-x", api="anthropic")
    assert crew_logins.opencode_route_config(native)["provider"]["crew"]["npm"] == "@ai-sdk/anthropic"

    models = tmp_path / ".pi" / "agent" / "models.json"
    models.parent.mkdir(parents=True)
    models.write_text(json.dumps({"mine": 1, "providers": {"theirs": {"baseUrl": "http://x"}}}))
    crew_logins._sync_pi_route(tmp_path, plain)
    crew_logins._sync_pi_route(tmp_path, native)
    crew_logins._sync_pi_route(tmp_path, Route(url=route.url, model="other", api="openai"))
    saved = json.loads(models.read_text())
    assert saved["mine"] == 1 and saved["providers"]["theirs"] == {"baseUrl": "http://x"}, "what else is in the file stays"
    assert saved["providers"]["crew"]["api"] == "openai-completions" and saved["providers"]["crew"]["baseUrl"] == route.url
    assert [m["id"] for m in saved["providers"]["crew"]["models"]] == ["big:9b", "other"], "the tasks of one home share it"
    assert saved["providers"]["crew-anthropic"]["api"] == "anthropic-messages" and saved["providers"]["crew-anthropic"]["baseUrl"] == "http://127.0.0.1:9/fd/crew/llm"
    assert saved["providers"]["crew"]["apiKey"] == "$CREW_MODEL_TOKEN"
    assert crew_harness.route_model("pi", native) == "crew-anthropic/claude-x" and crew_harness.route_model("pi", plain) == "crew/big:9b"


@pytest.mark.parametrize("harness", ["codex", "opencode", "pi"])
async def test_the_others_run_on_it_through_their_own_settings(deck, monkeypatch, harness):
    _stand_in(deck, monkeypatch, harness)
    task = await _dispatch(deck, "SCENARIO:envdump", harness=harness, model_source=TYPED)
    await _settle(task, 10)
    fresh = crew_store.get_store().get(task["user_id"], task["id"])
    env = _env_of(fresh)
    assert env["CREW_MODEL_TOKEN"] == fresh["model_token"] and THEIR_KEY not in (Path(fresh["worktree"]) / "env.txt").read_text()
    assert "127.0.0.1" in env["NO_PROXY"]
    if harness == "opencode":
        assert json.loads(env["OPENCODE_CONFIG_CONTENT"])["provider"]["crew"]["models"]["local-model"]["limit"]["context"] == 32000
        assert "--model crew/local-model" in _seen(fresh)
    elif harness == "pi":
        assert "--model crew/local-model" in _seen(fresh)
    else:
        assert 'model_provider="crew"' in _seen(fresh)


# ── A key typed in ───────────────────────────────────────────────────────


async def test_a_typed_in_key_is_sealed_never_shown_and_gone_when_the_task_closes(deck):
    done = await _settle(await _dispatch(deck, "SCENARIO:work", model_source=TYPED))
    raw = crew_store.get_store()._conn.execute("SELECT model_source FROM tasks WHERE id = ?", (done["id"],)).fetchone()[0]
    assert THEIR_KEY not in raw and json.loads(raw)["key"]
    shown = crew_runner.view(done, full=True)
    assert shown["model_source"] == {"kind": "typed", "label": "Typed in for this task"} and THEIR_KEY not in json.dumps(shown)
    assert (await crew_model.resolve(deck["db"], deck["admin"], ADMIN_GRANT, done["model_source"])).key == THEIR_KEY
    await crew_runner.close(done["user_id"], done["id"], discard=True)
    after = crew_store.get_store().get(done["user_id"], done["id"])
    assert "key" not in after["model_source"] and not after["model_token"] and after["model_source"]["model"] == "local-model"


@pytest.mark.parametrize("bad, said", [
    ({"url": "ftp://x"}, "http://host"), ({"url": "http://a:b@x/v1"}, "http://host"), ({"model": ""}, "model's name"),
    ({"model": "a b"}, "model's name"), ({"context": "lots"}, "number of tokens"), ({"context": "5"}, "number of tokens"), ({"key": "k" * 70000}, "too long"),
])
async def test_what_can_be_typed_in(deck, bad, said):
    with pytest.raises(HTTPException) as refused:
        await _dispatch(deck, "SCENARIO:work", model_source={**TYPED, **bad})
    assert said in refused.value.detail


async def test_a_blank_key_is_the_saved_one_but_only_for_the_server_it_was_saved_for(deck):
    await crew_logins.save_own_model(deck["db"], deck["admin"]["id"], TYPED["url"], THEIR_KEY, "local-model", 0)
    same = await crew_model._typed(deck["db"], deck["admin"]["id"], {**TYPED, "key": ""})
    assert (await crew_model.resolve(deck["db"], deck["admin"], ADMIN_GRANT, same)).key == THEIR_KEY
    elsewhere = await crew_model._typed(deck["db"], deck["admin"]["id"], {**TYPED, "url": "https://else.example.com/v1", "key": ""})
    assert "key" not in elsewhere


# ── A retry ──────────────────────────────────────────────────────────────


async def test_asked_again_its_the_same_task_on_another_model_its_another(deck):
    first = await _dispatch(deck, "SCENARIO:slow", model_source=TYPED)
    assert (await _dispatch(deck, "SCENARIO:slow", model_source=TYPED))["id"] == first["id"], "sealed twice, and still the same choice"
    assert (await _dispatch(deck, "SCENARIO:slow", model_source={**TYPED, "model": "another"}))["id"] != first["id"]
    assert (await _dispatch(deck, "SCENARIO:slow"))["id"] != first["id"]


# ── The calls ────────────────────────────────────────────────────────────


LOOPBACK = ("127.0.0.1", 40000)
_REAL_CLIENT = httpx.AsyncClient


def _worker(client=LOOPBACK, token: str = "") -> httpx.AsyncClient:
    app = FastAPI()
    app.include_router(crew_routes.router)
    return _REAL_CLIENT(transport=httpx.ASGITransport(app=app, client=client), base_url="http://127.0.0.1",
                             headers={"authorization": f"Bearer {token}"} if token else {})


class _Sent(httpx.AsyncByteStream):
    """As a network hands a reply over: not yet read."""

    def __init__(self, body: bytes) -> None:
        self.body = body

    async def __aiter__(self):
        yield self.body


def _reply(status: int, body: bytes, media: str = "application/json") -> httpx.Response:
    return httpx.Response(status, headers={"content-type": media}, stream=_Sent(body))


@pytest.fixture
def upstream(monkeypatch):
    """The model's server: what it was asked, and what it says back."""
    calls: list[httpx.Request] = []
    replies: list[httpx.Response] = []

    def answer(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return replies.pop(0) if replies else _reply(200, json.dumps({"id": "r", "usage": {"prompt_tokens": 100, "completion_tokens": 20}}).encode())

    def client(**kwargs) -> httpx.AsyncClient:
        return _REAL_CLIENT(**{**kwargs, "transport": httpx.MockTransport(answer)})

    monkeypatch.setattr(crew_model.httpx, "AsyncClient", client)
    return calls, replies


def _sent(call: httpx.Request) -> dict:
    return json.loads(call.content)


async def _running_on(deck, source: dict, *, user=None) -> dict:
    if user:
        return await _as_them(deck, user, "SCENARIO:slow", model_source=source)
    return await _dispatch(deck, "SCENARIO:slow", model_source=source)


async def _spent_today(deck, uid: str) -> dict:
    await crew_model.asyncio.sleep(0.05)  # it's counted once the reply has gone
    return await deck["db"].get_daily_usage(uid, usage_quota.today())


async def test_a_call_goes_out_on_the_real_key_and_the_chosen_model(deck, upstream):
    calls, _ = upstream
    task = await _running_on(deck, TYPED)
    async with _worker(token=task["model_token"]) as worker:
        got = await worker.post("/fd/crew/llm/v1/chat/completions", json={"model": "whatever-it-wanted", "messages": []},
                                headers={"cookie": "fd=secret", "x-other": "dropped"})
    assert got.status_code == 200 and got.json()["id"] == "r"
    call = calls[0]
    assert str(call.url) == "http://10.0.0.9:8000/v1/chat/completions"
    assert call.headers["authorization"] == f"Bearer {THEIR_KEY}" and _sent(call)["model"] == "local-model"
    assert "cookie" not in call.headers and "x-other" not in call.headers and task["model_token"] not in str(call.headers)


async def test_each_kind_of_client_gets_to_its_path_with_what_its_server_wants(deck, upstream):
    calls, _ = upstream
    await _team_tiers(deck, balanced=ANTHROPIC_TIER)
    task = await _running_on(deck, {"kind": "tier", "tier": "balanced"})
    async with _worker() as worker:
        got = await worker.post("/fd/crew/llm/v1/messages?beta=true", json={"model": "claude-haiku", "max_tokens": 1, "messages": []},
                                headers={"x-api-key": task["model_token"], "anthropic-version": "2023-06-01", "anthropic-beta": "b1"})
    assert got.status_code == 200
    call = calls[0]
    assert str(call.url) == "https://api.anthropic.com/v1/messages?beta=true"
    assert call.headers["x-api-key"] == TEAM_KEY and "authorization" not in call.headers, "Anthropic's own takes it there, and only there"
    assert call.headers["anthropic-version"] == "2023-06-01" and call.headers["anthropic-beta"] == "b1"
    assert _sent(call)["model"] == "claude-test", "a CLI's small model for background work runs on what was chosen too"


async def test_the_address_is_read_with_or_without_v1(deck, upstream):
    calls, _ = upstream
    task = await _running_on(deck, TYPED)
    async with _worker(token=task["model_token"]) as worker:
        for path in ("chat/completions", "v1/chat/completions", "v1/v1/chat/completions"):
            assert (await worker.post(f"/fd/crew/llm/{path}", json={})).status_code == 200
    assert {str(c.url) for c in calls} == {"http://10.0.0.9:8000/v1/chat/completions"}


@pytest.mark.parametrize("path", ["files", "fine_tuning/jobs", "../../secrets", "v1/embeddings", "api/oauth/token"])
async def test_only_a_models_calls_go_through(deck, upstream, path):
    calls, _ = upstream
    task = await _running_on(deck, TYPED)
    async with _worker(token=task["model_token"]) as worker:
        assert (await worker.post(f"/fd/crew/llm/{path}", json={})).status_code == 404
    assert not calls


async def test_a_key_is_good_only_while_its_turn_runs_and_only_from_here(deck, upstream):
    calls, _ = upstream
    task = await _running_on(deck, TYPED)
    token = task["model_token"]
    body = {"model": "x", "messages": []}
    async with _worker() as nobody:
        assert (await nobody.post("/fd/crew/llm/v1/chat/completions", json=body)).status_code == 401
    async with _worker(token="crew-not-a-real-one") as wrong:
        assert (await wrong.post("/fd/crew/llm/v1/chat/completions", json=body)).status_code == 401
    async with _worker(("192.168.1.9", 40000), token) as elsewhere:
        assert (await elsewhere.post("/fd/crew/llm/v1/chat/completions", json=body)).status_code == 403
    async with _worker(token=token) as worker:
        got = await worker.post("/fd/crew/llm/v1/chat/completions", json=body, headers={"x-forwarded-for": "8.8.8.8"})
        assert got.status_code == 403, "through a reverse proxy, the address says nothing"
        assert "error" in got.json() and got.json()["error"]["message"]
    assert not calls
    await crew_runner.stop(task["user_id"], task["id"])
    async with _worker(token=token) as worker:
        assert (await worker.post("/fd/crew/llm/v1/chat/completions", json=body)).status_code == 401, "its turn is over"
    assert not calls


async def test_the_key_of_someone_who_lost_their_approval_stops_working(deck, upstream, monkeypatch):
    calls, _ = upstream
    task = await _running_on(deck, TYPED)
    monkeypatch.setattr(crew_access, "grant_of", lambda row: Grant(False, "", (), 0))
    async with _worker(token=task["model_token"]) as worker:
        assert (await worker.post("/fd/crew/llm/v1/chat/completions", json={})).status_code == 403
    assert not calls


async def test_a_servers_error_comes_back_without_the_key_in_it(deck, upstream):
    _, replies = upstream
    replies.append(_reply(401, json.dumps({"error": {"message": f"Bad key: {THEIR_KEY}"}}).encode()))
    task = await _running_on(deck, TYPED)
    async with _worker(token=task["model_token"]) as worker:
        got = await worker.post("/fd/crew/llm/v1/chat/completions", json={})
    assert got.status_code == 401 and THEIR_KEY not in got.text and "Bad key" in got.text


async def test_a_server_saying_wait_is_passed_on_so_a_cli_can(deck, upstream):
    _, replies = upstream
    busy = _reply(429, b'{"error":{"message":"slow down"}}')
    busy.headers["retry-after"] = "60"
    replies.append(busy)
    task = await _running_on(deck, TYPED)
    async with _worker(token=task["model_token"]) as worker:
        got = await worker.post("/fd/crew/llm/v1/chat/completions", json={})
    assert got.status_code == 429 and got.headers["retry-after"] == "60", "that long isn't a call to hold open here"


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504, 529])
async def test_a_server_failing_before_it_says_anything_is_asked_again(deck, upstream, status):
    calls, replies = upstream
    replies.append(_reply(status, b'{"error":{"message":"busy"}}'))
    task = await _running_on(deck, TYPED)
    async with _worker(token=task["model_token"]) as worker:
        got = await worker.post("/fd/crew/llm/v1/chat/completions", json={})
    assert got.status_code == 200 and len(calls) == 2, "and the worker never hears of it"


async def test_it_stops_asking_and_says_what_the_server_said(deck, upstream):
    calls, replies = upstream
    replies.extend(_reply(503, b'{"error":{"message":"down for now"}}') for _ in range(5))
    task = await _running_on(deck, TYPED)
    async with _worker(token=task["model_token"]) as worker:
        got = await worker.post("/fd/crew/llm/v1/chat/completions", json={})
    assert got.status_code == 503 and "down for now" in got.text and len(calls) == crew_model._CALL_TRIES


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
async def test_asking_again_wont_change_a_refusal(deck, upstream, status):
    calls, replies = upstream
    replies.append(_reply(status, b'{"error":{"message":"no"}}'))
    task = await _running_on(deck, TYPED)
    async with _worker(token=task["model_token"]) as worker:
        assert (await worker.post("/fd/crew/llm/v1/chat/completions", json={})).status_code == status
    assert len(calls) == 1


async def test_a_dropped_connection_is_asked_again(deck, monkeypatch):
    seen = []

    def flaky(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if len(seen) == 1:
            raise httpx.ReadError("the connection went", request=request)
        return _reply(200, b'{"id":"r"}')

    monkeypatch.setattr(crew_model.httpx, "AsyncClient", lambda **kw: _REAL_CLIENT(**{**kw, "transport": httpx.MockTransport(flaky)}))
    task = await _running_on(deck, TYPED)
    async with _worker(token=task["model_token"]) as worker:
        assert (await worker.post("/fd/crew/llm/v1/chat/completions", json={})).status_code == 200
    assert len(seen) == 2


async def test_a_server_that_says_nothing_is_hung_up_on_and_not_asked_again(deck, monkeypatch):
    seen = []

    def silent(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        raise httpx.ReadTimeout("nothing", request=request)

    monkeypatch.setattr(crew_model.httpx, "AsyncClient", lambda **kw: _REAL_CLIENT(**{**kw, "transport": httpx.MockTransport(silent)}))
    monkeypatch.setenv("FD_CREW_MODEL_SILENCE_SECONDS", "45")
    task = await _running_on(deck, TYPED)
    async with _worker(token=task["model_token"]) as worker:
        got = await worker.post("/fd/crew/llm/v1/chat/completions", json={})
    assert got.status_code == 504 and "said nothing for 45 seconds" in got.json()["error"]["message"]
    assert len(seen) == 1, "that's already been a long wait"


async def test_a_reply_cut_off_part_way_ends_there_for_the_cli_to_ask_again(deck, upstream, monkeypatch):
    """What happened on a real deck: DeepSeek dropped the connection 52 seconds into a reply."""
    _, replies = upstream

    class _CutOff(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
            raise httpx.ReadError("the connection went")

    replies.append(httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=_CutOff()))
    warned = []
    monkeypatch.setattr(crew_model.log, "warning", lambda *a, **k: warned.append(k))
    person = await _a_person(deck)
    task = await _running_on(deck, TYPED, user=person)
    async with _worker(token=task["model_token"]) as worker:
        got = await worker.post("/fd/crew/llm/v1/chat/completions", json={"stream": True, "messages": ["x" * 4000]})
    assert got.status_code == 200 and got.content.startswith(b"data:") and b"[DONE]" not in got.content, "no crash, and no pretending it finished"
    assert warned and warned[0]["error"] == "ReadError" and warned[0]["task"] == task["id"]
    spent = await _spent_today(deck, person["id"])
    assert spent["byok_tokens"] >= 1000, "it was never told what it used, and that isn't free"


async def test_its_model_being_heard_from_is_seen_by_the_watcher(deck, upstream):
    task = await _running_on(deck, TYPED)
    assert crew_model.last_heard(task["id"]) == 0.0 and crew_model.call_under_way(task["id"]) == {}
    async with _worker(token=task["model_token"]) as worker:
        await worker.post("/fd/crew/llm/v1/chat/completions", json={})
    assert crew_model.last_heard(task["id"]) > 0
    await crew_model.asyncio.sleep(0.05)
    call = crew_model.call_under_way(task["id"])
    assert call["open"] == 0 and call["got"] > 0, "it knows how much came back, and that it's over"


@pytest.mark.parametrize("status", [401, 503])
async def test_a_call_that_failed_isnt_still_under_way(deck, upstream, status):
    _, replies = upstream
    replies.extend(_reply(status, b'{"error":{"message":"no"}}') for _ in range(5))
    task = await _running_on(deck, TYPED)
    async with _worker(token=task["model_token"]) as worker:
        await worker.post("/fd/crew/llm/v1/chat/completions", json={})
    assert crew_model.call_under_way(task["id"])["open"] == 0


async def test_a_call_it_couldnt_make_isnt_either(deck, monkeypatch):
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no", request=request)

    monkeypatch.setattr(crew_model.httpx, "AsyncClient", lambda **kw: _REAL_CLIENT(**{**kw, "transport": httpx.MockTransport(down)}))
    task = await _running_on(deck, TYPED)
    async with _worker(token=task["model_token"]) as worker:
        await worker.post("/fd/crew/llm/v1/chat/completions", json={})
    assert crew_model.call_under_way(task["id"])["open"] == 0


async def test_a_server_it_cant_reach_is_said_plainly(deck, monkeypatch):
    task = await _running_on(deck, TYPED)

    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to 10.0.0.9", request=request)

    monkeypatch.setattr(crew_model.httpx, "AsyncClient", lambda **kw: _REAL_CLIENT(**{**kw, "transport": httpx.MockTransport(down)}))
    async with _worker(token=task["model_token"]) as worker:
        got = await worker.post("/fd/crew/llm/v1/chat/completions", json={})
    assert got.status_code == 502 and "couldn't reach" in got.json()["error"]["message"] and "10.0.0.9" not in got.text


async def test_asked_what_models_there_are_its_the_one(deck, upstream):
    calls, _ = upstream
    task = await _running_on(deck, TYPED)
    async with _worker(token=task["model_token"]) as worker:
        got = await worker.get("/fd/crew/llm/v1/models")
    assert [m["id"] for m in got.json()["data"]] == ["local-model"] and not calls


async def test_it_says_what_a_stream_used(deck, upstream):
    calls, replies = upstream
    chunks = b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: {"choices":[],"usage":{"prompt_tokens":70,"completion_tokens":7}}\n\ndata: [DONE]\n\n'
    replies.append(_reply(200, chunks, "text/event-stream"))
    await _team_tiers(deck, balanced={**ANTHROPIC_TIER, "provider": "openrouter", "model": "a/b", "api_key": "or-key"})
    person = await _a_person(deck)
    task = await _running_on(deck, {"kind": "tier", "tier": "balanced"}, user=person)
    async with _worker(token=task["model_token"]) as worker:
        got = await worker.post("/fd/crew/llm/v1/chat/completions", json={"stream": True, "messages": []})
    assert got.content == chunks, "and passes it on as it came"
    assert _sent(calls[0])["stream_options"] == {"include_usage": True}, "or it wouldn't say"
    assert calls[0].headers["accept-encoding"] == "identity"
    spent = await _spent_today(deck, person["id"])
    assert spent["tokens"] == 77 and not spent["byok_tokens"], "the team's tokens count toward their day"


@pytest.mark.parametrize("events, total", [
    # Anthropic's: what went in with the first event, what came out with the last. Its cache reads don't count.
    (b'event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":10,"cache_read_input_tokens":90,"cache_creation_input_tokens":5,"output_tokens":1}}}\n\n'
     b'event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":30}}\n\n', 45),
    # OpenAI's Responses.
    (b'data: {"type":"response.created","response":{"usage":null}}\n\ndata: {"type":"response.completed","response":{"usage":{"input_tokens":40,"output_tokens":2}}}\n\n', 42),
])
def test_the_tokens_of_each_kind_of_stream(events, total):
    count = crew_model._Count(stream=True)
    for at in range(0, len(events), 17):  # cut anywhere, as a network does
        count.feed(events[at:at + 17])
    assert count.finish() == total


def test_a_reply_cut_off_before_it_says_is_counted_by_what_went_through():
    events = b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n' * 100
    count = crew_model._Count(stream=True, asked=8000)
    count.feed(events)
    used = count.finish()
    assert used >= 2000 and used < 8000, "not nothing: hanging up early isn't a way round the count"
    anthropic = crew_model._Count(stream=True, asked=8000)
    anthropic.feed(b'data: {"type":"message_start","message":{"usage":{"input_tokens":5,"output_tokens":1}}}\n\n')
    assert anthropic.finish() >= 2000, "what it says so far doesn't let it off either"
    said = crew_model._Count(stream=True, asked=8000)
    said.feed(b'data: {"usage":{"prompt_tokens":70,"completion_tokens":7}}\n\n')
    assert said.finish() == 77, "and one that said is taken at its word, however quickly it hung up: Codex does"


@pytest.mark.parametrize("usage, missed", [
    ({"prompt_tokens": 100, "completion_tokens": 5}, 105),
    # What a real task was charged for, on DeepSeek: a 100k history read back from its cache on every call.
    ({"prompt_tokens": 100000, "prompt_cache_hit_tokens": 99500, "prompt_cache_miss_tokens": 500, "completion_tokens": 100}, 600),
    ({"prompt_tokens": 100000, "prompt_tokens_details": {"cached_tokens": 99000}, "completion_tokens": 100}, 1100),
    ({"input_tokens": 100000, "input_tokens_details": {"cached_tokens": 90000}, "output_tokens": 100}, 10100),
    ({"input_tokens": 500, "cache_read_input_tokens": 99000, "cache_creation_input_tokens": 200, "output_tokens": 100}, 800),
    ({"input_tokens": 500, "output_tokens": 100}, 600),
    ({"prompt_tokens": 10, "prompt_tokens_details": {"cached_tokens": None}, "completion_tokens": None}, 10),
    ({"prompt_tokens": 10, "prompt_tokens_details": {"cached_tokens": 50}}, 0),
])
def test_only_what_missed_the_prompt_cache_counts(usage, missed):
    """The same as an agent's call: llm.uncached_tokens."""
    count = crew_model._Count(stream=False)
    count.feed(json.dumps({"usage": usage}).encode())
    assert count.finish() == missed


def test_it_agrees_with_what_an_agent_counts():
    from captain_claw.llm import _extract_usage, uncached_tokens

    for usage in ({"prompt_tokens": 100000, "prompt_tokens_details": {"cached_tokens": 99000}, "completion_tokens": 100}, {"prompt_tokens": 70, "completion_tokens": 7}):
        count = crew_model._Count(stream=False)
        count.feed(json.dumps({"usage": usage}).encode())
        assert count.finish() == uncached_tokens("openai", _extract_usage(usage))
    count = crew_model._Count(stream=False)
    count.feed(json.dumps({"usage": {"input_tokens": 500, "cache_read_input_tokens": 99000, "cache_creation_input_tokens": 200, "output_tokens": 100}}).encode())
    assert count.finish() == uncached_tokens("anthropic", {"prompt_tokens": 99700, "cache_read_input_tokens": 99000, "cache_creation_input_tokens": 200, "completion_tokens": 100})


async def test_a_cut_off_call_counts_what_it_added_not_its_whole_history(deck, upstream):
    """Or one dropped connection late in a long session would cost a whole context."""
    _, replies = upstream
    person = await _a_person(deck)
    task = await _running_on(deck, TYPED, user=person)
    history = "x" * 400_000
    async with _worker(token=task["model_token"]) as worker:
        await worker.post("/fd/crew/llm/v1/chat/completions", json={"messages": [history]})
        before = (await _spent_today(deck, person["id"]))["byok_tokens"]
        replies.append(_reply(200, b'data: {"choices":[]}\n\n', "text/event-stream"))
        await worker.post("/fd/crew/llm/v1/chat/completions", json={"stream": True, "messages": [history, "y" * 4000]})
    grew = (await _spent_today(deck, person["id"]))["byok_tokens"] - before
    assert 1000 <= grew <= 2000


async def test_tokens_spent_stop_it_whether_or_not_the_messages_are_too(deck):
    person = await _a_person(deck)
    db = deck["db"]
    assert await usage_quota.automation_blocked(person["id"]) is None
    await db.add_daily_usage(person["id"], usage_quota.today(), messages=30)
    assert await usage_quota.automation_blocked(person["id"]) is None, "out of messages isn't out of tokens"
    await db.add_daily_usage(person["id"], usage_quota.today(), tokens=300_000)
    spent = await usage_quota.automation_blocked(person["id"])
    assert spent and spent["blocked"] == "tokens" and "AI tokens" in usage_quota.limit_text(spent)
    assert await usage_quota.automation_blocked(deck["admin"]["id"]) is None, "an admin has no limit"


def test_the_tokens_of_a_whole_reply():
    count = crew_model._Count(stream=False)
    count.feed(b'{"id":"x",\n"usage":{"input_tokens":3,\n"output_tokens":4}}')
    assert count.finish() == 7
    nothing = crew_model._Count(stream=False)
    nothing.feed(b"not json")
    assert nothing.finish() == 0
    never_says = crew_model._Count(stream=False, asked=400)
    never_says.feed(b'{"id":"x"}')
    assert never_says.finish() == 100, "a server that doesn't say isn't free"


async def test_what_a_worker_can_send_that_isnt_a_crash(deck, upstream, monkeypatch):
    calls, _ = upstream
    task = await _running_on(deck, TYPED)
    async with _worker(token=task["model_token"]) as worker:
        got = await worker.post("/fd/crew/llm/v1/chat/completions", json={"stream": True, "stream_options": "yes", "models": ["other/one"], "route": "fallback"})
        assert got.status_code == 200 and _sent(calls[-1])["stream_options"] == {"include_usage": True}
        assert "models" not in _sent(calls[-1]) and "route" not in _sent(calls[-1]), "no other model on this key"
        assert (await worker.post("/fd/crew/llm/v1/chat/completions", content=b"not json")).status_code == 400
        assert (await worker.post("/fd/crew/llm/v1/chat/completions", content=b"[1]")).status_code == 400
    async with _worker() as wrong:
        got = await wrong.post("/fd/crew/llm/v1/chat/completions", json={}, headers={"x-api-key": "crew-caf\u00e9".encode("latin-1")})
        assert got.status_code == 401


async def test_a_bad_address_on_a_tier_is_said_plainly(deck):
    person = await _a_person(deck)
    await _team_tiers(deck, balanced={**ANTHROPIC_TIER, "provider": "openai", "base_url": "http://[not-a-host/v1", "api_key": "k"})
    task = await _running_on(deck, {"kind": "tier", "tier": "balanced"}, user=person)
    async with _worker(token=task["model_token"]) as worker:
        got = await worker.post("/fd/crew/llm/v1/chat/completions", json={})
    assert got.status_code == 502 and "couldn't reach" in got.json()["error"]["message"]


async def test_a_refused_requests_log_line_doesnt_carry_what_it_was_sent(monkeypatch):
    from fastapi.exceptions import RequestValidationError

    logged = []
    monkeypatch.setattr(server.log, "error", lambda *a, **k: logged.append(a))

    class _Asked:
        method = "POST"
        url = type("U", (), {"path": "/fd/crew/tasks"})

    bad = RequestValidationError([{"type": "missing", "loc": ["body", "harness"], "msg": "Field required", "input": {"model_source": {"key": THEIR_KEY}}}])
    await server.validation_exception_handler(_Asked(), bad)
    assert logged and THEIR_KEY not in repr(logged) and "harness" in repr(logged)


async def test_their_own_key_counts_apart(deck, upstream):
    person = await _a_person(deck)
    task = await _running_on(deck, TYPED, user=person)
    async with _worker(token=task["model_token"]) as worker:
        await worker.post("/fd/crew/llm/v1/chat/completions", json={})
    spent = await _spent_today(deck, person["id"])
    assert spent["byok_tokens"] == 120 and not spent["tokens"]


async def test_a_day_of_tokens_spent_stops_the_teams_but_not_their_own(deck, upstream, monkeypatch):
    calls, _ = upstream
    person = await _a_person(deck)
    await _team_tiers(deck, balanced=ANTHROPIC_TIER)
    team = await _running_on(deck, {"kind": "tier", "tier": "balanced"}, user=person)
    own = await _running_on(deck, TYPED, user=person)

    async def spent(uid):
        return {"blocked": "tokens", "plan_label": "Free", "tokens": {"limit": 100, "used": 100}, "resets_at": "2030-01-01T00:00:00+00:00"}

    monkeypatch.setattr(usage_quota, "automation_blocked", spent)
    async with _worker(token=team["model_token"]) as worker:
        got = await worker.post("/fd/crew/llm/v1/chat/completions", json={})
    assert got.status_code == 402 and "tokens" in got.json()["error"]["message"], "not a 429: it would only keep trying. Not a 403: Claude Code calls that failing to authenticate"
    assert not calls
    async with _worker(token=own["model_token"]) as worker:
        assert (await worker.post("/fd/crew/llm/v1/chat/completions", json={})).status_code == 200
    with pytest.raises(HTTPException) as refused:
        await _running_on(deck, {"kind": "tier", "tier": "balanced"}, user=await _a_person(deck, "b"))
    assert refused.value.status_code == 429


# ── Looked up again ──────────────────────────────────────────────────────


async def test_a_tier_that_changes_is_what_the_next_turn_and_the_next_call_run_on(deck, upstream):
    calls, _ = upstream
    await _team_tiers(deck, balanced=ANTHROPIC_TIER)
    person = await _a_person(deck)
    done = await _settle(await _as_them(deck, person, "SCENARIO:work", model_source={"kind": "tier", "tier": "balanced"}))
    await _team_tiers(deck, balanced={**ANTHROPIC_TIER, "model": "claude-newer"})
    again = await _settle(await crew_runner.steer(done["user_id"], done["id"], "go on"))
    assert again["model"] == "claude-newer" and "--model claude-newer" in _seen(again)
    said = [e["text"] for e in crew_store.read_events(again["user_id"], again["id"]) if e["kind"] == "note"]
    assert any("Balanced tier is claude-newer now" in s for s in said)


async def test_a_turn_that_cant_have_its_model_fails_and_says_why(deck, agents):
    person = await _a_person(deck)
    agents("mine", person["id"])
    done = await _settle(await _as_them(deck, person, "SCENARIO:work", model_source={"kind": "agent", "agent": "mine"}))
    assert done["model"] == "claude-agent" and done["model_source"]["label"] == "Mine's model"
    shutil.rmtree(deck["tmp"] / "fd-data" / "mine")
    server.PROCESS_REGISTRY_FILE.write_text("{}")
    again = await _settle(await crew_runner.steer(done["user_id"], done["id"], "go on"))
    assert again["status"] == "failed" and "no agent called mine" in again["note"]


async def test_a_placeholder_in_an_agents_config_is_no_key(deck, agents):
    person = await _a_person(deck)
    agents("mine", person["id"], env="ANTHROPIC_API_KEY=sk-from-its-env\n", config={"model": {"provider": "anthropic", "model": "m", "api_key": "@system"}})
    assert (await crew_model.resolve(deck["db"], person, NOT_ADMIN, {"kind": "agent", "agent": "mine"})).key == "sk-from-its-env"


async def test_an_agents_own_key_is_counted_as_its_own(deck, agents):
    person = await _a_person(deck)
    agents("mine", person["id"], env="ANTHROPIC_API_KEY=sk-their-agent\n", key_source="byok")
    got = await crew_model.resolve(deck["db"], person, NOT_ADMIN, {"kind": "agent", "agent": "mine"})
    assert got.key == "sk-their-agent" and got.own_keys and got.context == 90000 and got.apis == ("anthropic",)
