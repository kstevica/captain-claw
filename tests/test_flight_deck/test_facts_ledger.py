"""Increment 4 of the quality-tightening plan: the shared facts ledger.

The load-bearing behaviour: a second writer offering a DIFFERENT value never
silently overwrites — the original stays canonical, the offer is recorded, and
the writer is told to reconcile. Same-value writes merge metadata (a verified
write upgrades an assumed one). The ledger's numeric rows feed the consistency
check, closing the text-vs-ledger loop.
"""

from __future__ import annotations

from captain_claw.flight_deck import facts_ledger as fl
from captain_claw.flight_deck.research_consistency import verify
from captain_claw.flight_deck.research_verify import claim_check_prompt


# ── store basics ─────────────────────────────────────────────────────

def test_create_get_roundtrip_and_key_normalisation(tmp_path):
    res = fl.upsert(tmp_path, "Total Budget (EUR)", "300000", unit="EUR",
                    status="derived", provenance="derived from cost lines",
                    updated_by="analyst")
    assert res["ok"] and res["action"] == "created"
    r = fl.get(tmp_path, "total_budget_eur")
    assert r is not None
    assert r["key"] == "total_budget_eur"
    assert r["value"] == "300000" and r["unit"] == "EUR"
    assert r["status"] == "derived" and r["updated_by"] == "analyst"


def test_default_and_invalid_status_degrade_to_weakest(tmp_path):
    fl.upsert(tmp_path, "a", "1")
    assert fl.get(tmp_path, "a")["status"] == "assumed"
    fl.upsert(tmp_path, "b", "2", status="definitely-true")
    assert fl.get(tmp_path, "b")["status"] == "assumed"


def test_same_value_merges_metadata_upgrade(tmp_path):
    fl.upsert(tmp_path, "employees", "51", status="assumed", updated_by="writer")
    res = fl.upsert(tmp_path, "employees", "51", status="verified",
                    provenance="registry 2024", updated_by="researcher")
    assert res["ok"] and res["action"] == "merged"
    r = fl.get(tmp_path, "employees")
    assert r["status"] == "verified" and r["provenance"] == "registry 2024"
    assert fl.conflicts(tmp_path) == []


def test_numeric_tolerance_is_relative_only(tmp_path):
    # Rounding drift on a large figure merges…
    fl.upsert(tmp_path, "total", "300000")
    assert fl.upsert(tmp_path, "total", "300000.4")["action"] == "merged"
    # …but a small-magnitude difference conflicts (0.35 vs 0.36 is 2.8%).
    fl.upsert(tmp_path, "intensity", "0.35")
    res = fl.upsert(tmp_path, "intensity", "0.36")
    assert res["ok"] is False and res["reason"] == "conflict"


# ── the conflict rule ────────────────────────────────────────────────

def test_conflicting_value_is_not_saved_and_is_recorded(tmp_path):
    fl.upsert(tmp_path, "personnel_eur", "157000", updated_by="analyst")
    res = fl.upsert(tmp_path, "personnel_eur", "549620", updated_by="writer")
    assert res["ok"] is False and res["reason"] == "conflict"
    assert "157000" in res["message"] and "NOT saved" in res["message"]
    # Canonical value unchanged; the offer is on the record.
    assert fl.get(tmp_path, "personnel_eur")["value"] == "157000"
    confs = fl.conflicts(tmp_path)
    assert len(confs) == 1
    assert confs[0]["offered_value"] == "549620" and confs[0]["forced"] == 0


def test_force_replaces_but_keeps_history(tmp_path):
    fl.upsert(tmp_path, "k", "10")
    res = fl.upsert(tmp_path, "k", "20", force=True, updated_by="lead")
    assert res["ok"] and res["action"] == "forced" and res["previous"] == "10"
    assert fl.get(tmp_path, "k")["value"] == "20"
    confs = fl.conflicts(tmp_path)
    assert len(confs) == 1 and confs[0]["forced"] == 1


def test_row_cap_blocks_new_keys_not_updates(tmp_path, monkeypatch):
    monkeypatch.setattr(fl, "ROW_CAP", 2)
    assert fl.upsert(tmp_path, "a", "1")["ok"]
    assert fl.upsert(tmp_path, "b", "2")["ok"]
    assert fl.upsert(tmp_path, "c", "3")["reason"] == "full"
    # Existing keys still update at the cap.
    assert fl.upsert(tmp_path, "a", "1", status="verified")["ok"]


# ── exports ──────────────────────────────────────────────────────────

def test_export_rows_numeric_only_and_empty_without_db(tmp_path):
    assert fl.export_rows(tmp_path) == []  # no .facts.db → no rows, no side effects
    fl.upsert(tmp_path, "total_eur", "300000", unit="EUR")
    fl.upsert(tmp_path, "call_ref", "DIGIT.2.1.03")  # non-numeric
    rows = fl.export_rows(tmp_path)
    assert rows == [{"key": "total_eur", "value": 300000.0, "unit": "EUR"}]


def test_dump_markdown_lists_rows_and_open_conflicts(tmp_path):
    assert fl.dump_markdown(tmp_path) == ""
    fl.upsert(tmp_path, "total_eur", "300000", unit="EUR", status="derived")
    fl.upsert(tmp_path, "total_eur", "350000", updated_by="writer")  # conflict
    md = fl.dump_markdown(tmp_path)
    assert "total_eur" in md and "300000" in md
    assert "Unresolved value conflicts" in md and "350000" in md


# ── integration: ledger rows drive the consistency check ─────────────

def test_ledger_rows_catch_text_that_contradicts_the_ledger(tmp_path):
    fl.upsert(tmp_path, "total_eligible_cost", "300000", unit="EUR", status="derived")
    entries = {"values": [{
        "label": "total eligible cost", "kind": "figure", "raw": "€350,000",
        "value": 350000, "unit": "EUR", "quote": "a total of €350,000",
    }], "relations": []}
    findings = verify(entries, ledger_rows=fl.export_rows(tmp_path))
    assert len(findings) == 1
    assert findings[0]["kind"] == "ledger" and findings[0]["severity"] == "critical"


# ── claim-check prompt carries the ledger as claimed provenance ──────

def test_claim_check_prompt_includes_facts_block_only_when_given():
    plain = claim_check_prompt("the doc", "the task", 8)
    assert "facts ledger" not in plain.lower()
    with_facts = claim_check_prompt("the doc", "the task", 8,
                                    facts_block="| total_eur | 300000 |")
    assert "facts ledger" in with_facts.lower()
    assert "| total_eur | 300000 |" in with_facts
    assert with_facts.index("total_eur") < with_facts.index("## Deliverable to verify")


# ── the `facts` tool ─────────────────────────────────────────────────

async def test_facts_tool_set_get_list_and_conflict(tmp_path, monkeypatch):
    from captain_claw import vfs
    from captain_claw.tools.facts import FactsTool
    monkeypatch.setattr(vfs, "project_root", lambda *a, **k: tmp_path)
    monkeypatch.setattr(vfs, "agent_label", lambda: "analyst")
    tool = FactsTool()

    r = await tool.execute(action="set", key="grant_eur", value="105000",
                           unit="EUR", status="derived",
                           provenance="derived from intensity × total")
    assert r.success and "recorded" in r.content and "grant_eur" in r.content

    r = await tool.execute(action="get", key="grant_eur")
    assert r.success and "105000" in r.content and "[derived]" in r.content

    # Conflict comes back as readable content, not a tool error (no blind retry).
    r = await tool.execute(action="set", key="grant_eur", value="150000")
    assert r.success and r.content.startswith("CONFLICT")
    assert "105000" in r.content

    r = await tool.execute(action="list")
    assert r.success and "grant_eur" in r.content and "Unresolved value conflicts" in r.content

    r = await tool.execute(action="get", key="never_set")
    assert r.success and "not in the ledger" in r.content


async def test_facts_tool_errors_cleanly_without_a_folder(monkeypatch):
    from captain_claw import vfs
    from captain_claw.tools.facts import FactsTool

    def _boom(*a, **k):
        raise RuntimeError("no root")

    monkeypatch.setattr(vfs, "project_root", _boom)
    r = await FactsTool().execute(action="list")
    assert r.success is False and "no shared VFS folder" in r.error
