"""Unit tests for the pure deliverable-manifest logic (Increment 3)."""

from captain_claw.flight_deck import deliverable_manifest as dm


_SUBTASKS = [
    {"id": "s1", "title": "Story Bible", "owner_archetype_id": "planner", "depends_on": []},
    {"id": "s4", "title": "Part One", "owner_archetype_id": "editor-writer", "depends_on": ["s1"]},
    {"id": "s5", "title": "Part Two", "owner_archetype_id": "editor-writer", "depends_on": ["s1"]},
]


def test_word_to_int():
    assert dm.word_to_int("one") == 1
    assert dm.word_to_int("twelve") == 12
    assert dm.word_to_int("twenty-one") == 21
    assert dm.word_to_int("7") == 7
    assert dm.word_to_int("banana") is None


def test_parse_range():
    assert dm.parse_range("ch6-7") == (6, 7)
    assert dm.parse_range("ch6") == (6, 6)
    assert dm.parse_range("chapters six to seven") == (6, 7)
    assert dm.parse_range([3, 1]) == (1, 3)
    assert dm.parse_range("") is None
    assert dm.parse_range(None) is None


def test_chapters_in_and_count_sections():
    text = "# Chapter One\n\nbody\n\n## Chapter Two\n\nmore\n\n# Chapter 3\n"
    assert dm.chapters_in(text) == [1, 2, 3]
    assert dm.count_sections(text) == 3
    assert dm.count_sections("no headings here") == 0


def test_slugify():
    assert dm.slugify("Part One: The Archive!") == "part-one-the-archive"
    assert dm.slugify("x" * 80).count("x") == 40


def test_parse_explicit_manifest():
    raw = {
        "path": "fair-measure.md", "kind": "fiction", "min_bytes": 60000, "min_sections": 12,
        "parts": [
            {"path": "part-two-ch6-10.md", "order": 2, "range": "ch6-10", "owner": "s5"},
            {"path": "part-one-ch1-5.md", "order": 1, "range": "ch1-5", "owner": "s4"},
        ],
        "seam_owner": 0,
    }
    m = dm.parse(raw, _SUBTASKS, "vatra-abcd1234")
    assert m is not None
    assert m.path == "vfs:vatra-abcd1234/fair-measure.md"
    assert m.kind == "fiction"
    assert m.min_bytes == 60000 and m.min_sections == 12
    # parts sorted by order; both fiction → sequential
    assert [p.basename() for p in m.parts] == ["part-one-ch1-5.md", "part-two-ch6-10.md"]
    assert m.parts[0].range == (1, 5) and m.parts[1].range == (6, 10)
    assert all(p.sequential for p in m.parts)
    # seam_owner index 0 → the order-1 part's owner
    assert m.seam_owner == "s4"
    assert m.declared_basenames() == ["fair-measure.md", "part-one-ch1-5.md", "part-two-ch6-10.md"]


def test_parse_requires_extension_and_drops_unknown_owner_and_dupes():
    raw = {
        "path": "book",  # no ext → .md appended
        "parts": [
            {"path": "a.md", "owner": "s4"},
            {"path": "a.md", "owner": "s5"},   # duplicate path → dropped
            {"path": "b.md", "owner": "ghost"},  # unknown owner → dropped
        ],
    }
    m = dm.parse(raw, _SUBTASKS, "proj")
    assert m.path == "vfs:proj/book.md"
    assert [p.basename() for p in m.parts] == ["a.md"]


def test_parse_none_when_empty():
    assert dm.parse(None, _SUBTASKS, "proj") is None
    assert dm.parse({}, _SUBTASKS, "proj") is None
    assert dm.parse({"path": ""}, _SUBTASKS, "proj") is None


def test_topo_layers():
    layers = dm.topo_layers(_SUBTASKS)
    assert layers[0] == ["s1"]
    assert set(layers[1]) == {"s4", "s5"}


def test_topo_layers_cycle():
    cyc = [
        {"id": "a", "depends_on": ["b"]},
        {"id": "b", "depends_on": ["a"]},
    ]
    layers = dm.topo_layers(cyc)
    # cycle members collapse into one layer
    assert set(layers[-1]) == {"a", "b"}


def test_derive_from_plan_and_adds_range_edge():
    subs = [dict(s, depends_on=list(s["depends_on"])) for s in _SUBTASKS]
    g0 = {
        "s4": {"produces_file": "part-one.md", "range": "ch1-5"},
        "s5": {"produces_file": "part-two.md", "range": "ch6-10"},
    }
    m = dm.derive(g0, subs, "proj", kind_hint="fiction")
    assert m is not None
    paths = [p.basename() for p in m.parts]
    assert "part-one.md" in paths and "part-two.md" in paths
    # s1 (no produces_file) → derived name
    assert any(p.owner == "s1" for p in m.parts)
    # consecutive ranged parts get a depends_on edge (s5 depends on s4)
    s5 = next(s for s in subs if s["id"] == "s5")
    assert "s4" in s5["depends_on"]
    assert m.seam_owner == "s5"


def test_assign_artifacts_and_inputs_for():
    subs = [dict(s, depends_on=list(s["depends_on"])) for s in _SUBTASKS]
    m = dm.parse({"path": "book.md", "parts": [
        {"path": "s1.md", "owner": "s1"},
        {"path": "part-one.md", "owner": "s4"},
    ]}, subs, "proj")
    mapping = dm.assign_artifacts(subs, m)
    assert mapping["s4"] == "vfs:proj/part-one.md"
    s4 = next(s for s in subs if s["id"] == "s4")
    assert s4["artifact"] == "vfs:proj/part-one.md"
    # s4 depends_on s1 → its input is s1's part
    inputs = dm.inputs_for(m, "s4", subs)
    assert [p.basename() for p in inputs] == ["s1.md"]


def test_to_analysis():
    m = dm.parse({"path": "book.md", "kind": "document",
                  "parts": [{"path": "a.md", "owner": "s4", "order": 1}]}, _SUBTASKS, "proj")
    a = dm.to_analysis(m)
    assert a["path"] == "vfs:proj/book.md"
    assert a["kind"] == "document"
    assert a["parts"][0]["owner"] == "s4"


# ── Increment 4: part_status, assemble, gate ─────────────────────────

def _write(tmp_path, name, body):
    (tmp_path / name).write_text(body, encoding="utf-8")


def test_part_status_landed_and_placeholder(tmp_path):
    m = dm.parse({"path": "book.md", "kind": "fiction", "parts": [
        {"path": "part-one.md", "owner": "s4", "range": "ch1-2", "min_bytes": 10},
    ]}, _SUBTASKS, "proj")
    p = m.parts[0]
    # missing file
    assert dm.part_status(tmp_path, p)["landed"] is False
    # real content
    _write(tmp_path, "part-one.md", "# Chapter One\n\nx\n\n# Chapter Two\n\ny\n" * 3)
    st = dm.part_status(tmp_path, p, producer_done=True)
    assert st["exists"] and st["landed"] and st["bytes"] >= 10
    assert set(st["chapters"]) == {1, 2}
    # sequential + producer not done → not landed
    assert dm.part_status(tmp_path, p, producer_done=False)["landed"] is False
    # placeholder body
    _write(tmp_path, "part-one.md", "[written to disk: part-one.md, 1 lines, 0.0KB — use read tool to view]")
    stp = dm.part_status(tmp_path, p)
    assert stp["placeholder"] is True and stp["landed"] is False


def test_assemble_orders_and_flags_seams(tmp_path):
    m = dm.parse({"path": "book.md", "kind": "fiction", "parts": [
        {"path": "p2.md", "owner": "s5", "order": 2, "range": "ch6-7"},
        {"path": "p1.md", "owner": "s4", "order": 1, "range": "ch1-2"},
    ]}, _SUBTASKS, "proj")
    _write(tmp_path, "p1.md", "# Chapter One\n\na\n\n# Chapter Two\n\nb\n")
    _write(tmp_path, "p2.md", "# Chapter Six\n\nc\n")  # missing chapter 7
    res = dm.assemble(tmp_path, m)
    # ordered p1 then p2
    assert res["text"].index("Chapter One") < res["text"].index("Chapter Six")
    kinds = {f["kind"] for f in res["findings"]}
    assert "missing_chapter" in kinds


def test_assemble_duplicate_and_placeholder(tmp_path):
    m = dm.parse({"path": "book.md", "parts": [
        {"path": "p1.md", "owner": "s4", "order": 1},
        {"path": "p2.md", "owner": "s5", "order": 2},
    ]}, _SUBTASKS, "proj")
    _write(tmp_path, "p1.md", "# Chapter One\n\na\n")
    _write(tmp_path, "p2.md", "# Chapter One\n\nduplicate\n")  # dup chapter 1
    res = dm.assemble(tmp_path, m)
    assert any(f["kind"] == "duplicate_chapter" for f in res["findings"])
    # placeholder part
    _write(tmp_path, "p2.md", "[written to disk: p2.md, 1 lines, 0.0KB — use read tool to view]")
    res2 = dm.assemble(tmp_path, m)
    assert any(f["kind"] == "placeholder_part" for f in res2["findings"])


def test_assemble_part_missing(tmp_path):
    m = dm.parse({"path": "book.md", "parts": [{"path": "p1.md", "owner": "s4"}]}, _SUBTASKS, "proj")
    res = dm.assemble(tmp_path, m)
    assert any(f["kind"] == "part_missing" for f in res["findings"])


def test_gate_reasons():
    m = dm.parse({"path": "book.md", "min_bytes": 100, "min_sections": 2}, _SUBTASKS, "proj")
    assert dm.gate(m, "")["reasons"] == ["deliverable_missing"]
    assert dm.gate(m, "[written to disk: x — use read tool to view]")["reasons"] == ["deliverable_placeholder"]
    r = dm.gate(m, "short")
    assert any("below_min_bytes" in x for x in r["reasons"])
    big = "# Chapter One\n\n" + "x" * 200
    r2 = dm.gate(m, big)
    assert any("too_few_sections" in x for x in r2["reasons"])
    ok = "# Chapter One\n\n" + "x" * 100 + "\n\n# Chapter Two\n\n" + "y" * 100
    assert dm.gate(m, ok)["ok"] is True
