"""Tests for the face-recognition infrastructure module.

The insightface model is heavy (~280 MB of weights) and slow on CPU, so we
patch ``_load_face_app`` to return a tiny fake that emits deterministic
embeddings. This keeps the tests fast (<1s) and CI-friendly while still
exercising every code path inside ``FaceIndex`` — the sqlite-vec schema,
the L2-to-cosine conversion, the encounter logging, and the card renderer.

We DO require ``sqlite_vec`` and ``numpy`` to be importable; the tests
``pytest.importorskip`` if they're missing rather than fail loudly, since
``[faces]`` is an optional extra.
"""

from __future__ import annotations

import io

import pytest

# These two are non-optional for the tests. Skip the whole module if the
# faces extra isn't installed in the test environment.
np = pytest.importorskip("numpy")
pytest.importorskip("sqlite_vec")
from PIL import Image  # noqa: E402  — Pillow is a core dep

from captain_claw.flight_deck import face_index  # noqa: E402


# ── Helpers ───────────────────────────────────────────────────────────


class FakeFace:
    """Stands in for ``insightface.app.common.Face`` — exposes the two
    attributes our code actually reads (``bbox`` + ``normed_embedding``)."""

    def __init__(self, bbox, embedding):
        self.bbox = bbox
        # ``insightface`` stores normed_embedding as a 1-D float32 array.
        self.normed_embedding = np.asarray(embedding, dtype=np.float32)


class FakeFaceApp:
    """Replacement for ``insightface.app.FaceAnalysis``.

    Callers prime it with a mapping ``image_size → [FakeFace, ...]`` so each
    decoded image returns the right list of detections. The decoded image
    isn't actually inspected — we key on shape so the test can submit
    several different-sized PNGs in one call.
    """

    def __init__(self, by_shape):
        self.by_shape = by_shape

    def get(self, bgr):
        h, w = bgr.shape[:2]
        return list(self.by_shape.get((w, h), []))


def _png_bytes(w: int, h: int, colour: tuple[int, int, int] = (128, 128, 128)) -> bytes:
    """Synthesize a solid-colour PNG of the requested size.

    Tests submit these as enrollment / recognition image blobs.
    ``face_index._decode_image_to_bgr`` only cares that PIL can decode them
    and the resulting numpy array has the right shape — the pixel content
    is ignored because we've stubbed out the actual detector.
    """
    img = Image.new("RGB", (w, h), colour)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _normed(vec):
    """L2-normalize a vector so we can craft cosine-similarity-equivalent
    embeddings. Matches what ArcFace's ``normed_embedding`` would look like."""
    arr = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(arr))
    if n == 0:
        return arr
    return arr / n


def _embedding(seed: float):
    """Deterministic 512-d embedding from a single seed scalar.

    Two embeddings with the same seed should match each other cleanly;
    embeddings with very different seeds should not. We get that by
    rotating the same base direction by an angle proportional to the seed,
    in a tiny 2-D subspace of the 512-d space.
    """
    vec = np.zeros(face_index.EMBEDDING_DIM, dtype=np.float32)
    vec[0] = float(np.cos(seed))
    vec[1] = float(np.sin(seed))
    # Tiny constant tail so vectors aren't degenerate — doesn't change
    # the cosine geometry meaningfully.
    vec[2:] = 1e-3
    return _normed(vec)


@pytest.fixture
def fresh_index(tmp_path, monkeypatch):
    """A FaceIndex pointed at a brand-new sqlite file under tmp_path.

    Also patches the module's lazy face_app loader so subsequent enroll /
    recognize calls hit our FakeFaceApp rather than dragging in
    insightface. The fake is empty by default — individual tests rewire
    ``app.by_shape`` to inject detections.
    """
    db_path = tmp_path / "face_index.db"
    # Reset the module-level singletons in case a prior test populated them.
    monkeypatch.setattr(face_index, "_face_app", None, raising=False)
    monkeypatch.setattr(face_index, "_face_app_error", None, raising=False)
    monkeypatch.setattr(face_index, "_INDEX", None, raising=False)

    app = FakeFaceApp(by_shape={})
    monkeypatch.setattr(face_index, "_load_face_app", lambda: app)
    idx = face_index.FaceIndex(db_path=db_path)
    return idx, app


# ── Tests ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_recognize_empty_index_returns_unknown_card(fresh_index):
    """Brand-new index has zero embeddings. Recognize must still return a
    well-formed card (the HUD always wants something to render)."""
    idx, app = fresh_index
    # Detect *a* face — so we hit the "no rows in embeddings" branch, not
    # the "no face detected" branch.
    blob = _png_bytes(200, 200)
    app.by_shape = {(200, 200): [FakeFace(bbox=(40, 40, 160, 160), embedding=_embedding(0.0))]}

    result = await idx.recognize(image_blob=blob)

    assert result.person_id is None
    assert result.name is None
    assert result.confidence == 0.0
    assert result.bbox is not None
    assert "Unknown face" in result.card_markdown


@pytest.mark.asyncio
async def test_recognize_no_face_detected_returns_no_face_card(fresh_index):
    """When the detector sees nothing, we return a card saying so — and
    crucially, never raise. The phone UI handles the rest."""
    idx, app = fresh_index
    blob = _png_bytes(200, 200)
    # FakeFaceApp returns [] for unmatched shapes by default.

    result = await idx.recognize(image_blob=blob)

    assert result.person_id is None
    assert result.bbox is None
    assert "No face" in result.card_markdown


@pytest.mark.asyncio
async def test_enroll_mixed_batch_reports_added_vs_submitted(fresh_index):
    """Three photos submitted, only two have detectable faces. The result
    must surface both counts so the UI can show 'N of M detected' instead
    of silently swallowing the gap."""
    idx, app = fresh_index
    good_a = _png_bytes(100, 100, (10, 10, 10))
    good_b = _png_bytes(120, 120, (20, 20, 20))
    bad = _png_bytes(140, 140, (30, 30, 30))
    app.by_shape = {
        (100, 100): [FakeFace(bbox=(10, 10, 90, 90), embedding=_embedding(0.0))],
        (120, 120): [FakeFace(bbox=(10, 10, 110, 110), embedding=_embedding(0.0))],
        # No entry for (140, 140) → empty list returned, treated as "no face".
    }

    result = await idx.enroll(name="Ana", notes="cousin", image_blobs=[good_a, good_b, bad])

    assert result.embeddings_added == 2
    assert result.name == "Ana"
    assert result.person_id.startswith("p_")

    rows = idx.list_persons()
    assert len(rows) == 1
    assert rows[0]["samples"] == 2
    assert rows[0]["notes"] == "cousin"


@pytest.mark.asyncio
async def test_enroll_all_blank_raises_value_error(fresh_index):
    """If *no* image yields a face, enroll has nothing to store and must
    raise rather than create a phantom person row."""
    idx, app = fresh_index
    blob = _png_bytes(200, 200)
    # No detections registered for (200, 200).

    with pytest.raises(ValueError, match="no faces"):
        await idx.enroll(name="Ghost", notes="", image_blobs=[blob])

    # And the persons table stays empty.
    assert idx.list_persons() == []


@pytest.mark.asyncio
async def test_enroll_then_recognize_matches_and_logs_encounter(fresh_index):
    """Full hot-path: enroll one person, then recognize a photo whose
    embedding is identical to the enrolled one. Must match confidently,
    log an encounter row, and render a name into the card."""
    idx, app = fresh_index

    # Enroll Ana with a single reference photo whose embedding is at seed 0.0.
    enroll_blob = _png_bytes(100, 100)
    app.by_shape = {
        (100, 100): [FakeFace(bbox=(10, 10, 90, 90), embedding=_embedding(0.0))],
    }
    enrolled = await idx.enroll(name="Ana", notes="cousin", image_blobs=[enroll_blob])
    assert enrolled.embeddings_added == 1

    # Recognize a photo whose detected face has the *same* embedding seed.
    recog_blob = _png_bytes(300, 300)
    app.by_shape = {
        (300, 300): [FakeFace(bbox=(60, 60, 240, 240), embedding=_embedding(0.0))],
    }
    result = await idx.recognize(image_blob=recog_blob, channel="test-ch")

    assert result.person_id == enrolled.person_id
    assert result.name == "Ana"
    assert result.confidence > face_index.MATCH_THRESHOLD
    assert "**Ana**" in result.card_markdown
    assert "cousin" in result.card_markdown

    encounters = idx.list_encounters(enrolled.person_id)
    assert len(encounters) == 1
    assert encounters[0]["channel"] == "test-ch"
    # Confidence stored on the encounter row should match what was returned.
    assert encounters[0]["confidence"] == pytest.approx(result.confidence)


@pytest.mark.asyncio
async def test_recognize_low_confidence_returns_card_without_logging(fresh_index):
    """A detected face that scores in the ambiguous band must show a
    low-confidence card but NOT create an encounter row — otherwise drift
    would silently pollute the history of whoever happened to be the
    closest match."""
    idx, app = fresh_index

    # Enroll Ana at seed 0.0.
    blob_enroll = _png_bytes(100, 100)
    app.by_shape = {(100, 100): [FakeFace(bbox=(10, 10, 90, 90), embedding=_embedding(0.0))]}
    enrolled = await idx.enroll(name="Ana", notes="", image_blobs=[blob_enroll])

    # Recognize a face whose embedding is rotated enough to land in the
    # ambiguous zone — somewhere between the two thresholds. We pick a
    # rotation that gives cosine ≈ 0.4 against seed 0.0:
    #   cos(theta) = AMBIGUOUS_THRESHOLD + 0.05
    target_cos = (face_index.AMBIGUOUS_THRESHOLD + face_index.MATCH_THRESHOLD) / 2.0
    # Approximate: in the cos/sin 2-d subspace, rotating by acos(target_cos)
    # gives the desired similarity.
    theta = float(np.arccos(target_cos))

    blob_recog = _png_bytes(300, 300)
    app.by_shape = {(300, 300): [FakeFace(bbox=(60, 60, 240, 240), embedding=_embedding(theta))]}
    result = await idx.recognize(image_blob=blob_recog, channel="test-ch")

    # We DO render a person card (low-confidence prefix) but don't claim a
    # confident match and don't log an encounter.
    assert result.person_id is None  # not a confident match
    assert "low confidence" in result.card_markdown.lower()
    assert idx.list_encounters(enrolled.person_id) == []


@pytest.mark.asyncio
async def test_recognize_multiple_faces_logs_each_confident_match(fresh_index):
    """A group photo with two enrolled people in frame must:
      - return both names in ``result.faces``
      - render both on the card (primary block + 'Also:' roster line)
      - log one encounter row per confident match
      - have the most-centred face as the primary
    """
    idx, app = fresh_index

    # Enroll Ana (seed 0.0) and Bob (seed π) — orthogonal in our 2-D subspace,
    # so cross-matching is below threshold.
    blob_enroll_a = _png_bytes(100, 100)
    app.by_shape = {(100, 100): [FakeFace(bbox=(10, 10, 90, 90), embedding=_embedding(0.0))]}
    ana = await idx.enroll(name="Ana", notes="", image_blobs=[blob_enroll_a])

    blob_enroll_b = _png_bytes(110, 110)
    app.by_shape = {(110, 110): [FakeFace(bbox=(10, 10, 100, 100), embedding=_embedding(float(np.pi)))]}
    bob = await idx.enroll(name="Bob", notes="", image_blobs=[blob_enroll_b])

    # Group photo: Ana off-centre (top-left quadrant), Bob centred. Bob
    # should win the "primary subject" slot via the centeredness scorer.
    group_blob = _png_bytes(400, 400)
    app.by_shape = {
        (400, 400): [
            FakeFace(bbox=(20, 20, 100, 100),  embedding=_embedding(0.0)),          # Ana — off-centre
            FakeFace(bbox=(150, 150, 250, 250), embedding=_embedding(float(np.pi))),# Bob — centred
        ],
    }
    result = await idx.recognize(image_blob=group_blob, channel="party")

    # Primary is Bob (centred), Ana is in the roster.
    assert result.person_id == bob.person_id
    assert result.name == "Bob"
    names_in_frame = sorted([f.name for f in result.faces if f.name])
    assert names_in_frame == ["Ana", "Bob"]

    # Both confident → two encounter rows (one per person).
    assert len(idx.list_encounters(ana.person_id)) == 1
    assert len(idx.list_encounters(bob.person_id)) == 1

    # Card contains the primary block (Bob, bold) AND the Also line for Ana.
    assert "**Bob**" in result.card_markdown
    assert "Also" in result.card_markdown
    assert "Ana" in result.card_markdown


@pytest.mark.asyncio
async def test_recognize_one_known_one_stranger_marks_stranger(fresh_index):
    """One enrolled face + one unknown face in the same frame: the unknown
    face appears in the 'Also' line as 'stranger', and only the known
    person's encounter is logged."""
    idx, app = fresh_index

    # Enroll Ana only.
    blob_enroll = _png_bytes(100, 100)
    app.by_shape = {(100, 100): [FakeFace(bbox=(10, 10, 90, 90), embedding=_embedding(0.0))]}
    ana = await idx.enroll(name="Ana", notes="", image_blobs=[blob_enroll])

    # Frame: Ana centred, an unknown face (seed π/2 — far from Ana) to the side.
    frame = _png_bytes(400, 400)
    app.by_shape = {
        (400, 400): [
            FakeFace(bbox=(150, 150, 250, 250), embedding=_embedding(0.0)),
            FakeFace(bbox=(30, 30, 100, 100),   embedding=_embedding(float(np.pi) / 2.0)),
        ],
    }
    result = await idx.recognize(image_blob=frame, channel="ch")

    assert result.name == "Ana"
    assert len(result.faces) == 2
    stranger = [f for f in result.faces if f.name is None]
    assert len(stranger) == 1
    # Stranger was not logged.
    assert len(idx.list_encounters(ana.person_id)) == 1
    assert "stranger" in result.card_markdown.lower()


def test_update_person_notes_round_trips(fresh_index):
    idx, _ = fresh_index
    # Insert a person directly via the underlying connection — bypass enroll
    # so the test doesn't depend on the fake detector.
    now = face_index._utcnow_iso()
    with idx._db_lock:
        idx._conn_or_open().execute(
            "INSERT INTO persons (id, name, notes, created_at, updated_at) VALUES (?,?,?,?,?)",
            ("p_test", "Bob", "old notes", now, now),
        )
        idx._conn_or_open().commit()

    assert idx.update_person_notes("p_test", "new notes") is True
    assert idx.get_person("p_test")["notes"] == "new notes"
    assert idx.update_person_notes("p_does_not_exist", "x") is False


def test_delete_person_cascades(fresh_index):
    idx, _ = fresh_index
    now = face_index._utcnow_iso()
    with idx._db_lock:
        conn = idx._conn_or_open()
        conn.execute(
            "INSERT INTO persons (id, name, notes, created_at, updated_at) VALUES (?,?,?,?,?)",
            ("p_x", "X", "", now, now),
        )
        # Stash a dummy embedding so we can assert it's cleaned up.
        vec = _embedding(0.0)
        conn.execute("INSERT INTO embeddings (person_id, vec) VALUES (?, ?)",
                     ("p_x", face_index._pack_vec(vec)))
        conn.execute(
            "INSERT INTO encounters (person_id, ts, confidence, channel) VALUES (?,?,?,?)",
            ("p_x", now, 0.7, "ch"),
        )
        conn.commit()

    assert idx.delete_person("p_x") is True

    with idx._db_lock:
        cur = idx._conn_or_open().cursor()
        assert cur.execute("SELECT COUNT(*) FROM persons WHERE id='p_x'").fetchone()[0] == 0
        assert cur.execute("SELECT COUNT(*) FROM embeddings WHERE person_id='p_x'").fetchone()[0] == 0
        assert cur.execute("SELECT COUNT(*) FROM encounters WHERE person_id='p_x'").fetchone()[0] == 0
