"""Face recognition infrastructure for the glasses bridge.

Self-contained: owns its own sqlite store at ``~/.captain-claw/face_index.db``
with three tables (persons, embeddings via sqlite-vec, encounters). Does NOT
write into the agent's semantic memory — recognition is an infrastructure
service that lives in Flight Deck and survives without any agent running.

Pipeline per recognize call:
  decode image → insightface detect+align+embed (512-d, ArcFace) → cosine
  ANN over sqlite-vec → top match with confidence → if conf >= threshold,
  log encounter and return person card markdown.

Heavy imports (insightface, onnxruntime, cv2, sqlite_vec) are deferred to
first use so Flight Deck startup isn't blocked when the ``faces`` extra
isn't installed. A missing extra surfaces as a clear 503 from the routes.
"""

from __future__ import annotations

import asyncio
import io
import os
import secrets
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Eagerly initialise numpy on the **main** thread, at module import time.
#
# Recognition + enrollment offload to ``asyncio.to_thread`` so insightface
# CPU inference doesn't block the event loop. The first call would otherwise
# trigger numpy's import inside a worker thread, where numpy 2.x's internal
# cross-imports (numpy → lib → matrixlib → linalg → _typing) can race the
# import lock and leave ``numpy._typing`` partially initialised. The trace
# from a fresh Python 3.12 venv looked like:
#   ImportError: cannot import name 'NDArray' from partially initialized
#   module 'numpy._typing' (most likely due to a circular import)
# Importing numpy here, on the main thread, lets it finish init once; every
# later worker-thread ``import numpy`` is then a cheap cache hit.
import numpy  # noqa: F401  — load-bearing side effect; do not remove

UTC = timezone.utc

DEFAULT_DB_PATH = Path("~/.captain-claw/face_index.db").expanduser()

# ArcFace embedding dim for the buffalo_l pack.
EMBEDDING_DIM = 512


def _env_float(name: str, default: float) -> float:
    """Read a float env var, falling back silently to ``default`` on parse error."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# Cosine-similarity thresholds. insightface buffalo_l returns L2-normalized
# embeddings, so cosine == dot product. Values tuned conservatively:
#   >= MATCH_THRESHOLD     → confident match (encounter logged)
#   >= AMBIGUOUS_THRESHOLD → low-confidence match (card shown, not logged)
#   < AMBIGUOUS_THRESHOLD  → unknown
#
# Override in production via env without a code change:
#   FACE_MATCH_THRESHOLD=0.45
#   FACE_AMBIGUOUS_THRESHOLD=0.30
MATCH_THRESHOLD = _env_float("FACE_MATCH_THRESHOLD", 0.50)
AMBIGUOUS_THRESHOLD = _env_float("FACE_AMBIGUOUS_THRESHOLD", 0.35)

# Cap on how many faces we look up per recognize call. Keeps latency bounded
# at parties; faces past this are silently dropped from the result. Override
# with FACE_MAX_FACES.
MAX_FACES_PER_FRAME = int(_env_float("FACE_MAX_FACES", 10))


# ── Result types ──────────────────────────────────────────────────────


@dataclass
class FaceMatch:
    """One detected face in a recognize call (primary or otherwise).

    Three states encoded by the combination of fields:
      - **unknown**: ``person_id is None``, ``name is None`` — either no
        embeddings in the index or cosine fell below ``AMBIGUOUS_THRESHOLD``.
      - **ambiguous**: ``person_id`` and ``name`` set, ``confident is False``
        — a candidate matched but cosine is in the
        ``AMBIGUOUS_THRESHOLD..MATCH_THRESHOLD`` band. Shown on the card
        with a "(low confidence)" prefix; no encounter logged.
      - **confident**: ``person_id`` set, ``confident is True`` — encounter
        was logged. This is the only state surfaced on the route's
        top-level ``person_id`` / ``name`` fields.
    """

    bbox: tuple[int, int, int, int]      # (x, y, w, h)
    person_id: str | None
    name: str | None
    confidence: float                    # cosine similarity, 0.0 if nothing detected
    confident: bool                      # True only when cosine ≥ MATCH_THRESHOLD


@dataclass
class RecognizeResult:
    # Primary subject (most-centred face) — kept for backwards compat with
    # callers that just want one card. None when no faces at all detected.
    person_id: str | None
    name: str | None
    confidence: float
    bbox: tuple[int, int, int, int] | None  # (x, y, w, h) of primary face
    card_markdown: str                       # rendered card (covers all faces)
    # Every face we found in the frame, ordered by centeredness (primary first).
    faces: list[FaceMatch]


@dataclass
class EnrollResult:
    person_id: str
    name: str
    embeddings_added: int


# ── Lazy heavy-dep loader ─────────────────────────────────────────────


_app_lock = threading.Lock()
_face_app: Any = None  # insightface.app.FaceAnalysis instance
_face_app_error: str | None = None


def _load_face_app() -> Any:
    """Load the insightface FaceAnalysis app on first use.

    Caches the instance. On failure, caches the error string so subsequent
    calls fail fast with the same diagnostic rather than re-attempting the
    expensive import.
    """
    global _face_app, _face_app_error
    if _face_app is not None:
        return _face_app
    if _face_app_error is not None:
        raise RuntimeError(_face_app_error)

    with _app_lock:
        if _face_app is not None:
            return _face_app
        try:
            from insightface.app import FaceAnalysis  # type: ignore[import-not-found]
        except ImportError as exc:
            _face_app_error = (
                "insightface not installed — install with 'pip install captain-claw[faces]' "
                f"({exc})"
            )
            raise RuntimeError(_face_app_error) from exc

        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        # det_size matches insightface defaults — 640x640 is a good balance
        # between detection quality and CPU latency.
        app.prepare(ctx_id=-1, det_size=(640, 640))
        _face_app = app
        return app


def _face_area(face: Any) -> float:
    """Bounding-box area for an insightface Face. Used to pick the largest
    face during enrollment."""
    x1, y1, x2, y2 = face.bbox
    return max(0.0, (x2 - x1) * (y2 - y1))


def _decode_image_to_bgr(blob: bytes):
    """Decode arbitrary image bytes into a BGR numpy array for insightface.

    insightface expects OpenCV-style BGR (not RGB). We go via Pillow so we
    accept whatever the phone sends (JPEG, HEIC-via-conversion, PNG) without
    pulling in libheif explicitly — Pillow handles most modern formats.
    """
    from PIL import Image  # already a core dep
    import numpy as np  # already a core dep

    img = Image.open(io.BytesIO(blob))
    img = img.convert("RGB")
    arr = np.array(img)
    # RGB → BGR for OpenCV/insightface convention.
    return arr[:, :, ::-1].copy()


# ── sqlite store ──────────────────────────────────────────────────────


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


def _new_person_id() -> str:
    return "p_" + secrets.token_hex(6)


def _pack_vec(vec) -> bytes:
    """Pack a float32 numpy vector for sqlite-vec storage."""
    import numpy as np

    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.shape[0] != EMBEDDING_DIM:
        raise ValueError(f"expected {EMBEDDING_DIM}-d embedding, got {arr.shape[0]}")
    return arr.tobytes()


def _unpack_vec(blob: bytes):
    import numpy as np

    return np.frombuffer(blob, dtype=np.float32)


class FaceIndex:
    """Threadsafe-ish face index. All DB ops grab ``_db_lock`` so the
    background insightface inference (which we run in a thread) doesn't
    collide with FastAPI request handlers."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = (db_path or DEFAULT_DB_PATH).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._ensure_db()

    # ── DB plumbing ──

    def _conn_or_open(self) -> sqlite3.Connection:
        if self._conn is None:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            try:
                import sqlite_vec  # type: ignore[import-not-found]

                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
                conn.enable_load_extension(False)
            except ImportError as exc:
                raise RuntimeError(
                    "sqlite-vec not installed — install with "
                    f"'pip install captain-claw[faces]' ({exc})"
                ) from exc
            self._conn = conn
        return self._conn

    def _ensure_db(self) -> None:
        with self._db_lock:
            conn = self._conn_or_open()
            cur = conn.cursor()
            cur.executescript(
                """
                CREATE TABLE IF NOT EXISTS persons (
                    id          TEXT PRIMARY KEY,
                    name        TEXT NOT NULL,
                    notes       TEXT NOT NULL DEFAULT '',
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS encounters (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id   TEXT NOT NULL,
                    ts          TEXT NOT NULL,
                    confidence  REAL NOT NULL,
                    channel     TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_encounters_person_ts
                    ON encounters(person_id, ts DESC);
                """
            )
            # sqlite-vec virtual table for the embeddings. One person can have
            # multiple rows here (3-5 reference photos at enrollment).
            cur.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS embeddings USING vec0(
                    person_id TEXT,
                    vec FLOAT[{EMBEDDING_DIM}]
                );
                """
            )
            conn.commit()

    # ── Public API ──

    def list_persons(self) -> list[dict[str, Any]]:
        with self._db_lock:
            cur = self._conn_or_open().cursor()
            rows = cur.execute(
                """
                SELECT p.id, p.name, p.notes, p.created_at, p.updated_at,
                       (SELECT COUNT(*) FROM encounters e WHERE e.person_id = p.id) AS encounters,
                       (SELECT MAX(ts) FROM encounters e WHERE e.person_id = p.id) AS last_seen,
                       (SELECT COUNT(*) FROM embeddings v WHERE v.person_id = p.id) AS samples
                FROM persons p
                ORDER BY p.name COLLATE NOCASE
                """
            ).fetchall()
            return [dict(r) for r in rows]

    def get_person(self, person_id: str) -> dict[str, Any] | None:
        with self._db_lock:
            cur = self._conn_or_open().cursor()
            row = cur.execute(
                "SELECT id, name, notes, created_at, updated_at FROM persons WHERE id = ?",
                (person_id,),
            ).fetchone()
            return dict(row) if row else None

    def delete_person(self, person_id: str) -> bool:
        with self._db_lock:
            conn = self._conn_or_open()
            cur = conn.cursor()
            cur.execute("DELETE FROM embeddings WHERE person_id = ?", (person_id,))
            cur.execute("DELETE FROM encounters WHERE person_id = ?", (person_id,))
            cur.execute("DELETE FROM persons WHERE id = ?", (person_id,))
            conn.commit()
            return cur.rowcount > 0

    def list_encounters(
        self,
        person_id: str,
        *,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Return up to ``limit`` recent encounters for one person, newest first."""
        with self._db_lock:
            cur = self._conn_or_open().cursor()
            rows = cur.execute(
                """
                SELECT id, ts, confidence, channel
                FROM encounters
                WHERE person_id = ?
                ORDER BY ts DESC
                LIMIT ?
                """,
                (person_id, int(max(1, limit))),
            ).fetchall()
            return [dict(r) for r in rows]

    def update_person_notes(self, person_id: str, notes: str) -> bool:
        with self._db_lock:
            conn = self._conn_or_open()
            cur = conn.cursor()
            cur.execute(
                "UPDATE persons SET notes = ?, updated_at = ? WHERE id = ?",
                (notes, _utcnow_iso(), person_id),
            )
            conn.commit()
            return cur.rowcount > 0

    # ── Enrollment ──

    async def enroll(
        self,
        *,
        name: str,
        notes: str,
        image_blobs: list[bytes],
        person_id: str | None = None,
    ) -> EnrollResult:
        """Embed each image and store under a (new or existing) person.

        At least one image must yield a detectable face. Images with no
        detectable face are silently skipped — the caller can compare
        ``embeddings_added`` against the number submitted to surface this.

        Inference runs in a thread because insightface is sync CPU work and
        we don't want to block the event loop.
        """
        if not name.strip():
            raise ValueError("name required")
        if not image_blobs:
            raise ValueError("at least one image required")

        embeddings = await asyncio.to_thread(self._embed_many, image_blobs)
        if not embeddings:
            raise ValueError("no faces detected in any of the submitted images")

        now = _utcnow_iso()
        with self._db_lock:
            conn = self._conn_or_open()
            cur = conn.cursor()
            if person_id:
                row = cur.execute(
                    "SELECT id FROM persons WHERE id = ?", (person_id,)
                ).fetchone()
                if not row:
                    raise ValueError(f"person_id {person_id} not found")
                cur.execute(
                    "UPDATE persons SET name = ?, notes = ?, updated_at = ? WHERE id = ?",
                    (name.strip(), notes.strip(), now, person_id),
                )
                pid = person_id
            else:
                pid = _new_person_id()
                cur.execute(
                    """
                    INSERT INTO persons (id, name, notes, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (pid, name.strip(), notes.strip(), now, now),
                )

            for vec in embeddings:
                cur.execute(
                    "INSERT INTO embeddings (person_id, vec) VALUES (?, ?)",
                    (pid, _pack_vec(vec)),
                )
            conn.commit()

        return EnrollResult(person_id=pid, name=name.strip(), embeddings_added=len(embeddings))

    def _embed_many(self, blobs: list[bytes]) -> list[Any]:
        """Sync embed helper for enrollment. Picks the **largest** face per
        image — at enrollment time the user takes deliberate portraits, so
        size is the right signal for "the subject"."""
        app = _load_face_app()
        out: list[Any] = []
        for blob in blobs:
            try:
                bgr = _decode_image_to_bgr(blob)
            except Exception:
                continue
            faces = app.get(bgr)
            if not faces:
                continue
            faces.sort(key=_face_area, reverse=True)
            out.append(faces[0].normed_embedding)
        return out

    # ── Recognition ──

    async def recognize(self, *, image_blob: bytes, channel: str = "") -> RecognizeResult:
        """Detect every face in the image and look each one up.

        The face closest to the image centre is the **primary subject**
        (full card on the HUD). Other detected faces appear as a compact
        roster line below the primary.

        Logs one encounter row per **confident** match (``cos ≥
        MATCH_THRESHOLD``) — a group photo where three friends are matched
        logs three encounters. Ambiguous matches show on the card but are
        not logged, so drift doesn't pollute the history.
        """
        detections = await asyncio.to_thread(self._detect_all, image_blob)
        if not detections:
            return RecognizeResult(
                person_id=None,
                name=None,
                confidence=0.0,
                bbox=None,
                card_markdown="**No face detected.**",
                faces=[],
            )

        matches: list[FaceMatch] = []
        with self._db_lock:
            conn = self._conn_or_open()
            cur = conn.cursor()
            now = _utcnow_iso()
            for bbox, embedding in detections:
                pid, cosine = self._lookup_embedding(cur, embedding)
                name: str | None = None
                if pid is not None:
                    row = cur.execute(
                        "SELECT name FROM persons WHERE id = ?", (pid,)
                    ).fetchone()
                    if row:
                        name = row["name"]
                confident = pid is not None and cosine >= MATCH_THRESHOLD
                if confident:
                    cur.execute(
                        "INSERT INTO encounters (person_id, ts, confidence, channel) VALUES (?, ?, ?, ?)",
                        (pid, now, cosine, channel),
                    )
                matches.append(FaceMatch(
                    bbox=bbox,
                    person_id=pid,
                    name=name,
                    confidence=cosine,
                    confident=confident,
                ))
            conn.commit()

            # Card composition reads back from the DB for the primary's
            # notes / encounter count, so we render it while still holding
            # the lock — same connection, no extra contention.
            card_markdown = self._render_multi_card(matches, cur)

        # Primary is the first detection (sorted most-centred-first).
        # Top-level person_id/name surface ONLY the confident match — this
        # preserves the API contract callers rely on for "I confirmed it
        # was Ana" semantics. Low-confidence matches show on the card but
        # leave the top-level fields ``None``.
        primary = matches[0]
        return RecognizeResult(
            person_id=primary.person_id if primary.confident else None,
            name=primary.name if primary.confident else None,
            confidence=primary.confidence,
            bbox=primary.bbox,
            card_markdown=card_markdown,
            faces=matches,
        )

    def _lookup_embedding(self, cur, embedding) -> tuple[str | None, float]:
        """Resolve one embedding against the index.

        Returns ``(person_id, cosine)``. ``person_id`` is ``None`` when the
        best cosine falls below ``AMBIGUOUS_THRESHOLD`` (or the index is
        empty). Pulls the top-5 nearest vectors via sqlite-vec, then
        collapses to one row per person by keeping each person's best
        distance — one person typically has several reference embeddings.
        """
        rows = cur.execute(
            """
            SELECT person_id, distance
            FROM embeddings
            WHERE vec MATCH ?
            ORDER BY distance
            LIMIT 5
            """,
            (_pack_vec(embedding),),
        ).fetchall()
        if not rows:
            return None, 0.0
        per_person: dict[str, float] = {}
        for r in rows:
            pid = r["person_id"]
            dist = float(r["distance"])
            if pid not in per_person or dist < per_person[pid]:
                per_person[pid] = dist
        best_pid, best_dist = min(per_person.items(), key=lambda kv: kv[1])
        # L2 → cosine for L2-normalized vectors:  cos = 1 − d² / 2
        cosine = max(-1.0, min(1.0, 1.0 - (best_dist * best_dist) / 2.0))
        if cosine < AMBIGUOUS_THRESHOLD:
            return None, cosine
        return best_pid, cosine

    def _detect_all(self, blob: bytes):
        """Detect every face in the image and return them sorted with the
        most-centred face first.

        Returns ``[(bbox, embedding), …]`` where ``bbox`` is ``(x, y, w, h)``
        in image pixels. Empty list when no faces detected. Capped at
        ``MAX_FACES_PER_FRAME`` so a crowd photo can't blow the latency
        budget (the cap drops faces past it, never raises).

        Centeredness scoring (lower = better, primary first):
            (centre-to-centre distance / image diagonal)
            − 0.25 * sqrt(face_area / image_area)
        """
        app = _load_face_app()
        try:
            bgr = _decode_image_to_bgr(blob)
        except Exception:
            return []
        faces = app.get(bgr)
        if not faces:
            return []

        import math

        h, w = bgr.shape[:2]
        img_diag = math.hypot(w, h) or 1.0
        img_area = float(w * h) or 1.0
        cx, cy = w / 2.0, h / 2.0

        def score(f) -> float:
            x1, y1, x2, y2 = f.bbox
            fx = (x1 + x2) / 2.0
            fy = (y1 + y2) / 2.0
            center_dist = math.hypot(fx - cx, fy - cy) / img_diag
            area_frac = max(0.0, (x2 - x1) * (y2 - y1)) / img_area
            return center_dist - 0.25 * math.sqrt(area_frac)

        faces.sort(key=score)
        out: list[tuple[tuple[int, int, int, int], Any]] = []
        for f in faces[:MAX_FACES_PER_FRAME]:
            x1, y1, x2, y2 = (int(v) for v in f.bbox)
            bbox = (x1, y1, x2 - x1, y2 - y1)
            out.append((bbox, f.normed_embedding))
        return out

    # ── Card rendering ──

    def _render_multi_card(self, matches: list[FaceMatch], cur) -> str:
        """Compose the HUD markdown for a recognize call.

        Layout:
          [primary face: full block — name, notes, last seen, count, match%]
          [if any extra faces: a single roster line — "Also: Bob 72% · stranger"]
        """
        primary = matches[0]
        lines: list[str] = self._render_primary_block(primary, cur)

        if len(matches) > 1:
            extras = []
            for m in matches[1:]:
                pct = max(0, int(m.confidence * 100))
                if m.name:
                    suffix = "" if m.confident else " low"
                    extras.append(f"{m.name} {pct}%{suffix}")
                elif m.confidence > 0:
                    extras.append(f"stranger {pct}%")
                else:
                    extras.append("stranger")
            if extras:
                lines.append("")  # blank line separator
                lines.append(f"_Also: {' · '.join(extras)}_")
        return "\n".join(lines)

    def _render_primary_block(self, m: FaceMatch, cur) -> list[str]:
        """Render the primary subject's lines. Reused inside _render_multi_card."""
        # No matched person → unknown card. ``confidence > 0`` here means we
        # detected a face but every candidate fell below AMBIGUOUS_THRESHOLD;
        # we show the cosine so the user can judge if they should retake.
        if m.person_id is None:
            if m.confidence > 0:
                return [
                    "**Unknown face**",
                    f"_match: {int(m.confidence * 100)}% (below threshold)_",
                ]
            return [
                "**Unknown face**",
                "_Open the enrollment page to add them._",
            ]

        # Confident or ambiguous — both render the full block; only the
        # prefix differs.
        row = cur.execute(
            "SELECT notes FROM persons WHERE id = ?",
            (m.person_id,),
        ).fetchone()
        notes = ((row["notes"] if row else "") or "").strip()

        stats = cur.execute(
            "SELECT COUNT(*) AS n, MAX(ts) AS last_seen FROM encounters WHERE person_id = ?",
            (m.person_id,),
        ).fetchone()
        n = int(stats["n"] or 0)
        last_seen_iso = stats["last_seen"]

        out: list[str] = []
        prefix = "" if m.confident else "_(low confidence)_ "
        out.append(f"{prefix}**{m.name}**")
        if notes:
            out.append(notes if len(notes) <= 80 else notes[:77] + "…")
        if last_seen_iso:
            out.append(f"_Last seen: {_humanize_ago(last_seen_iso)}_")
        if n > 0:
            out.append(f"_Encounters: {n}_")
        out.append(f"_match: {int(m.confidence * 100)}%_")
        return out


def _humanize_ago(iso_ts: str) -> str:
    """Render an ISO timestamp as a short relative string (e.g. '5m ago')."""
    try:
        when = datetime.fromisoformat(iso_ts)
    except Exception:
        return iso_ts
    now = datetime.now(UTC)
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    delta = now - when
    secs = int(delta.total_seconds())
    if secs < 60:
        return "just now"
    if secs < 3600:
        return f"{secs // 60}m ago"
    if secs < 86400:
        return f"{secs // 3600}h ago"
    days = secs // 86400
    if days < 30:
        return f"{days}d ago"
    months = days // 30
    if months < 12:
        return f"{months}mo ago"
    return f"{days // 365}y ago"


# ── Module-level singleton ────────────────────────────────────────────


_INDEX: FaceIndex | None = None
_INDEX_LOCK = threading.Lock()


def get_index() -> FaceIndex:
    """Process-wide singleton. Created on first call."""
    global _INDEX
    if _INDEX is not None:
        return _INDEX
    with _INDEX_LOCK:
        if _INDEX is None:
            _INDEX = FaceIndex()
        return _INDEX


async def warmup() -> dict[str, Any]:
    """Force the heavy insightface load so the first /face/recognize doesn't
    pay a ~1-minute weight-download tax in the middle of a user request."""
    await asyncio.to_thread(_load_face_app)
    return {"ok": True, "ts": _utcnow_iso()}
