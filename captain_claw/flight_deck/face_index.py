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


# ── Result types ──────────────────────────────────────────────────────


@dataclass
class RecognizeResult:
    person_id: str | None
    name: str | None
    confidence: float
    bbox: tuple[int, int, int, int] | None  # (x, y, w, h) of detected face
    card_markdown: str  # always populated — "unknown face" card if no match


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
        """Detect the largest face in the image and look it up.

        Records an encounter row only if ``confidence >= MATCH_THRESHOLD``.
        Always returns a card — for unknown faces, a "stranger" card so the
        glasses HUD has something to show.
        """
        bbox, embedding = await asyncio.to_thread(self._embed_subject, image_blob)
        if embedding is None:
            return RecognizeResult(
                person_id=None,
                name=None,
                confidence=0.0,
                bbox=None,
                card_markdown="**No face detected.**",
            )

        with self._db_lock:
            conn = self._conn_or_open()
            cur = conn.cursor()
            # sqlite-vec KNN: distance is L2; since we use normed embeddings,
            # cosine_sim = 1 - L2_dist^2 / 2. We compute that conversion
            # inline so the threshold semantics stay readable.
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
                return RecognizeResult(
                    person_id=None,
                    name=None,
                    confidence=0.0,
                    bbox=bbox,
                    card_markdown=self._render_unknown_card(),
                )

            # Aggregate by person_id — take the best (smallest) distance per
            # person, since one person has multiple reference embeddings.
            per_person: dict[str, float] = {}
            for r in rows:
                pid = r["person_id"]
                dist = float(r["distance"])
                if pid not in per_person or dist < per_person[pid]:
                    per_person[pid] = dist

            best_pid, best_dist = min(per_person.items(), key=lambda kv: kv[1])
            # L2 → cosine for L2-normalized vectors.
            cosine = max(-1.0, min(1.0, 1.0 - (best_dist * best_dist) / 2.0))

            if cosine < AMBIGUOUS_THRESHOLD:
                return RecognizeResult(
                    person_id=None,
                    name=None,
                    confidence=cosine,
                    bbox=bbox,
                    card_markdown=self._render_unknown_card(),
                )

            person = cur.execute(
                "SELECT id, name, notes FROM persons WHERE id = ?",
                (best_pid,),
            ).fetchone()
            if not person:
                return RecognizeResult(
                    person_id=None,
                    name=None,
                    confidence=cosine,
                    bbox=bbox,
                    card_markdown=self._render_unknown_card(),
                )

            confident = cosine >= MATCH_THRESHOLD
            if confident:
                cur.execute(
                    "INSERT INTO encounters (person_id, ts, confidence, channel) VALUES (?, ?, ?, ?)",
                    (best_pid, _utcnow_iso(), cosine, channel),
                )
                conn.commit()

            card = self._render_person_card(
                person_row=person,
                confidence=cosine,
                confident=confident,
            )
            return RecognizeResult(
                person_id=best_pid if confident else None,
                name=person["name"] if confident else None,
                confidence=cosine,
                bbox=bbox,
                card_markdown=card,
            )

    def _embed_subject(self, blob: bytes):
        """Return (bbox, embedding) for the **subject** face in a recognition
        snapshot, or (None, None).

        Selection policy: the user is aiming the phone at one specific person.
        The closest-to-center face is the most honest signal; we use area as
        a soft tiebreaker so tiny background faces near the centre don't win
        over a clearly framed subject slightly off-centre.

        Score (lower is better) =
            (distance from face center to image center) / image_diagonal
            − 0.25 * sqrt(face_area / image_area)
        """
        app = _load_face_app()
        try:
            bgr = _decode_image_to_bgr(blob)
        except Exception:
            return None, None
        faces = app.get(bgr)
        if not faces:
            return None, None

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
        f = faces[0]
        x1, y1, x2, y2 = (int(v) for v in f.bbox)
        bbox = (x1, y1, x2 - x1, y2 - y1)
        return bbox, f.normed_embedding

    # ── Card rendering ──

    def _render_person_card(
        self,
        *,
        person_row: sqlite3.Row,
        confidence: float,
        confident: bool,
    ) -> str:
        pid = person_row["id"]
        name = person_row["name"]
        notes = (person_row["notes"] or "").strip()

        # Pull encounter stats (we already hold _db_lock during recognize).
        cur = self._conn_or_open().cursor()
        stats = cur.execute(
            """
            SELECT COUNT(*) AS n,
                   MAX(ts) AS last_seen
            FROM encounters WHERE person_id = ?
            """,
            (pid,),
        ).fetchone()
        n = int(stats["n"] or 0)
        last_seen_iso = stats["last_seen"]

        lines: list[str] = []
        prefix = "" if confident else "_(low confidence)_ "
        lines.append(f"{prefix}**{name}**")
        if notes:
            lines.append(notes if len(notes) <= 80 else notes[:77] + "…")
        if last_seen_iso:
            lines.append(f"_Last seen: {_humanize_ago(last_seen_iso)}_")
        if n > 0:
            lines.append(f"_Encounters: {n}_")
        lines.append(f"_match: {int(confidence * 100)}%_")
        return "\n".join(lines)

    def _render_unknown_card(self) -> str:
        return "**Unknown face**\n_Open the enrollment page to add them._"


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
