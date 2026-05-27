"""HTTP routes for the face recognition infrastructure tool.

These routes are not LLM tools — they're called directly by the mobile PWA
(``glasses_mobile.html`` / ``glasses_enroll.html``). On a confident
recognition match, the route pushes a person-card payload onto the same
in-memory channel bus that ``glasses_bridge.py`` uses, so the glasses HUD
view receives it like any other agent message.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from captain_claw.flight_deck import face_index
from captain_claw.flight_deck.glasses_bridge import (
    _broadcast,
    _check_token,
    _get_or_create_channel,
)

UTC = timezone.utc

router = APIRouter()

_STATIC_DIR = Path(__file__).resolve().parent / "static"

_NO_CACHE = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


# ── Enrollment UI page ────────────────────────────────────────────────


@router.get("/glasses/enroll", response_class=HTMLResponse)
async def face_enroll_page(request: Request) -> HTMLResponse:
    """Enrollment UI — capture 3–5 photos of a new person and POST them.

    Shares the no-cache headers / shape of the other glasses pages.
    """
    _check_token(request)
    path = _STATIC_DIR / "glasses_enroll.html"
    html = path.read_text(encoding="utf-8")
    return HTMLResponse(content=html, headers=_NO_CACHE)


# ── Admin / listing ───────────────────────────────────────────────────


@router.get("/face/persons")
async def list_persons(request: Request) -> JSONResponse:
    _check_token(request)
    try:
        rows = face_index.get_index().list_persons()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return JSONResponse(rows, headers=_NO_CACHE)


@router.get("/face/persons/{person_id}")
async def get_person(person_id: str, request: Request) -> JSONResponse:
    _check_token(request)
    try:
        idx = face_index.get_index()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    person = idx.get_person(person_id)
    if not person:
        raise HTTPException(status_code=404, detail="person not found")
    return JSONResponse(person, headers=_NO_CACHE)


@router.get("/face/persons/{person_id}/encounters")
async def list_encounters(
    person_id: str,
    request: Request,
    limit: int = 200,
) -> JSONResponse:
    _check_token(request)
    try:
        idx = face_index.get_index()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if not idx.get_person(person_id):
        raise HTTPException(status_code=404, detail="person not found")
    rows = idx.list_encounters(person_id, limit=limit)
    return JSONResponse(rows, headers=_NO_CACHE)


@router.get("/glasses/persons/{person_id}", response_class=HTMLResponse)
async def person_detail_page(person_id: str, request: Request) -> HTMLResponse:
    """Server-renders ``glasses_person.html`` as-is; the page fetches its own
    data via the JSON endpoints above. Existence check happens client-side."""
    _check_token(request)
    path = _STATIC_DIR / "glasses_person.html"
    html = path.read_text(encoding="utf-8")
    return HTMLResponse(content=html, headers=_NO_CACHE)


@router.delete("/face/persons/{person_id}")
async def delete_person(person_id: str, request: Request) -> JSONResponse:
    _check_token(request)
    try:
        ok = face_index.get_index().delete_person(person_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if not ok:
        raise HTTPException(status_code=404, detail="person not found")
    return JSONResponse({"ok": True}, headers=_NO_CACHE)


@router.patch("/face/persons/{person_id}")
async def update_person(person_id: str, request: Request) -> JSONResponse:
    _check_token(request)
    body = await request.json()
    notes = str(body.get("notes", ""))
    try:
        ok = face_index.get_index().update_person_notes(person_id, notes)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if not ok:
        raise HTTPException(status_code=404, detail="person not found")
    return JSONResponse({"ok": True}, headers=_NO_CACHE)


# ── Warmup ────────────────────────────────────────────────────────────


@router.post("/face/warmup")
async def warmup(request: Request) -> JSONResponse:
    """Force the insightface model to load. First call after a fresh
    install will download ~280 MB of weights and may take a minute. Call
    this once at startup or on the enrollment page to surface the wait."""
    _check_token(request)
    try:
        result = await face_index.warmup()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return JSONResponse(result, headers=_NO_CACHE)


# ── Enrollment ────────────────────────────────────────────────────────


@router.post("/face/enroll")
async def enroll(
    request: Request,
    name: str = Form(...),
    notes: str = Form(""),
    person_id: str = Form(""),
    files: list[UploadFile] = File(...),
) -> JSONResponse:
    """Enroll (or update) a person with 1+ reference photos.

    Pass ``person_id`` to append photos to an existing person instead of
    creating a new one — useful for the silent re-enrollment path.
    """
    _check_token(request)
    if not files:
        raise HTTPException(status_code=400, detail="at least one file required")
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="max 10 photos per enroll call")

    blobs: list[bytes] = []
    for f in files:
        data = await f.read()
        if data:
            blobs.append(data)

    if not blobs:
        raise HTTPException(status_code=400, detail="all files were empty")

    try:
        result = await face_index.get_index().enroll(
            name=name,
            notes=notes,
            image_blobs=blobs,
            person_id=person_id or None,
        )
    except RuntimeError as exc:
        # Heavy-dep missing.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return JSONResponse(
        {
            "person_id": result.person_id,
            "name": result.name,
            "embeddings_added": result.embeddings_added,
            "submitted": len(blobs),
        },
        headers=_NO_CACHE,
    )


# ── Recognition ───────────────────────────────────────────────────────


@router.post("/face/recognize")
async def recognize(
    request: Request,
    channel: str = Form(""),
    file: UploadFile = File(...),
) -> JSONResponse:
    """Recognize one face from a single photo and (optionally) push a card.

    When ``channel`` is supplied, the rendered person card is broadcast to
    the channel bus exactly like an agent message — the glasses HUD picks
    it up automatically. Phone still gets the result in the HTTP response.
    """
    _check_token(request)
    blob = await file.read()
    if not blob:
        raise HTTPException(status_code=400, detail="empty file")

    try:
        result = await face_index.get_index().recognize(image_blob=blob, channel=channel)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # Broadcast the card to the glasses HUD if a channel was specified.
    # We use ``type: "agent"`` so the existing glasses_view rendering path
    # treats it like an agent reply (markdown + TTS-eligible). The
    # ``source: "face_index"`` field lets future client code distinguish
    # face cards from real agent replies if it wants to.
    if channel and result.card_markdown:
        ch = await _get_or_create_channel(channel)
        await _broadcast(
            ch,
            {
                "type": "agent",
                "text": result.card_markdown,
                "source": "face_index",
                "person_id": result.person_id,
                "confidence": result.confidence,
                "ts": _now_iso(),
            },
        )

    return JSONResponse(
        {
            "person_id": result.person_id,
            "name": result.name,
            "confidence": result.confidence,
            "bbox": list(result.bbox) if result.bbox else None,
            "card_markdown": result.card_markdown,
            "faces": [
                {
                    "person_id": f.person_id,
                    "name": f.name,
                    "confidence": f.confidence,
                    "bbox": list(f.bbox),
                }
                for f in result.faces
            ],
        },
        headers=_NO_CACHE,
    )
