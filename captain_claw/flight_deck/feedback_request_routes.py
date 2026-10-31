"""Ask a user how the deck is working for them, by email from the admin.

Users → a user → Ask for feedback: a short email asking what they use the deck
for, what works, what confuses them, what they need and what they'd improve.
The admin reads and edits the draft, and it goes from their OWN connected Gmail
or Outlook (invite_routes.send_from_admin), so a reply reaches them directly and
nothing about the answers is kept here. The list shows when someone was last
asked; a failed email changes nothing.

    GET  /fd/admin/users/{id}/feedback-request  admin  the draft + how this admin can email
    POST /fd/admin/users/{id}/feedback-request  admin  {subject, body, send_via} → send it
"""

from __future__ import annotations

import html
import json

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from captain_claw.flight_deck.admin_routes import require_admin
from captain_claw.flight_deck.auth import get_db
from captain_claw.flight_deck.auth_routes import _clean_text, _valid_email
from captain_claw.flight_deck.invite_routes import (
    _admin_name,
    _branding,
    _clean_note,
    _require_auth_deck,
    mail_options,
    send_from_admin,
)
from captain_claw.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/fd/admin/users", tags=["admin-feedback"])

SUBJECT_MAX = 200
BODY_MAX = 5000
SEND_VIA = ("google", "microsoft")


class SendFeedbackRequest(BaseModel):
    subject: str
    body: str
    send_via: str


def compose_request(*, instance_name: str, admin_name: str, user_name: str) -> tuple[str, str]:
    """(subject, plain text) of the draft the admin starts from."""
    greeting = f"Hi {user_name}," if user_name else "Hi,"
    return f"How is {instance_name} working for you?", "\n".join([
        greeting,
        "",
        f"I'd like to hear how {instance_name} is working for you. A few honest lines are plenty:",
        "",
        "- What do you mostly use it for?",
        "- What has worked well for you?",
        "- What confuses you or gets in your way?",
        "- Is there anything you need that's missing?",
        "- Any tips on how we could make it better?",
        "",
        "Just reply to this email. It comes straight to me.",
        "",
        "Thank you,",
        admin_name,
    ])


def _html(text: str) -> str:
    return (
        '<div style="font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,'
        'sans-serif;font-size:15px;line-height:1.5;color:#1a1a1a;max-width:520px;white-space:pre-wrap">'
        f"{html.escape(text)}</div>"
    )


async def _user(db, user_id: str) -> dict:
    user = await db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(404, "User not found")
    return user


@router.get("/{user_id}/feedback-request")
async def draft_feedback_request(user_id: str, admin: dict = Depends(require_admin)):
    """The email to start from, how this admin can send it and when this user
    was last asked."""
    _require_auth_deck()
    db = get_db()
    user = await _user(db, user_id)
    instance_name, _ = await _branding(db)
    subject, body = compose_request(
        instance_name=instance_name, admin_name=_admin_name(admin),
        user_name=user.get("display_name") or "",
    )
    return {"to": user["email"], "subject": subject, "body": body,
            "asked_at": user.get("feedback_asked_at") or "",
            "mail": await mail_options(db, admin["id"])}


@router.post("/{user_id}/feedback-request")
async def send_feedback_request(user_id: str, body: SendFeedbackRequest,
                                admin: dict = Depends(require_admin)):
    """Email the user from the admin's own account. An email that fails comes
    back as ``ok: false`` with the reason, and they aren't marked as asked."""
    _require_auth_deck()
    if body.send_via not in SEND_VIA:
        raise HTTPException(400, "send_via must be one of: google, microsoft")
    subject = _clean_text(body.subject, SUBJECT_MAX)
    text = _clean_note(body.body, BODY_MAX)
    if not subject or not text:
        raise HTTPException(400, "Write a subject and a message")
    db = get_db()
    user = await _user(db, user_id)
    if not _valid_email(user["email"]):
        raise HTTPException(400, "This account has no email address to write to")
    mail = await send_from_admin(db, admin, body.send_via, user["email"], subject, text, _html(text))
    if not mail["sent"]:
        log.warning("Feedback request failed", user=user_id, via=body.send_via, error=mail["error"][:200])
        return {"ok": False, "email": mail, "asked_at": user.get("feedback_asked_at") or ""}
    asked_at = await db.set_feedback_asked(user_id)
    await db.log_usage(str(admin["id"]), "feedback_requested",
                       json.dumps({"user_id": user_id, "via": body.send_via}))
    log.info("Feedback request emailed", user=user_id, via=body.send_via)
    return {"ok": True, "email": mail, "asked_at": asked_at}
