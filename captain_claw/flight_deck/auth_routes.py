"""Authentication REST endpoints for Flight Deck."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Response, Request, status
from pydantic import BaseModel, EmailStr

from captain_claw.flight_deck.auth import (
    REFRESH_COOKIE,
    REFRESH_TOKEN_TTL,
    create_access_token,
    create_refresh_token,
    get_current_user,
    get_db,
    hash_password,
    hash_token,
    verify_password,
)

router = APIRouter(prefix="/fd/auth", tags=["auth"])


# ── Request / response models ───────────────────────────────────────

class RegisterRequest(BaseModel):
    email: str
    password: str
    display_name: str = ""


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


class UserResponse(BaseModel):
    id: str
    email: str
    display_name: str
    role: str


class UpdateProfileRequest(BaseModel):
    display_name: str | None = None
    password: str | None = None
    current_password: str | None = None


# ── Helpers ──────────────────────────────────────────────────────────

def _set_refresh_cookie(response: Response, refresh_token: str) -> None:
    response.set_cookie(
        key=REFRESH_COOKIE,
        value=refresh_token,
        max_age=int(REFRESH_TOKEN_TTL.total_seconds()),
        httponly=True,
        samesite="lax",
        path="/fd/auth",
        secure=False,  # Set True behind TLS reverse proxy
    )


def _clear_refresh_cookie(response: Response) -> None:
    response.delete_cookie(key=REFRESH_COOKIE, path="/fd/auth")


# ── Endpoints ────────────────────────────────────────────────────────

@router.post("/register", response_model=TokenResponse)
async def register(body: RegisterRequest, response: Response):
    db = get_db()
    if not body.email or not body.password:
        raise HTTPException(status_code=400, detail="Email and password required")
    if len(body.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    existing = await db.get_user_by_email(body.email)
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")

    pw_hash = hash_password(body.password)
    display = body.display_name or body.email.split("@")[0]

    # First user becomes admin
    user_count = await db.count_users()
    role = "admin" if user_count == 0 else "user"

    user = await db.create_user(
        email=body.email, password_hash=pw_hash,
        display_name=display, role=role,
    )

    access_token = create_access_token(user["id"], role=role)
    refresh_token = create_refresh_token()

    expires_at = (datetime.now(timezone.utc) + REFRESH_TOKEN_TTL).isoformat()
    await db.create_refresh_session(user["id"], hash_token(refresh_token), expires_at)
    _set_refresh_cookie(response, refresh_token)

    return TokenResponse(
        access_token=access_token,
        user={"id": user["id"], "email": user["email"],
              "display_name": display, "role": role},
    )


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, response: Response):
    db = get_db()
    user = await db.get_user_by_email(body.email)
    if not user or not verify_password(body.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token = create_access_token(user["id"], role=user["role"])
    refresh_token = create_refresh_token()

    expires_at = (datetime.now(timezone.utc) + REFRESH_TOKEN_TTL).isoformat()
    await db.create_refresh_session(user["id"], hash_token(refresh_token), expires_at)
    _set_refresh_cookie(response, refresh_token)

    return TokenResponse(
        access_token=access_token,
        user={"id": user["id"], "email": user["email"],
              "display_name": user["display_name"], "role": user["role"]},
    )


@router.post("/refresh")
async def refresh(request: Request, response: Response):
    refresh_token = request.cookies.get(REFRESH_COOKIE)
    if not refresh_token:
        raise HTTPException(status_code=401, detail="No refresh token")

    db = get_db()
    token_hash = hash_token(refresh_token)

    # Find the session matching this refresh token
    # We need to search by hash since we don't store the session_id in the cookie
    assert db._db is not None
    async with db._db.execute(
        "SELECT * FROM user_sessions WHERE refresh_token_hash = ?", (token_hash,)
    ) as cur:
        session = await cur.fetchone()

    if not session:
        _clear_refresh_cookie(response)
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    session = dict(session)
    now = datetime.now(timezone.utc)
    expires = datetime.fromisoformat(session["expires_at"])
    if now > expires:
        await db.delete_refresh_session(session["id"])
        _clear_refresh_cookie(response)
        raise HTTPException(status_code=401, detail="Refresh token expired")

    user = await db.get_user_by_id(session["user_id"])
    if not user:
        await db.delete_refresh_session(session["id"])
        _clear_refresh_cookie(response)
        raise HTTPException(status_code=401, detail="User not found")

    # Rotate: delete old session, create new tokens
    await db.delete_refresh_session(session["id"])
    new_access = create_access_token(user["id"], role=user["role"])
    new_refresh = create_refresh_token()
    new_expires = (now + REFRESH_TOKEN_TTL).isoformat()
    await db.create_refresh_session(user["id"], hash_token(new_refresh), new_expires)
    _set_refresh_cookie(response, new_refresh)

    return {
        "access_token": new_access,
        "token_type": "bearer",
        "user": {"id": user["id"], "email": user["email"],
                 "display_name": user["display_name"], "role": user["role"]},
    }


@router.post("/logout")
async def logout(request: Request, response: Response):
    refresh_token = request.cookies.get(REFRESH_COOKIE)
    if refresh_token:
        db = get_db()
        token_hash = hash_token(refresh_token)
        assert db._db is not None
        await db._db.execute(
            "DELETE FROM user_sessions WHERE refresh_token_hash = ?", (token_hash,)
        )
        await db._db.commit()
    _clear_refresh_cookie(response)
    return {"ok": True}


@router.get("/me")
async def get_me(user: dict = Depends(get_current_user)):
    return {
        "id": user["id"], "email": user["email"],
        "display_name": user["display_name"], "role": user["role"],
    }


@router.put("/me")
async def update_me(body: UpdateProfileRequest, user: dict = Depends(get_current_user)):
    db = get_db()
    updates: dict = {}

    if body.display_name is not None:
        updates["display_name"] = body.display_name

    if body.password is not None:
        if not body.current_password:
            raise HTTPException(status_code=400, detail="Current password required")
        full_user = await db.get_user_by_email(user["email"])
        if not full_user or not verify_password(body.current_password, full_user["password_hash"]):
            raise HTTPException(status_code=400, detail="Current password is incorrect")
        if len(body.password) < 6:
            raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
        updates["password_hash"] = hash_password(body.password)

    if updates:
        await db.update_user(user["id"], **updates)

    updated = await db.get_user_by_id(user["id"])
    return {
        "id": updated["id"], "email": updated["email"],
        "display_name": updated["display_name"], "role": updated["role"],
    }
