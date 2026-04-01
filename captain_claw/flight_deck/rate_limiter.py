"""Per-user rate limiting for Flight Deck SaaS."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from typing import TYPE_CHECKING

from fastapi import HTTPException, Request, status

if TYPE_CHECKING:
    from captain_claw.flight_deck.db import FlightDeckDB

# ── Plan tier definitions ──

DEFAULT_PLAN_LIMITS: dict[str, dict] = {
    "free": {
        "max_agents": 2,
        "max_storage_mb": 500,
        "requests_per_minute": 60,
        "spawns_per_hour": 5,
    },
    "pro": {
        "max_agents": 10,
        "max_storage_mb": 5000,
        "requests_per_minute": 300,
        "spawns_per_hour": 50,
    },
    "enterprise": {
        "max_agents": 100,
        "max_storage_mb": 50000,
        "requests_per_minute": 1000,
        "spawns_per_hour": 500,
    },
}

# Runtime plan limits — starts as defaults, overwritten from DB on startup
PLAN_LIMITS: dict[str, dict] = {k: {**v} for k, v in DEFAULT_PLAN_LIMITS.items()}

PLAN_FIELDS = ("max_agents", "max_storage_mb", "requests_per_minute", "spawns_per_hour")

DEFAULT_PLAN = "free"


def load_plan_limits_from_db_sync(raw: str | None) -> None:
    """Load plan limits from a DB-stored JSON string. Called on startup."""
    if not raw:
        return
    try:
        stored = json.loads(raw)
        for plan_name, limits in stored.items():
            if plan_name in PLAN_LIMITS and isinstance(limits, dict):
                for field in PLAN_FIELDS:
                    if field in limits:
                        PLAN_LIMITS[plan_name][field] = limits[field]
    except (json.JSONDecodeError, TypeError):
        pass


def update_plan_limits(plan: str, limits: dict) -> None:
    """Update a single plan's limits in memory."""
    if plan not in PLAN_LIMITS:
        PLAN_LIMITS[plan] = {**DEFAULT_PLAN_LIMITS.get(DEFAULT_PLAN, {})}
    for field in PLAN_FIELDS:
        if field in limits:
            PLAN_LIMITS[plan][field] = limits[field]


def get_plan_limits_json() -> str:
    """Serialize current plan limits for DB storage."""
    return json.dumps(PLAN_LIMITS)


def get_user_plan(user: dict) -> str:
    """Extract plan from user metadata, default to free."""
    try:
        meta = json.loads(user.get("metadata", "{}"))
        return meta.get("plan", DEFAULT_PLAN)
    except (json.JSONDecodeError, TypeError):
        return DEFAULT_PLAN


def get_plan_limits(plan: str) -> dict:
    """Return limits for a plan tier."""
    return PLAN_LIMITS.get(plan, PLAN_LIMITS.get(DEFAULT_PLAN, {}))


def get_user_limits(user: dict) -> dict:
    """Return limits for a user, with per-user overrides from metadata."""
    plan = get_user_plan(user)
    limits = {**get_plan_limits(plan)}
    # Allow per-user overrides in metadata
    try:
        meta = json.loads(user.get("metadata", "{}"))
        for key in ("max_agents", "max_storage_mb", "requests_per_minute", "spawns_per_hour"):
            if key in meta:
                limits[key] = meta[key]
    except (json.JSONDecodeError, TypeError):
        pass
    return limits


# ── In-memory sliding window rate limiter ──

class _SlidingWindow:
    """Simple sliding-window counter per user."""

    def __init__(self):
        self._requests: dict[str, list[float]] = defaultdict(list)

    def check(self, user_id: str, max_count: int, window_seconds: float) -> bool:
        """Return True if the request is allowed, False if rate limited."""
        now = time.monotonic()
        cutoff = now - window_seconds
        entries = self._requests[user_id]
        # Prune old entries
        self._requests[user_id] = [t for t in entries if t > cutoff]
        if len(self._requests[user_id]) >= max_count:
            return False
        self._requests[user_id].append(now)
        return True

    def count(self, user_id: str, window_seconds: float) -> int:
        """Return current count in the window."""
        now = time.monotonic()
        cutoff = now - window_seconds
        entries = self._requests[user_id]
        self._requests[user_id] = [t for t in entries if t > cutoff]
        return len(self._requests[user_id])


# Global rate limiter instances
_api_limiter = _SlidingWindow()
_spawn_limiter = _SlidingWindow()


def check_api_rate_limit(user: dict) -> None:
    """Check per-minute API rate limit. Raises 429 if exceeded."""
    limits = get_user_limits(user)
    rpm = limits["requests_per_minute"]
    if not _api_limiter.check(user["id"], rpm, 60.0):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded ({rpm} requests/minute). Upgrade your plan for higher limits.",
        )


def check_spawn_rate_limit(user: dict) -> None:
    """Check per-hour agent spawn rate limit. Raises 429 if exceeded."""
    limits = get_user_limits(user)
    sph = limits["spawns_per_hour"]
    if not _spawn_limiter.check(user["id"], sph, 3600.0):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Spawn rate limit exceeded ({sph} spawns/hour). Upgrade your plan for higher limits.",
        )


async def check_agent_count_limit(user: dict, current_count: int) -> None:
    """Check if user has reached their max agent count. Raises 403 if exceeded."""
    limits = get_user_limits(user)
    max_agents = limits["max_agents"]
    if current_count >= max_agents:
        plan = get_user_plan(user)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Agent limit reached ({max_agents} agents on '{plan}' plan). Upgrade for more agents.",
        )
