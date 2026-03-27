"""Twitter/X API v2 bridge (OAuth 2.0 User Context).

Provides polling for mentions and DMs, posting tweets, searching,
liking, retweeting, and media upload.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from captain_claw.logging import get_logger

log = get_logger(__name__)

# Twitter API v2 tweet character limit.
TWEET_MAX_CHARS = 280


@dataclass
class TwitterMessage:
    """Normalized incoming tweet or mention."""

    id: str
    text: str
    author_id: str
    author_username: str = ""
    in_reply_to_tweet_id: str = ""
    conversation_id: str = ""
    created_at: str = ""
    is_dm: bool = False


@dataclass
class TwitterDM:
    """Normalized incoming DM event."""

    id: str
    text: str
    sender_id: str
    dm_conversation_id: str = ""
    created_at: str = ""


class TwitterBridge:
    """Twitter/X API v2 helper with OAuth 2.0 User Context."""

    def __init__(
        self,
        *,
        client_id: str = "",
        client_secret: str = "",
        access_token: str = "",
        refresh_token: str = "",
        api_base_url: str = "https://api.twitter.com/2",
        upload_base_url: str = "https://upload.twitter.com/1.1",
        oauth2_token_url: str = "https://api.twitter.com/2/oauth2/token",
    ) -> None:
        self.client_id = str(client_id or "").strip()
        self.client_secret = str(client_secret or "").strip()
        self.access_token = str(access_token or "").strip()
        self.refresh_token = str(refresh_token or "").strip()
        self.api_base_url = (api_base_url or "https://api.twitter.com/2").rstrip("/")
        self.upload_base_url = (upload_base_url or "https://upload.twitter.com/1.1").rstrip("/")
        self.oauth2_token_url = oauth2_token_url or "https://api.twitter.com/2/oauth2/token"
        self._client = httpx.AsyncClient(timeout=40.0)
        self._me_id: str = ""
        self._me_username: str = ""
        self._token_expires_at: float = 0.0
        # Simple rate-limit tracking: {endpoint: (remaining, reset_epoch)}
        self._rate_limits: dict[str, tuple[int, float]] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.access_token)

    async def close(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Auth helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }

    def _upload_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.access_token}"}

    async def _maybe_refresh_token(self) -> None:
        """Refresh the access token if it's expired or about to expire."""
        if not self.refresh_token or not self.client_id:
            return
        # Refresh if within 60s of expiry (or if we've never set expiry).
        if self._token_expires_at and time.time() < self._token_expires_at - 60:
            return
        try:
            response = await self._client.post(
                self.oauth2_token_url,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": self.refresh_token,
                    "client_id": self.client_id,
                },
                auth=(self.client_id, self.client_secret) if self.client_secret else None,
            )
            response.raise_for_status()
            data = response.json()
            self.access_token = str(data.get("access_token", "")).strip()
            new_refresh = str(data.get("refresh_token", "")).strip()
            if new_refresh:
                self.refresh_token = new_refresh
            expires_in = int(data.get("expires_in", 7200))
            self._token_expires_at = time.time() + expires_in
            log.info("Twitter access token refreshed", expires_in=expires_in)
        except Exception as e:
            log.error("Twitter token refresh failed", error=str(e))

    def _track_rate_limit(self, endpoint: str, headers: httpx.Headers) -> None:
        """Track rate-limit headers from a response."""
        remaining = headers.get("x-rate-limit-remaining")
        reset = headers.get("x-rate-limit-reset")
        if remaining is not None and reset is not None:
            try:
                self._rate_limits[endpoint] = (int(remaining), float(reset))
            except (ValueError, TypeError):
                pass

    def _check_rate_limit(self, endpoint: str) -> bool:
        """Return True if we should proceed, False if rate-limited."""
        info = self._rate_limits.get(endpoint)
        if info is None:
            return True
        remaining, reset_epoch = info
        if remaining <= 1 and time.time() < reset_epoch:
            return False
        return True

    # ------------------------------------------------------------------
    # Core API methods
    # ------------------------------------------------------------------

    async def _api_get(
        self, path: str, params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        await self._maybe_refresh_token()
        url = f"{self.api_base_url}/{path.lstrip('/')}"
        response = await self._client.get(url, headers=self._headers(), params=params or {})
        self._track_rate_limit(path, response.headers)
        response.raise_for_status()
        return response.json()

    async def _api_post(
        self, path: str, json_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        await self._maybe_refresh_token()
        url = f"{self.api_base_url}/{path.lstrip('/')}"
        response = await self._client.post(url, headers=self._headers(), json=json_payload)
        self._track_rate_limit(path, response.headers)
        response.raise_for_status()
        return response.json()

    async def _api_delete(self, path: str) -> dict[str, Any]:
        await self._maybe_refresh_token()
        url = f"{self.api_base_url}/{path.lstrip('/')}"
        response = await self._client.delete(url, headers=self._headers())
        self._track_rate_limit(path, response.headers)
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    async def get_me(self) -> tuple[str, str]:
        """Return (user_id, username) for the authenticated user. Cached."""
        if self._me_id:
            return self._me_id, self._me_username
        data = await self._api_get("users/me", {"user.fields": "username"})
        user = data.get("data", {})
        self._me_id = str(user.get("id", "")).strip()
        self._me_username = str(user.get("username", "")).strip()
        return self._me_id, self._me_username

    # ------------------------------------------------------------------
    # Polling (mentions + DMs)
    # ------------------------------------------------------------------

    async def get_mentions(
        self, *, since_id: str = "",
    ) -> list[TwitterMessage]:
        """Fetch recent mentions of the authenticated user."""
        me_id, _ = await self.get_me()
        if not me_id:
            return []
        params: dict[str, Any] = {
            "tweet.fields": "author_id,conversation_id,created_at,in_reply_to_user_id,referenced_tweets",
            "expansions": "author_id",
            "user.fields": "username",
            "max_results": 100,
        }
        if since_id:
            params["since_id"] = since_id
        try:
            data = await self._api_get(f"users/{me_id}/mentions", params)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                log.warning("Twitter mentions rate-limited")
                return []
            raise
        tweets = data.get("data") or []
        # Build author lookup from includes.
        includes = data.get("includes", {})
        users = {
            str(u.get("id", "")): str(u.get("username", ""))
            for u in (includes.get("users") or [])
            if isinstance(u, dict)
        }
        messages: list[TwitterMessage] = []
        for tweet in tweets:
            if not isinstance(tweet, dict):
                continue
            author_id = str(tweet.get("author_id", "")).strip()
            # Skip our own tweets.
            if author_id == me_id:
                continue
            in_reply_to = ""
            refs = tweet.get("referenced_tweets") or []
            for ref in refs:
                if isinstance(ref, dict) and ref.get("type") == "replied_to":
                    in_reply_to = str(ref.get("id", "")).strip()
                    break
            messages.append(
                TwitterMessage(
                    id=str(tweet.get("id", "")).strip(),
                    text=str(tweet.get("text", "")).strip(),
                    author_id=author_id,
                    author_username=users.get(author_id, ""),
                    in_reply_to_tweet_id=in_reply_to,
                    conversation_id=str(tweet.get("conversation_id", "")).strip(),
                    created_at=str(tweet.get("created_at", "")).strip(),
                )
            )
        return messages

    async def get_dms(
        self, *, since_id: str = "",
    ) -> list[TwitterDM]:
        """Fetch recent DM events for the authenticated user."""
        params: dict[str, Any] = {
            "dm_event.fields": "id,text,sender_id,dm_conversation_id,created_at",
            "max_results": 100,
        }
        if since_id:
            params["since_id"] = since_id  # Not standard; pagination_token preferred.
        me_id, _ = await self.get_me()
        try:
            data = await self._api_get("dm_events", params)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                log.warning("Twitter DMs rate-limited")
                return []
            raise
        events = data.get("data") or []
        dms: list[TwitterDM] = []
        for event in events:
            if not isinstance(event, dict):
                continue
            sender_id = str(event.get("sender_id", "")).strip()
            # Skip messages we sent.
            if sender_id == me_id:
                continue
            dms.append(
                TwitterDM(
                    id=str(event.get("id", "")).strip(),
                    text=str(event.get("text", "")).strip(),
                    sender_id=sender_id,
                    dm_conversation_id=str(event.get("dm_conversation_id", "")).strip(),
                    created_at=str(event.get("created_at", "")).strip(),
                )
            )
        return dms

    async def get_updates(
        self, offsets: dict[str, str] | None = None,
    ) -> tuple[list[TwitterMessage | TwitterDM], dict[str, str]]:
        """Unified polling entry point: fetch mentions + DMs.

        Returns (updates, next_offsets).
        """
        current = dict(offsets or {})
        next_offsets = dict(current)
        all_updates: list[TwitterMessage | TwitterDM] = []

        # Mentions
        mentions_since = current.get("mentions_since_id", "")
        if self._check_rate_limit("mentions"):
            mentions = await self.get_mentions(since_id=mentions_since)
            if mentions:
                all_updates.extend(mentions)
                # Track highest ID for next poll.
                max_id = max(mentions, key=lambda m: int(m.id))
                next_offsets["mentions_since_id"] = max_id.id

        # DMs
        dm_since = current.get("dm_since_id", "")
        if self._check_rate_limit("dms"):
            dms = await self.get_dms(since_id=dm_since)
            if dms:
                all_updates.extend(dms)
                max_dm = max(dms, key=lambda d: int(d.id))
                next_offsets["dm_since_id"] = max_dm.id

        return all_updates, next_offsets

    # ------------------------------------------------------------------
    # Posting
    # ------------------------------------------------------------------

    async def post_tweet(
        self,
        text: str,
        *,
        reply_to_id: str = "",
        media_ids: list[str] | None = None,
        quote_tweet_id: str = "",
    ) -> str:
        """Post a tweet. Returns the new tweet ID."""
        payload: dict[str, Any] = {"text": str(text or "").strip()[:TWEET_MAX_CHARS]}
        if reply_to_id:
            payload["reply"] = {"in_reply_to_tweet_id": str(reply_to_id).strip()}
        if media_ids:
            payload["media"] = {"media_ids": [str(m).strip() for m in media_ids]}
        if quote_tweet_id:
            payload["quote_tweet_id"] = str(quote_tweet_id).strip()
        data = await self._api_post("tweets", payload)
        tweet_data = data.get("data", {})
        return str(tweet_data.get("id", "")).strip()

    async def post_thread(self, texts: list[str]) -> list[str]:
        """Post a series of tweets as a thread. Returns list of tweet IDs."""
        if not texts:
            return []
        ids: list[str] = []
        prev_id = ""
        for text in texts:
            tweet_id = await self.post_tweet(text, reply_to_id=prev_id)
            ids.append(tweet_id)
            prev_id = tweet_id
        return ids

    async def send_dm(self, participant_id: str, text: str) -> str:
        """Send a DM to a user. Returns the DM event ID."""
        payload: dict[str, Any] = {
            "text": str(text or "").strip(),
        }
        data = await self._api_post(
            f"dm_conversations/with/{str(participant_id).strip()}/messages",
            payload,
        )
        event_data = data.get("data", {})
        return str(event_data.get("dm_event_id", "")).strip()

    # ------------------------------------------------------------------
    # Engagement
    # ------------------------------------------------------------------

    async def like_tweet(self, tweet_id: str) -> bool:
        me_id, _ = await self.get_me()
        data = await self._api_post(
            f"users/{me_id}/likes", {"tweet_id": str(tweet_id).strip()},
        )
        return bool(data.get("data", {}).get("liked"))

    async def unlike_tweet(self, tweet_id: str) -> bool:
        me_id, _ = await self.get_me()
        data = await self._api_delete(f"users/{me_id}/likes/{str(tweet_id).strip()}")
        return not bool(data.get("data", {}).get("liked"))

    async def retweet(self, tweet_id: str) -> bool:
        me_id, _ = await self.get_me()
        data = await self._api_post(
            f"users/{me_id}/retweets", {"tweet_id": str(tweet_id).strip()},
        )
        return bool(data.get("data", {}).get("retweeted"))

    async def unretweet(self, tweet_id: str) -> bool:
        me_id, _ = await self.get_me()
        data = await self._api_delete(f"users/{me_id}/retweets/{str(tweet_id).strip()}")
        return not bool(data.get("data", {}).get("retweeted"))

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search_tweets(
        self, query: str, *, max_results: int = 10,
    ) -> list[TwitterMessage]:
        """Search recent tweets (last 7 days)."""
        params: dict[str, Any] = {
            "query": str(query or "").strip(),
            "tweet.fields": "author_id,conversation_id,created_at",
            "expansions": "author_id",
            "user.fields": "username",
            "max_results": max(10, min(100, int(max_results))),
        }
        data = await self._api_get("tweets/search/recent", params)
        tweets = data.get("data") or []
        includes = data.get("includes", {})
        users = {
            str(u.get("id", "")): str(u.get("username", ""))
            for u in (includes.get("users") or [])
            if isinstance(u, dict)
        }
        results: list[TwitterMessage] = []
        for tweet in tweets:
            if not isinstance(tweet, dict):
                continue
            author_id = str(tweet.get("author_id", "")).strip()
            results.append(
                TwitterMessage(
                    id=str(tweet.get("id", "")).strip(),
                    text=str(tweet.get("text", "")).strip(),
                    author_id=author_id,
                    author_username=users.get(author_id, ""),
                    conversation_id=str(tweet.get("conversation_id", "")).strip(),
                    created_at=str(tweet.get("created_at", "")).strip(),
                )
            )
        return results

    # ------------------------------------------------------------------
    # Timeline
    # ------------------------------------------------------------------

    async def get_my_tweets(self, *, max_results: int = 10) -> list[TwitterMessage]:
        """Fetch recent tweets from the authenticated user's timeline."""
        me_id, me_username = await self.get_me()
        if not me_id:
            return []
        params: dict[str, Any] = {
            "tweet.fields": "conversation_id,created_at,in_reply_to_user_id",
            "max_results": max(5, min(100, int(max_results))),
        }
        data = await self._api_get(f"users/{me_id}/tweets", params)
        tweets = data.get("data") or []
        results: list[TwitterMessage] = []
        for tweet in tweets:
            if not isinstance(tweet, dict):
                continue
            results.append(
                TwitterMessage(
                    id=str(tweet.get("id", "")).strip(),
                    text=str(tweet.get("text", "")).strip(),
                    author_id=me_id,
                    author_username=me_username,
                    conversation_id=str(tweet.get("conversation_id", "")).strip(),
                    created_at=str(tweet.get("created_at", "")).strip(),
                )
            )
        return results

    # ------------------------------------------------------------------
    # Media upload (v1.1 endpoint — required by v2 tweet creation)
    # ------------------------------------------------------------------

    async def upload_media(self, file_path: str | Path) -> str:
        """Upload media file and return media_id string.

        Uses the v1.1 media/upload endpoint (simple upload for images < 5MB).
        """
        path = Path(file_path).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Media file not found: {path}")
        mime_types: dict[str, str] = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".mp4": "video/mp4",
        }
        mime = mime_types.get(path.suffix.lower(), "application/octet-stream")
        await self._maybe_refresh_token()
        with path.open("rb") as handle:
            files = {"media": (path.name, handle, mime)}
            response = await self._client.post(
                f"{self.upload_base_url}/media/upload.json",
                headers=self._upload_headers(),
                files=files,
            )
        response.raise_for_status()
        data = response.json()
        return str(data.get("media_id_string", "")).strip()

    # ------------------------------------------------------------------
    # Utility: split text into tweet-sized chunks for threads
    # ------------------------------------------------------------------

    @staticmethod
    def split_for_thread(text: str, max_chars: int = TWEET_MAX_CHARS) -> list[str]:
        """Split long text into tweet-sized chunks, breaking at sentence/line boundaries."""
        text = str(text or "").strip()
        if not text:
            return []
        if len(text) <= max_chars:
            return [text]

        chunks: list[str] = []
        while text:
            if len(text) <= max_chars:
                chunks.append(text)
                break
            # Try to split at newline.
            split_at = text.rfind("\n", 0, max_chars)
            if split_at < 80:
                # Try sentence boundary.
                split_at = text.rfind(". ", 0, max_chars)
                if split_at < 80:
                    split_at = max_chars
                else:
                    split_at += 1  # Include the period.
            chunks.append(text[:split_at].rstrip())
            text = text[split_at:].lstrip()
        return chunks
