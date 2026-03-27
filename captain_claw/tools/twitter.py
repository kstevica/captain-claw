"""Twitter/X tool for posting, searching, liking, retweeting, and DMs."""

from __future__ import annotations

import os
from typing import Any

from captain_claw.config import get_config
from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)

# Module-level singleton so the bridge is shared across tool invocations.
_bridge_instance = None


def _get_bridge():
    """Lazily create or return the shared TwitterBridge instance."""
    global _bridge_instance
    if _bridge_instance is not None:
        return _bridge_instance

    from captain_claw.twitter_bridge import TwitterBridge

    cfg = get_config()

    # Try tool-specific config first, then platform config, then env vars.
    tool_cfg = getattr(cfg.tools, "twitter", None)
    platform_cfg = getattr(cfg, "twitter", None)

    client_id = (
        str(getattr(tool_cfg, "client_id", "") or "").strip()
        or str(getattr(platform_cfg, "client_id", "") or "").strip()
        or os.environ.get("TWITTER_CLIENT_ID", "").strip()
    )
    client_secret = (
        str(getattr(tool_cfg, "client_secret", "") or "").strip()
        or str(getattr(platform_cfg, "client_secret", "") or "").strip()
        or os.environ.get("TWITTER_CLIENT_SECRET", "").strip()
    )
    access_token = (
        str(getattr(tool_cfg, "access_token", "") or "").strip()
        or str(getattr(platform_cfg, "access_token", "") or "").strip()
        or os.environ.get("TWITTER_ACCESS_TOKEN", "").strip()
    )
    refresh_token = (
        str(getattr(tool_cfg, "refresh_token", "") or "").strip()
        or str(getattr(platform_cfg, "refresh_token", "") or "").strip()
        or os.environ.get("TWITTER_REFRESH_TOKEN", "").strip()
    )

    if not access_token:
        return None

    _bridge_instance = TwitterBridge(
        client_id=client_id,
        client_secret=client_secret,
        access_token=access_token,
        refresh_token=refresh_token,
    )
    return _bridge_instance


class TwitterTool(Tool):
    """Interact with Twitter/X: post tweets, reply, search, like, retweet, DM, and more."""

    name = "twitter"
    description = (
        "Interact with Twitter/X. Actions: post (tweet text, optional reply_to_id, "
        "optional image_path), reply (tweet_id, text), search (query, optional count), "
        "like/unlike (tweet_id), retweet/unretweet (tweet_id), dm (user_id, text), "
        "timeline (get own recent tweets), mentions (get recent mentions), "
        "thread (texts as JSON array of strings)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "post", "reply", "search", "like", "unlike",
                    "retweet", "unretweet", "dm", "timeline",
                    "mentions", "thread", "me",
                ],
                "description": "Action to perform.",
            },
            "text": {
                "type": "string",
                "description": "Tweet text, DM text, or search query.",
            },
            "tweet_id": {
                "type": "string",
                "description": "Tweet ID for reply, like, unlike, retweet, unretweet.",
            },
            "user_id": {
                "type": "string",
                "description": "User ID for DM.",
            },
            "image_path": {
                "type": "string",
                "description": "Local image path to attach to a tweet.",
            },
            "count": {
                "type": "number",
                "description": "Max results for search/timeline/mentions (default 10).",
            },
            "texts": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of tweet texts for thread action.",
            },
        },
        "required": ["action"],
    }
    timeout_seconds = 60.0

    async def execute(self, action: str, **kwargs: Any) -> ToolResult:
        bridge = _get_bridge()
        if bridge is None:
            return ToolResult(
                success=False,
                error=(
                    "Twitter not configured. Set twitter credentials in config.yaml "
                    "(tools.twitter or twitter section) or via TWITTER_ACCESS_TOKEN "
                    "environment variable."
                ),
            )
        action = str(action or "").strip().lower()
        try:
            if action == "post":
                return await self._post(bridge, **kwargs)
            elif action == "reply":
                return await self._reply(bridge, **kwargs)
            elif action == "search":
                return await self._search(bridge, **kwargs)
            elif action == "like":
                return await self._like(bridge, **kwargs)
            elif action == "unlike":
                return await self._unlike(bridge, **kwargs)
            elif action == "retweet":
                return await self._retweet(bridge, **kwargs)
            elif action == "unretweet":
                return await self._unretweet(bridge, **kwargs)
            elif action == "dm":
                return await self._dm(bridge, **kwargs)
            elif action == "timeline":
                return await self._timeline(bridge, **kwargs)
            elif action == "mentions":
                return await self._mentions(bridge, **kwargs)
            elif action == "thread":
                return await self._thread(bridge, **kwargs)
            elif action == "me":
                return await self._me(bridge)
            else:
                return ToolResult(success=False, error=f"Unknown action: {action}")
        except Exception as e:
            log.error("twitter tool error", action=action, error=str(e))
            return ToolResult(success=False, error=str(e))

    # -- Action handlers ---------------------------------------------------

    @staticmethod
    async def _post(bridge, **kw: Any) -> ToolResult:
        text = str(kw.get("text", "") or "").strip()
        if not text:
            return ToolResult(success=False, error="text is required for post action.")
        image_path = str(kw.get("image_path", "") or "").strip()
        media_ids = None
        if image_path:
            media_id = await bridge.upload_media(image_path)
            media_ids = [media_id]
        reply_to = str(kw.get("tweet_id", "") or "").strip()
        tweet_id = await bridge.post_tweet(text, reply_to_id=reply_to, media_ids=media_ids)
        return ToolResult(success=True, content=f"Tweet posted. ID: {tweet_id}")

    @staticmethod
    async def _reply(bridge, **kw: Any) -> ToolResult:
        tweet_id = str(kw.get("tweet_id", "") or "").strip()
        text = str(kw.get("text", "") or "").strip()
        if not tweet_id or not text:
            return ToolResult(success=False, error="tweet_id and text are required for reply.")
        new_id = await bridge.post_tweet(text, reply_to_id=tweet_id)
        return ToolResult(success=True, content=f"Reply posted. ID: {new_id}")

    @staticmethod
    async def _search(bridge, **kw: Any) -> ToolResult:
        query = str(kw.get("text", "") or "").strip()
        if not query:
            return ToolResult(success=False, error="text (query) is required for search.")
        count = int(kw.get("count", 10) or 10)
        tweets = await bridge.search_tweets(query, max_results=count)
        if not tweets:
            return ToolResult(success=True, content="No tweets found.")
        lines = [f"Found {len(tweets)} tweets:\n"]
        for t in tweets:
            lines.append(
                f"- [{t.id}] @{t.author_username}: {t.text[:200]}"
                f"{' ...' if len(t.text) > 200 else ''}"
                f" ({t.created_at})"
            )
        return ToolResult(success=True, content="\n".join(lines))

    @staticmethod
    async def _like(bridge, **kw: Any) -> ToolResult:
        tweet_id = str(kw.get("tweet_id", "") or "").strip()
        if not tweet_id:
            return ToolResult(success=False, error="tweet_id is required for like.")
        liked = await bridge.like_tweet(tweet_id)
        return ToolResult(success=True, content=f"Liked tweet {tweet_id}." if liked else f"Like may have failed for {tweet_id}.")

    @staticmethod
    async def _unlike(bridge, **kw: Any) -> ToolResult:
        tweet_id = str(kw.get("tweet_id", "") or "").strip()
        if not tweet_id:
            return ToolResult(success=False, error="tweet_id is required for unlike.")
        unliked = await bridge.unlike_tweet(tweet_id)
        return ToolResult(success=True, content=f"Unliked tweet {tweet_id}." if unliked else f"Unlike may have failed for {tweet_id}.")

    @staticmethod
    async def _retweet(bridge, **kw: Any) -> ToolResult:
        tweet_id = str(kw.get("tweet_id", "") or "").strip()
        if not tweet_id:
            return ToolResult(success=False, error="tweet_id is required for retweet.")
        retweeted = await bridge.retweet(tweet_id)
        return ToolResult(success=True, content=f"Retweeted {tweet_id}." if retweeted else f"Retweet may have failed for {tweet_id}.")

    @staticmethod
    async def _unretweet(bridge, **kw: Any) -> ToolResult:
        tweet_id = str(kw.get("tweet_id", "") or "").strip()
        if not tweet_id:
            return ToolResult(success=False, error="tweet_id is required for unretweet.")
        unretweeted = await bridge.unretweet(tweet_id)
        return ToolResult(success=True, content=f"Unretweeted {tweet_id}." if unretweeted else f"Unretweet may have failed for {tweet_id}.")

    @staticmethod
    async def _dm(bridge, **kw: Any) -> ToolResult:
        user_id = str(kw.get("user_id", "") or "").strip()
        text = str(kw.get("text", "") or "").strip()
        if not user_id or not text:
            return ToolResult(success=False, error="user_id and text are required for dm.")
        dm_id = await bridge.send_dm(user_id, text)
        return ToolResult(success=True, content=f"DM sent. Event ID: {dm_id}")

    @staticmethod
    async def _timeline(bridge, **kw: Any) -> ToolResult:
        count = int(kw.get("count", 10) or 10)
        tweets = await bridge.get_my_tweets(max_results=count)
        if not tweets:
            return ToolResult(success=True, content="No recent tweets found.")
        lines = [f"Your {len(tweets)} most recent tweets:\n"]
        for t in tweets:
            lines.append(
                f"- [{t.id}] {t.text[:200]}"
                f"{' ...' if len(t.text) > 200 else ''}"
                f" ({t.created_at})"
            )
        return ToolResult(success=True, content="\n".join(lines))

    @staticmethod
    async def _mentions(bridge, **kw: Any) -> ToolResult:
        mentions = await bridge.get_mentions()
        if not mentions:
            return ToolResult(success=True, content="No recent mentions found.")
        lines = [f"Found {len(mentions)} recent mentions:\n"]
        for m in mentions:
            lines.append(
                f"- [{m.id}] @{m.author_username}: {m.text[:200]}"
                f"{' ...' if len(m.text) > 200 else ''}"
                f" ({m.created_at})"
            )
        return ToolResult(success=True, content="\n".join(lines))

    @staticmethod
    async def _thread(bridge, **kw: Any) -> ToolResult:
        texts = kw.get("texts") or []
        if not texts or not isinstance(texts, list):
            return ToolResult(success=False, error="texts (array of strings) is required for thread.")
        texts = [str(t).strip() for t in texts if str(t).strip()]
        if not texts:
            return ToolResult(success=False, error="texts must contain at least one non-empty string.")
        ids = await bridge.post_thread(texts)
        return ToolResult(
            success=True,
            content=f"Thread posted ({len(ids)} tweets). IDs: {', '.join(ids)}",
        )

    @staticmethod
    async def _me(bridge) -> ToolResult:
        me_id, me_username = await bridge.get_me()
        return ToolResult(
            success=True,
            content=f"Authenticated as @{me_username} (ID: {me_id})",
        )
