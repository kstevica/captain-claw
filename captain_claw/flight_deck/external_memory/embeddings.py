"""Turning text into vectors for the memories that don't do it themselves.

Qdrant and ArcadeDB keep vectors but don't make them, so Flight Deck does,
with what the connection names:

* nothing — a small model2vec model on Flight Deck itself (the one Captain's
  own memory uses by default): no key, no server, nothing leaves the deck;
* a model name alone — that model2vec model instead (``minishlab/…``);
* an address — any OpenAI-compatible ``/embeddings`` endpoint (Ollama's
  ``/v1``, OpenAI, a gateway…) with that model and, if it needs one, a key.

A memory must keep using the model it was filled with: vectors from two models
can't be compared. :attr:`Embedder.vector_name` names the model in the store
(Qdrant: the collection's named vector; ArcadeDB: a field on each record), so a
connection that changes model never mixes them up.
"""

from __future__ import annotations

import asyncio
import math
import re
import threading
from typing import Any

import httpx

from captain_claw.flight_deck.external_memory.base import (
    CredentialField,
    ProviderError,
    ProviderInputError,
)
from captain_claw.logging import get_logger

log = get_logger(__name__)

DEFAULT_LOCAL_MODEL = "minishlab/potion-base-8M"
# How alike (cosine) a saved item must be to a message to be recalled on its
# own. Measured with the default model on Flight Deck: messages about nothing
# the memory holds scored at most 0.18 against every stored fact, related ones
# mostly 0.28-0.66. Other models sit elsewhere (some score unrelated text
# 0.4+), so a connection can set its own.
DEFAULT_FLOOR_LOCAL = 0.25
DEFAULT_FLOOR_REMOTE = 0.3
TIMEOUT_S = 60.0
MAX_BATCH = 64

FIELDS: tuple[CredentialField, ...] = (
    CredentialField(
        name="embedding_url",
        label="Embeddings address",
        secret=False,
        required=False,
        placeholder="Leave empty to make them on Flight Deck",
        help=(
            "Where vectors are made: an OpenAI-compatible endpoint such as http://127.0.0.1:11434/v1 (Ollama) "
            "or https://api.openai.com/v1. Empty: a small model on Flight Deck itself — no key, nothing leaves "
            "the deck."
        ),
        advanced=True,
    ),
    CredentialField(
        name="embedding_model",
        label="Embeddings model",
        secret=False,
        required=False,
        placeholder=DEFAULT_LOCAL_MODEL,
        help=(
            "The model at that address (e.g. nomic-embed-text, text-embedding-3-small) — or, with no address, "
            "another model2vec model. Keep it once the memory has things in it: vectors from two models can't "
            "be compared."
        ),
        advanced=True,
    ),
    CredentialField(
        name="embedding_key",
        label="Embeddings key",
        required=False,
        placeholder="Only if that endpoint needs one",
        help="Sent as a Bearer token to the embeddings address.",
        advanced=True,
    ),
    CredentialField(
        name="recall_floor",
        label="Recall threshold",
        secret=False,
        required=False,
        placeholder="Leave empty for the model's default",
        help=(
            "How alike (0-1) a saved item must be to a message to be brought up before a reply on its own "
            "(items that share its words always are). Empty: 0.25 for the model on Flight Deck, 0.3 for "
            "others. Raise it if unrelated things come up; lower it if too little does."
        ),
        advanced=True,
    ),
)


def clean(out: dict[str, str]) -> dict[str, str]:
    """Check the embedding fields of cleaned credentials (in place)."""
    url = out.get("embedding_url", "")
    if url:
        if not url.lower().startswith(("http://", "https://")):
            raise ProviderInputError("The embeddings address must start with http:// or https://")
        out["embedding_url"] = url.rstrip("/")
        if not out.get("embedding_model"):
            raise ProviderInputError("With an embeddings address, name its model too (e.g. nomic-embed-text).")
    elif out.get("embedding_key"):
        raise ProviderInputError("An embeddings key needs the embeddings address it is for.")
    if out.get("recall_floor"):
        try:
            floor = float(out["recall_floor"])
        except ValueError:
            floor = -1.0
        if not 0.0 <= floor <= 1.0:
            raise ProviderInputError("The recall threshold is a number from 0 to 1 (e.g. 0.3).")
        out["recall_floor"] = f"{floor:g}"
    return out


def _normalize(values: list[float]) -> list[float]:
    vec = [float(v) if math.isfinite(float(v)) else 0.0 for v in values]
    norm = math.sqrt(sum(v * v for v in vec))
    return vec if norm <= 1e-12 else [v / norm for v in vec]


# One model2vec model per name for the whole process: loading takes seconds.
_local: dict[str, Any] = {}
_local_lock = threading.Lock()


def _local_vectors(model: str, texts: list[str]) -> list[list[float]]:
    """model2vec on this machine (blocking; called in a thread)."""
    with _local_lock:
        enc = _local.get(model)
        if enc is None:
            from captain_claw.semantic_memory import _Model2VecEmbeddingProvider

            enc = _local[model] = _Model2VecEmbeddingProvider(model=model)
    return enc.embed_batch(texts)


class Embedder:
    """Makes vectors for one connection (see the module docstring)."""

    def __init__(self, credentials: dict[str, str], *,
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.url = (credentials.get("embedding_url") or "").rstrip("/")
        self.key = credentials.get("embedding_key") or ""
        self.model = credentials.get("embedding_model") or ("" if self.url else DEFAULT_LOCAL_MODEL)
        try:
            self.floor = float(credentials.get("recall_floor") or "nan")
        except ValueError:
            self.floor = float("nan")
        if self.floor != self.floor:  # not set
            self.floor = DEFAULT_FLOOR_REMOTE if self.url else DEFAULT_FLOOR_LOCAL
        self._transport = transport

    @property
    def local(self) -> bool:
        return not self.url

    @property
    def vector_name(self) -> str:
        """The model as a store may name it (letters, digits, - and _)."""
        base = self.model.rsplit("/", 1)[-1] if self.local else self.model
        slug = re.sub(r"[^A-Za-z0-9_-]+", "-", base).strip("-").lower() or "vec"
        return (("m2v-" if self.local else "") + slug)[:60]

    @property
    def label(self) -> str:
        if self.local:
            return f"{self.model} on Flight Deck"
        return f"{self.model} at {self.url}"

    async def dimensions(self) -> int:
        """Make one vector to learn its size — a connection whose vectors can't
        be made is refused when it is checked (:class:`ProviderInputError`),
        not saved to fail on every write."""
        try:
            return len((await self.embed(["Captain memory check"]))[0])
        except ProviderError as exc:
            raise ProviderInputError(str(exc)) from None

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """One unit vector per text. Raises :class:`ProviderError`."""
        texts = [t if t.strip() else "(empty)" for t in texts]
        if not texts:
            return []
        if self.local:
            try:
                return await asyncio.to_thread(_local_vectors, self.model, texts)
            except Exception as exc:
                log.warning("external memory: local embeddings failed", model=self.model,
                            error=f"{type(exc).__name__}: {exc}")
                raise ProviderError(
                    f"Flight Deck couldn't make vectors with {self.model} ({type(exc).__name__}) — check the "
                    "model name, or set an embeddings address.", retryable=True,
                ) from None
        out: list[list[float]] = []
        for i in range(0, len(texts), MAX_BATCH):
            out.extend(await self._remote(texts[i:i + MAX_BATCH]))
        return out

    async def _remote(self, texts: list[str]) -> list[list[float]]:
        url = self.url if self.url.endswith("/embeddings") else self.url + "/embeddings"
        headers = {"Accept": "application/json", "User-Agent": "captain-flight-deck"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT_S, transport=self._transport) as client:
                resp = await client.post(url, json={"model": self.model, "input": texts}, headers=headers)
        except httpx.HTTPError as exc:
            raise ProviderError(f"Could not reach the embeddings address {self.url}: {type(exc).__name__}",
                                retryable=True) from None
        if resp.status_code in (401, 403):
            raise ProviderError(f"The embeddings address refused the key (HTTP {resp.status_code}).",
                                status=resp.status_code, code="embeddings_unauthorized")
        if resp.status_code >= 400:
            detail = ""
            try:
                body = resp.json()
                err = body.get("error") if isinstance(body, dict) else None
                detail = str((err.get("message") if isinstance(err, dict) else err) or body.get("detail") or "")
            except (ValueError, AttributeError):
                detail = resp.text[:200]
            raise ProviderError(
                f"The embeddings address answered HTTP {resp.status_code}" + (f": {detail[:300]}" if detail else ""),
                status=resp.status_code, retryable=resp.status_code >= 500 or resp.status_code == 429,
            )
        try:
            rows = resp.json().get("data")
            rows = sorted(rows, key=lambda r: r.get("index", 0)) if isinstance(rows, list) else None
            vectors = [_normalize(r["embedding"]) for r in rows] if rows else []
        except (ValueError, AttributeError, KeyError, TypeError):
            vectors = []
        if len(vectors) != len(texts) or not all(vectors):
            raise ProviderError("The embeddings address answered without one vector per text "
                                "(is it OpenAI-compatible?).")
        return vectors
