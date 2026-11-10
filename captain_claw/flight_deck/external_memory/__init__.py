"""External memory on Flight Deck — one per user, shared by all of their agents.

* ``base``      — what a provider is (actions, rules, mirroring), provider-neutral.
* ``membase``   — Membase, the first provider (a key per person).
* ``hindsight`` — Hindsight (one server, a bank per person; learns facts itself).
* ``qdrant``    — Qdrant (one collection, a space per person; vectors + BM25).
* ``arcadedb``  — ArcadeDB (one database, a space per person; full-text, vectors, topic graph).
* ``embeddings``— the vectors Flight Deck makes for the stores that don't.
* ``providers`` — the table of known providers.
* ``outbox``    — the durable queue of mirrored items waiting to be written.
* ``service``   — connections (per user; the deck's with sign-in off), actions, mirroring.
* ``routes``    — ``/fd/external-memory/*`` for the dashboard and the agents.
"""
