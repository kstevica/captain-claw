"""Agent bundles: one agent — its identity, onboarding, memory, data,
conversations, files and scheduled tasks — in a ``.captain`` file, to back it
up, restore it, or pass it to another Flight Deck.

* :mod:`catalog`  — what an agent is made of, part by part, and its category
* :mod:`stores`   — SQLite snapshots out (backup API), checks on the way in
* :mod:`archive`  — the zip and its manifest; reading never trusts the file
* :mod:`export`   — make a bundle of an agent
* :mod:`restore`  — make a new agent of a bundle, through the normal spawn
* :mod:`routes`   — ``/fd/agent-bundle/*``

See docs/agent-export-plan.md.
"""
