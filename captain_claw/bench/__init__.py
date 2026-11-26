"""Headless runs for outside benchmarks (Harbor tasks such as ShellBench).

``captain-claw-once -p <provider> -m <model> "<instruction>"`` runs one task
through Flight Deck — a private deck spawns an agent on the current folder and
gets the task through its inbound MCP — and writes a result file: the answer,
token usage, changed files and the transcript (``run_once``, ``deck_run``).
``evals/harbor/`` holds the Harbor adapter that calls it.
"""
