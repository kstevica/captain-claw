"""Eco mode strips deferred tool schemas, keeping only core + intent-matched
tools. The `hosting` tool used to match no intent pattern, so a "publish this to
the web" request left the agent without the tool's schema — it then hallucinated
`hosting` as a shell command and reached for external hosts (surge/npx) outside
our sandbox. These tests pin the intent mapping so hosting stays reachable."""

from __future__ import annotations

import pytest

from captain_claw.agent_orchestration_mixin import _eco_select_tools_by_intent


@pytest.mark.parametrize(
    "message",
    [
        # The exact production phrasing that failed.
        "Publish the saved landing page at saved/showcase/"
        "bcca81ae-ee4f-41f5-8e5e-8807a960cc43/index.html to get a public URL",
        "deploy the weather app",
        "host it and give me a link",
        "put it online",
        "serve the dist folder",
        "make it go live",
        "I need a public url for the site",
        "open /vfs-apps/weather/ for me",
    ],
)
def test_hosting_intent_surfaces_tool(message):
    assert "hosting" in _eco_select_tools_by_intent(message)


@pytest.mark.parametrize(
    "message",
    [
        "what's the weather today",
        "read the report and summarize it",
        "run the tests",
    ],
)
def test_unrelated_messages_do_not_surface_hosting(message):
    assert "hosting" not in _eco_select_tools_by_intent(message)
