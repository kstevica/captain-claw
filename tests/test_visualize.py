"""The visualize tool: HTML in, the saved page and a PNG of it out.

The page-building and wiring tests are pure. The render tests start a real
headless Chromium (skipped when it can't launch) on pages that need no
network: what they pin is that a drawing comes back as a PNG, that a blank
one is refused with its script error, and that the page can't reach the host.
"""

from __future__ import annotations

import asyncio

import pytest

from captain_claw.agent_orchestration_mixin import _eco_select_tools_by_intent
from captain_claw.config import ToolsConfig
from captain_claw.tools import visualize as viz
from captain_claw.tools.visualize import VisualizeTool, build_document


def test_fragment_is_wrapped_with_its_libraries():
    page = build_document("<canvas id='c'></canvas>", "Sales <Q3>", ["chartjs", "mermaid"])
    assert page.startswith("<!doctype html>")
    assert "<title>Sales &lt;Q3&gt;</title>" in page
    assert "chart.js@4.4.1/dist/chart.umd.js" in page
    assert "mermaid.initialize({startOnLoad: true})" in page
    assert page.index("chart.umd.js") < page.index("<canvas id='c'>")


def test_full_document_gets_libraries_in_its_head_not_its_header():
    doc = ("<!doctype html><html><head><title>x</title></head>"
           "<body><header>top</header><div id='p'></div></body></html>")
    page = build_document(doc, "", ["echarts"])
    assert page.index("echarts.min.js") < page.index("<title>x</title>")
    assert "<header>top</header>" in page
    assert build_document(doc, "", []) == doc


def test_full_document_without_head_gets_one():
    page = build_document("<html><body>hi</body></html>", "", ["d3"])
    assert "<html>\n<head>\n<script src=\"https://cdn.jsdelivr.net/npm/d3@7.9.0" in page


def test_library_names_aliases_and_unknown():
    known, unknown = viz._libraries(["Chart.js", "plotly", "chartjs", "highcharts"])
    assert known == ["chartjs", "plotly"]
    assert unknown == ["highcharts"]
    assert viz._libraries("d3, mermaid") == (["d3", "mermaid"], [])


def test_slug_is_ascii_and_never_empty():
    assert viz._slug("Prodaja po kvartalima — Čakovec") == "prodaja-po-kvartalima-cakovec"
    assert viz._slug("📊") == "visualization"


@pytest.mark.parametrize("message", [
    "draw a bar chart of revenue by quarter",
    "can you plot this data",
    "make a flowchart of the onboarding",
    "napravi graf prodaje po mjesecima",
    "treba mi dijagram procesa",
    "vizualiziraj ove podatke",
])
def test_chart_requests_surface_the_tool(message):
    assert "visualize" in _eco_select_tools_by_intent(message)


@pytest.mark.parametrize("message", ["what's the weather today", "send the email to Ana"])
def test_other_requests_dont(message):
    assert "visualize" not in _eco_select_tools_by_intent(message)


def test_every_agent_has_it():
    # An archetype or spawn list replaces the default; it comes back anyway.
    assert "visualize" in ToolsConfig().enabled
    assert "visualize" in ToolsConfig(enabled=["read"]).enabled


# ── Rendering (real Chromium) ─────────────────────────────────────────


def _chromium_ok() -> bool:
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        return False

    async def probe() -> bool:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            await browser.close()
        return True

    try:
        return asyncio.run(probe())
    except Exception:
        return False


needs_chromium = pytest.mark.skipif(not _chromium_ok(), reason="headless Chromium unavailable")


def _run(tool: VisualizeTool, tmp_path, **args):
    kwargs = {"_saved_base_path": str(tmp_path / "saved"), "_session_id": "s1"}
    return asyncio.run(tool.execute(**args, **kwargs))


@needs_chromium
def test_svg_renders_to_png_and_page(tmp_path):
    html = ('<h2>Bars</h2><svg width="300" height="120">'
            '<rect x="10" y="20" width="60" height="100" fill="#4f7cff"/>'
            '<rect x="90" y="60" width="60" height="60" fill="#ff8a3d"/></svg>')
    result = _run(VisualizeTool(), tmp_path, html=html, title="Bars", width=400, scale=1)
    assert result.success, result.error
    lines = dict(line.split(": ", 1) for line in result.content.splitlines() if ": " in line)
    png = tmp_path / "saved" / "showcase" / "s1" / "bars.png"
    page = png.with_suffix(".html")
    assert lines["Path"] == str(png) and lines["HTML"] == str(page)
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert "<rect" in page.read_text()
    assert "400x" in lines["Picture"]
    assert "Problems" not in result.content


@needs_chromium
def test_blank_picture_is_refused_with_the_script_error(tmp_path):
    result = _run(VisualizeTool(), tmp_path, html="<div></div><script>throw new Error('no data')</script>")
    assert not result.success
    assert "blank" in result.error
    assert "no data" in result.error
    assert not (tmp_path / "saved" / "showcase" / "s1" / "visualization.png").exists()


@needs_chromium
def test_page_cannot_reach_the_host(tmp_path):
    html = """<p id="r">waiting</p><script>
      fetch('http://127.0.0.1:9/secret').then(() => { r.textContent = 'LEAKED' })
        .catch(() => { r.textContent = 'refused' }).finally(() => { window.vizReady = true });
    </script>"""
    result = _run(VisualizeTool(), tmp_path, html=html, title="probe")
    assert result.success, result.error
    assert "127.0.0.1 is not a public address" in result.content
