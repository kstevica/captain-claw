"""Telling a robot check from a page in the person's Chrome
(browser-extension/chrome/lib/robot-check.js, run in node).

Pinned here: Google's /sorry/ page, Cloudflare's "Just a moment…" and its
Turnstile, reCAPTCHA / hCaptcha / DataDome / Arkose frames, PerimeterX's "Press
& hold", Akamai, Amazon's characters, a short page asking to prove a person is
there — each is a check a person can answer; Cloudflare's passes by itself
first. A long page with a captcha box in a form, an article about captchas, an
invisible reCAPTCHA and an ordinary page are pages. A short page that only
refuses ("Access denied") is a block, never handed to the person.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROBOT_JS = Path(__file__).resolve().parents[2] / "browser-extension" / "chrome" / "lib" / "robot-check.js"

_RUN = r"""
const m = await import(process.argv[1])
const cases = JSON.parse(process.argv[2])
const out = {}
for (const [key, facts] of Object.entries(cases)) out[key] = m.judge(facts)
console.log(JSON.stringify(out))
"""

LONG = "An ordinary article about many things, with paragraphs of text. " * 60


def _judge(**cases):
    if shutil.which("node") is None:
        pytest.skip("needs node")
    out = subprocess.run(["node", "--input-type=module", "-e", _RUN, ROBOT_JS.as_uri(), json.dumps(cases)],
                         capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def _page(url="https://example.org/a", title="", text="", frames=(), boxes=()):
    return {"url": url, "title": title, "text": text, "length": len(text), "frames": list(frames), "boxes": list(boxes),
            "ready": "complete"}


def test_checks_a_person_can_answer():
    got = _judge(
        google=_page("https://www.google.com/sorry/index?continue=x", "https://www.google.com/search?q=x",
                     "Our systems have detected unusual traffic from your computer network.", boxes=["#captcha-form"]),
        cloudflare=_page(title="Just a moment...", text="Verify you are human by completing the action below.",
                         frames=["https://challenges.cloudflare.com/cdn-cgi/challenge-platform/h/b/turnstile/if/ov2"]),
        turnstile=_page(title="example.org", text="example.org needs to review the security of your connection",
                        boxes=[".cf-turnstile"]),
        recaptcha=_page(title="Security check", text="Please confirm you are human.",
                        frames=["https://www.google.com/recaptcha/api2/anchor?k=x&size=normal"]),
        hcaptcha=_page(text="One more step", frames=["https://newassets.hcaptcha.com/captcha/v1/x/static/hcaptcha.html"]),
        datadome=_page(title="site.example", text="", frames=["https://geo.captcha-delivery.com/captcha/?initialCid=x"]),
        perimeterx=_page(title="Access to this page has been denied", text="Press & Hold to confirm you are a human",
                         boxes=["#px-captcha"]),
        akamai=_page(text="Please wait while we check", boxes=["#sec-if-cpt-container"]),
        amazon=_page("https://www.amazon.com/errors/validateCaptcha", "Amazon.com",
                     "Enter the characters you see below. Sorry, we just need to make sure you're not a robot.",
                     boxes=['form[action*="validateCaptcha"]']),
        arkose=_page(text="Verify your account", frames=["https://client-api.arkoselabs.com/fc/gc/?token=x"]),
        words=_page(title="Welcome", text="Please complete the security check to continue."),
        cf_path=_page("https://example.org/cdn-cgi/challenge-platform/h/g/orchestrate"),
    )
    for key, j in got.items():
        assert j["check"] is True and j["block"] is False, (key, j)
    assert got["google"]["vendor"] == "Google" and got["perimeterx"]["vendor"] == "PerimeterX"
    assert got["datadome"]["vendor"] == "DataDome" and got["amazon"]["vendor"] == "Amazon"
    assert got["cloudflare"]["passes"] is True and got["turnstile"]["passes"] is True
    assert got["google"]["passes"] is False and got["recaptcha"]["passes"] is False


def test_pages_are_pages_and_a_refusal_is_no_check():
    got = _judge(
        ordinary=_page(title="Docs", text=LONG),
        contact_form=_page(title="Contact us", text=LONG, boxes=[".g-recaptcha"],
                           frames=["https://www.google.com/recaptcha/api2/anchor?k=x&size=normal"]),
        article=_page(title="How CAPTCHA works — an explainer", text=LONG),
        quoted=_page(title="Blog", text=LONG + " The page said: verify you are human."),
        short_plain=_page(title="Home", text="Welcome to our small site."),
        empty=_page(),
        refused=_page(title="Access denied", text="Access denied. You don't have permission to access this server."),
        broken={"url": "not a url"},
        nothing=None,
    )
    for key in ("ordinary", "contact_form", "article", "quoted", "short_plain", "empty", "broken", "nothing"):
        assert got[key]["check"] is False and got[key]["block"] is False, (key, got[key])
    assert got["refused"]["check"] is False and got["refused"]["block"] is True


def test_the_extension_loads_it_and_tests_and_git_see_it():
    text = ROBOT_JS.read_text("utf-8")
    assert "export function judge" in text and "export function pageFacts" in text
    root = ROBOT_JS.parents[3]
    for name in ("search.js", "pages.js"):
        assert "./lib/robot-check.js" in (root / "browser-extension" / "chrome" / name).read_text("utf-8")
    ignored = subprocess.run(["git", "check-ignore", "-q", str(ROBOT_JS)], cwd=root, capture_output=True)
    assert ignored.returncode == 1   # kept by git (the repo's Python rule ignores lib/ folders)
