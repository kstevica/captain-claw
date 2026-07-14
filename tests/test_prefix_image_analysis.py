"""Tests for server-side image auto-analysis (`_prefix_image_analysis`).

This is the fix for the failure where a weak, non-vision model (e.g.
deepseek-v4-flash), told to describe a pasted image, grabbed the always-on `cv`
(OpenCV pixel-ops) tool instead of image_vision and returned region boxes. The
image is now described server-side — like video already is — and injected into the
turn, so the model never has to pick the tool. These tests pin the three routing
paths and the guardrail directive.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from captain_claw.tools.registry import ToolResult
from captain_claw.web import chat_handler as ch


def _run(coro):
    return asyncio.run(coro)


def _agent():
    a = MagicMock()
    a.session = MagicMock()
    return a


def test_local_vision_model_runs_image_vision_and_injects():
    agent = _agent()
    agent._execute_tool_with_guard = AsyncMock(
        return_value=ToolResult(success=True, content="Five people at a table.")
    )
    with patch("captain_claw.tools.image_ocr.ImageVisionTool") as ivt_cls:
        ivt_cls.return_value._find_model.return_value = object()  # local vision model exists
        out = _run(ch._prefix_image_analysis(agent, "how many people?", ["/x/img.png"], lambda m: None))

    # Ran image_vision (not cv), with the describe prompt.
    assert agent._execute_tool_with_guard.await_args.args[0] == "image_vision"
    assert agent._execute_tool_with_guard.await_args.args[1]["path"] == "/x/img.png"
    # Description injected ahead of the original message.
    assert "Five people at a table." in out
    assert out.index("Five people at a table.") < out.index("how many people?")
    # Directive steers the model to answer from the injected description (non-destructive:
    # it does NOT forbid image_ocr/cv, so explicit OCR / pixel tasks still work).
    assert "Answer the user from that description" in out
    assert "do NOT use the cv tool to 'read' or 'understand'" in out


def test_no_local_model_delegates_to_vision_peer():
    agent = _agent()
    with patch("captain_claw.tools.image_ocr.ImageVisionTool") as ivt_cls, \
         patch("captain_claw.tools.video_vision._find_vision_peer", return_value="MiniMax"), \
         patch("captain_claw.tools.flight_deck.FlightDeckTool") as fdt_cls, \
         patch("captain_claw.tools.video_vision._describe_frame_via_peer",
               new=AsyncMock(return_value="A crowd of about 8 people.")) as dfp:
        ivt_cls.return_value._find_model.return_value = None
        fdt_cls.return_value._get_fd_url.return_value = "http://fd"
        out = _run(ch._prefix_image_analysis(agent, "count them", ["/x/i.png"], lambda m: None))

    assert dfp.await_count == 1
    assert "A crowd of about 8 people." in out


def test_no_vision_path_injects_honest_note_not_a_guess():
    agent = _agent()
    with patch("captain_claw.tools.image_ocr.ImageVisionTool") as ivt_cls, \
         patch("captain_claw.tools.video_vision._find_vision_peer", return_value=None):
        ivt_cls.return_value._find_model.return_value = None
        out = _run(ch._prefix_image_analysis(agent, "what is this?", ["/x/i.png"], lambda m: None))

    assert "no vision model or multimodal peer" in out
    assert "do NOT guess" in out


def test_peer_unreachable_without_fd_url_falls_back_to_note():
    agent = _agent()
    with patch("captain_claw.tools.image_ocr.ImageVisionTool") as ivt_cls, \
         patch("captain_claw.tools.video_vision._find_vision_peer", return_value="MiniMax"), \
         patch("captain_claw.tools.flight_deck.FlightDeckTool") as fdt_cls:
        ivt_cls.return_value._find_model.return_value = None
        fdt_cls.return_value._get_fd_url.return_value = ""  # no way to reach the peer
        out = _run(ch._prefix_image_analysis(agent, "describe", ["/x/i.png"], lambda m: None))

    assert "no vision model or multimodal peer" in out


def test_tool_failure_is_reported_not_raised():
    agent = _agent()
    agent._execute_tool_with_guard = AsyncMock(
        return_value=ToolResult(success=False, error="model unavailable")
    )
    with patch("captain_claw.tools.image_ocr.ImageVisionTool") as ivt_cls:
        ivt_cls.return_value._find_model.return_value = object()
        out = _run(ch._prefix_image_analysis(agent, "hi", ["/x/i.png"], lambda m: None))

    assert "automatic analysis failed" in out
    assert "model unavailable" in out


def test_multiple_images_each_analyzed():
    agent = _agent()
    agent._execute_tool_with_guard = AsyncMock(
        side_effect=[
            ToolResult(success=True, content="First image."),
            ToolResult(success=True, content="Second image."),
        ]
    )
    with patch("captain_claw.tools.image_ocr.ImageVisionTool") as ivt_cls:
        ivt_cls.return_value._find_model.return_value = object()
        out = _run(ch._prefix_image_analysis(agent, "both?", ["/a.png", "/b.png"], lambda m: None))

    assert "First image." in out and "Second image." in out
    assert agent._execute_tool_with_guard.await_count == 2
