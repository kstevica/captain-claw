from captain_claw.tools.registry import ToolResult


def test_tool_result_populates_error_from_content_on_failure() -> None:
    result = ToolResult(success=False, content="command failed with exit code 1")

    assert result.error == "command failed with exit code 1"


def test_tool_result_keeps_explicit_error_on_failure() -> None:
    result = ToolResult(success=False, content="stderr output", error="explicit error")

    assert result.error == "explicit error"
