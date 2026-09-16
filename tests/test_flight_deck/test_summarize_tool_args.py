"""The action-log summariser must not truncate identifier-shaped values."""

from captain_claw.flight_deck.basna_routes import _summarize_tool_args


def test_long_vfs_path_survives_untruncated():
    # 43 chars — was cut to 40 (…eppo-authenticity-pac) before the fix.
    path = "vfs:vatra-abcd1234/eppo-authenticity-pack.md"
    out = _summarize_tool_args({"path": path, "content": "x" * 5000})
    assert path in out
    # content is still capped short
    assert "x" * 100 not in out


def test_non_identifier_values_stay_short():
    out = _summarize_tool_args({"text": "y" * 300})
    assert "y" * 41 not in out


def test_empty_and_scalar():
    assert _summarize_tool_args({}) == ""
    assert _summarize_tool_args("hi") == "hi"
