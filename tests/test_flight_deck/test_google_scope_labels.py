"""Granted access names every scope, including the long names Google reports
the email and profile scopes by once it has granted them."""

from captain_claw.flight_deck.google_oauth_routes import _label_scopes


def test_long_form_identity_scopes_get_their_labels():
    out = _label_scopes([
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile",
        "https://www.googleapis.com/auth/gmail.readonly",
        "openid",
    ])
    labels = [o["label"] for o in out]
    assert labels[:2] == ["Email address", "Basic profile"]
    assert not any(label.startswith("https://") for label in labels)
    assert out[0]["scope"] == "https://www.googleapis.com/auth/userinfo.email"  # the scope itself unchanged


def test_unknown_scope_keeps_its_name():
    assert _label_scopes(["https://example.com/x"]) == [{"scope": "https://example.com/x", "label": "https://example.com/x"}]
