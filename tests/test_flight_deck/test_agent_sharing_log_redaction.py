"""A1 — FD's log never carries a token from a request's query string.

uvicorn logs the accepted WebSocket path and every HTTP access line WITH the
query string: the member socket's ``fd_token`` (a member's FD access JWT), the
owner socket's ``fd_token`` / ``token`` and download links' ``t``. FD installs
a filter on ``uvicorn.error`` and ``uvicorn.access`` that blanks those values.
"""

from __future__ import annotations

import logging

import pytest

from captain_claw.flight_deck import server

CLIENT = "203.0.113.9:40001"
WS_LINE = '%s - "WebSocket %s" [accepted]'           # uvicorn websockets_impl / wsproto_impl
ACCESS_LINE = '%s - "%s %s HTTP/%s" %d'               # uvicorn h11_impl / httptools_impl
JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1LW1lbWJlciJ9.c2lnbmF0dXJl"


@pytest.mark.parametrize("logger_name,fmt,args,logged", [
    ("uvicorn.error", WS_LINE,
     (CLIENT, f"/fd/agent-ws-shared?ref=process%3Ahelper%3A0123456789abcdef&lane=A&fd_token={JWT}"),
     f'{CLIENT} - "WebSocket /fd/agent-ws-shared?ref=process%3Ahelper%3A0123456789abcdef'
     f'&lane=A&fd_token=…" [accepted]'),
    ("uvicorn.error", WS_LINE,
     (CLIENT, f"/fd/agent-ws/localhost/24987?token=web-auth-secret&fd_token={JWT}"),
     f'{CLIENT} - "WebSocket /fd/agent-ws/localhost/24987?token=…&fd_token=…" [accepted]'),
    ("uvicorn.access", ACCESS_LINE,
     (CLIENT, "GET", "/fd/files/download?path=a%26b.txt&t=dl-secret&at=keep", "1.1", 200),
     f'{CLIENT} - "GET /fd/files/download?path=a%26b.txt&t=…&at=keep HTTP/1.1" 200'),
    ("uvicorn.access", ACCESS_LINE,
     (CLIENT, "GET", "/fd/processes?mytoken=keep&token_type=keep", "1.1", 200),
     f'{CLIENT} - "GET /fd/processes?mytoken=keep&token_type=keep HTTP/1.1" 200'),
])
def test_uvicorn_lines_carry_no_token(caplog, logger_name, fmt, args, logged):
    with caplog.at_level(logging.INFO, logger=logger_name):
        logging.getLogger(logger_name).info(fmt, *args)
    assert [r.getMessage() for r in caplog.records] == [logged]
    for secret in (JWT, "web-auth-secret", "dl-secret"):
        assert secret not in caplog.text


def test_installed_once_by_the_logging_setup():
    root = logging.getLogger()
    saved_root = (list(root.handlers), root.level)
    names = ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi", "flight_deck")
    saved = {n: (list(logging.getLogger(n).handlers), list(logging.getLogger(n).filters),
                 logging.getLogger(n).propagate, logging.getLogger(n).level) for n in names}
    try:
        server._configure_fd_logging()
        server._configure_fd_logging()  # a second load of the module adds no second filter
        for name in ("uvicorn.error", "uvicorn.access"):
            redactors = [f for f in logging.getLogger(name).filters
                         if type(f).__name__ == "_RedactQueryTokenFilter"]
            assert len(redactors) == 1, name
    finally:
        root.handlers[:] = saved_root[0]
        root.setLevel(saved_root[1])
        for n, (handlers, filters, propagate, level) in saved.items():
            lg = logging.getLogger(n)
            lg.handlers[:], lg.filters[:] = handlers, filters
            lg.propagate = propagate
            lg.setLevel(level)


def test_dict_args_and_non_strings_pass_through():
    f = server._RedactQueryTokenFilter()
    record = logging.LogRecord("uvicorn.error", logging.INFO, __file__, 1,
                               "%(path)s %(n)d", ({"path": "/x?fd_token=abc", "n": 3},), None)
    assert f.filter(record) and record.getMessage() == "/x?fd_token=… 3"
    record = logging.LogRecord("uvicorn.error", logging.INFO, __file__, 1, ValueError("e"), None, None)
    assert f.filter(record) and record.getMessage() == "e"
