"""Proof that this package never talks to a provider.

The README promises there is no network code here at all. These tests hold
that promise to two different fires: the source must not import a networking
module, and a full route-and-complete cycle must survive with every socket
in the process sawn off.
"""

import ast
import pathlib
import socket

import pytest

from llm_router_lite import Router, complexity, route

SOURCE_DIR = pathlib.Path(__file__).resolve().parents[1] / "src" / "llm_router_lite"

#: Anything that could reach off the machine.
FORBIDDEN = {
    "asyncio",
    "ftplib",
    "http",
    "httpx",
    "imaplib",
    "poplib",
    "requests",
    "smtplib",
    "socket",
    "socketserver",
    "ssl",
    "telnetlib",
    "urllib",
    "urllib3",
    "webbrowser",
    "xmlrpc",
}


def _imported_modules(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
    return found


def test_no_source_file_imports_a_networking_module():
    offenders = {}
    for path in sorted(SOURCE_DIR.rglob("*.py")):
        bad = _imported_modules(path) & FORBIDDEN
        if bad:
            offenders[path.name] = sorted(bad)
    assert offenders == {}, "networking imports found: {0}".format(offenders)


def test_the_package_only_depends_on_the_standard_library():
    pyproject = (SOURCE_DIR.parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    assert "dependencies = []" in pyproject


class _Blocked(socket.socket):
    def __init__(self, *args, **kw):
        raise AssertionError("llm-router-lite opened a socket")


@pytest.fixture()
def no_sockets(monkeypatch):
    """Make any attempt to open a socket a loud test failure."""
    monkeypatch.setattr(socket, "socket", _Blocked)
    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda *a, **kw: pytest.fail("llm-router-lite opened a connection"),
    )
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **kw: pytest.fail("llm-router-lite resolved a host"),
    )
    return None


def test_a_full_cycle_runs_with_every_socket_blocked(no_sockets):
    router = Router()
    router.add("small", lambda p, **kw: "small: " + p, cost=0.0002, quality=0.4)
    router.add("large", lambda p, **kw: "large: " + p, cost=0.01, quality=0.95)

    assert complexity("What is 2 + 2?").band == "simple"

    decision = router.route("What is 2 + 2?")
    assert decision.model == "small"

    answer = router.complete("What is 2 + 2?")
    assert answer.text == "small: What is 2 + 2?"
    assert answer.attempts[0].ok

    assert router.stats().calls == 1


def test_fallback_after_a_failure_also_stays_offline(no_sockets):
    def broken(prompt, **kw):
        raise RuntimeError("provider down")

    router = Router()
    router.add("small", broken, cost=0.0002, quality=0.4)
    router.add("large", lambda p, **kw: "large: " + p, cost=0.01, quality=0.95)

    answer = router.complete("What is 2 + 2?")
    assert answer.model == "large"
    assert [a.ok for a in answer.attempts] == [False, True]


def test_the_convenience_route_stays_offline(no_sockets):
    decision = route("hello", {"only": lambda p, **kw: p})
    assert decision.model == "only"
