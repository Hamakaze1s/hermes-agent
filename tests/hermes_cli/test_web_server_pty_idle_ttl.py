"""Local patch P12: the keep-alive PTY idle TTL is configurable.

Upstream hardcodes 30 minutes for the dashboard ``/chat`` keep-alive PTY
sessions. A detached TUI frontend (``node entry.js``, 150-900 MB) holds its
heap for the whole window, so memory-tight hosts shorten it through
``dashboard.pty_idle_ttl_seconds``. Missing / zero / negative / garbage values
fall back to the upstream default, so a typo can never disable the reaper.
"""

from __future__ import annotations

import pytest

DEFAULT_TTL = 30 * 60


def _patch_config(monkeypatch, value):
    """Patch the config loader the TTL helper reads at call time."""
    from hermes_cli import config as hermes_config

    monkeypatch.setattr(
        hermes_config,
        "load_config",
        lambda: {"dashboard": {"pty_idle_ttl_seconds": value}},
    )


def test_pty_idle_ttl_reads_configured_value(monkeypatch):
    from hermes_cli import web_server_chat

    _patch_config(monkeypatch, 600)
    assert web_server_chat._pty_idle_ttl_seconds() == 600


def test_missing_key_falls_back_to_default(monkeypatch):
    from hermes_cli import config as hermes_config
    from hermes_cli import web_server_chat

    monkeypatch.setattr(hermes_config, "load_config", lambda: {"dashboard": {}})
    assert web_server_chat._pty_idle_ttl_seconds() == DEFAULT_TTL


@pytest.mark.parametrize("bad", [0, -5, "abc", None])
def test_invalid_values_fall_back_to_default(monkeypatch, bad):
    from hermes_cli import web_server_chat

    _patch_config(monkeypatch, bad)
    assert web_server_chat._pty_idle_ttl_seconds() == DEFAULT_TTL


def test_unreadable_config_falls_back_to_default(monkeypatch):
    from hermes_cli import config as hermes_config
    from hermes_cli import web_server_chat

    def _boom():
        raise RuntimeError("config unreadable")

    monkeypatch.setattr(hermes_config, "load_config", _boom)
    assert web_server_chat._pty_idle_ttl_seconds() == DEFAULT_TTL


def test_shipped_default_is_a_positive_ttl():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["dashboard"]["pty_idle_ttl_seconds"] > 0
