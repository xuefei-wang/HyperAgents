"""Tests for env-driven polyglot agent timeout resolution (kcsi #1125)."""

from domains.polyglot.constants import _resolve_polyglot_timeout


def test_default_is_600(monkeypatch):
    monkeypatch.delenv("CROSS_RUNNER_AGENT_TIMEOUT_SEC", raising=False)
    monkeypatch.delenv("POLYGLOT_AGENT_TIMEOUT_SECONDS_ENV", raising=False)
    assert _resolve_polyglot_timeout() == 600


def test_cross_runner_env_overrides_default(monkeypatch):
    monkeypatch.setenv("CROSS_RUNNER_AGENT_TIMEOUT_SEC", "3600")
    monkeypatch.delenv("POLYGLOT_AGENT_TIMEOUT_SECONDS_ENV", raising=False)
    assert _resolve_polyglot_timeout() == 3600


def test_ha_specific_env_overrides_default(monkeypatch):
    monkeypatch.delenv("CROSS_RUNNER_AGENT_TIMEOUT_SEC", raising=False)
    monkeypatch.setenv("POLYGLOT_AGENT_TIMEOUT_SECONDS_ENV", "1800")
    assert _resolve_polyglot_timeout() == 1800


def test_cross_runner_wins_over_ha_specific(monkeypatch):
    monkeypatch.setenv("CROSS_RUNNER_AGENT_TIMEOUT_SEC", "3600")
    monkeypatch.setenv("POLYGLOT_AGENT_TIMEOUT_SECONDS_ENV", "1800")
    assert _resolve_polyglot_timeout() == 3600


def test_invalid_env_falls_through_to_default(monkeypatch):
    monkeypatch.setenv("CROSS_RUNNER_AGENT_TIMEOUT_SEC", "not_an_int")
    monkeypatch.delenv("POLYGLOT_AGENT_TIMEOUT_SECONDS_ENV", raising=False)
    assert _resolve_polyglot_timeout() == 600
