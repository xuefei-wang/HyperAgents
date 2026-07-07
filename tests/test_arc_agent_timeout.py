"""Tests for env-driven ARC agent timeout resolution (kcsi #1196 / #1125)."""

from domains.arc.constants import _resolve_arc_agent_timeout


def test_default_is_600(monkeypatch):
    monkeypatch.delenv("CROSS_RUNNER_AGENT_TIMEOUT_SEC", raising=False)
    monkeypatch.delenv("ARC_AGENT_TIMEOUT_SECONDS_ENV", raising=False)
    assert _resolve_arc_agent_timeout() == 600


def test_cross_runner_env_overrides_default(monkeypatch):
    monkeypatch.setenv("CROSS_RUNNER_AGENT_TIMEOUT_SEC", "3600")
    monkeypatch.delenv("ARC_AGENT_TIMEOUT_SECONDS_ENV", raising=False)
    assert _resolve_arc_agent_timeout() == 3600


def test_ha_specific_env_overrides_default(monkeypatch):
    monkeypatch.delenv("CROSS_RUNNER_AGENT_TIMEOUT_SEC", raising=False)
    monkeypatch.setenv("ARC_AGENT_TIMEOUT_SECONDS_ENV", "1800")
    assert _resolve_arc_agent_timeout() == 1800


def test_cross_runner_wins_over_ha_specific(monkeypatch):
    monkeypatch.setenv("CROSS_RUNNER_AGENT_TIMEOUT_SEC", "3600")
    monkeypatch.setenv("ARC_AGENT_TIMEOUT_SECONDS_ENV", "1800")
    assert _resolve_arc_agent_timeout() == 3600


def test_invalid_env_falls_through_to_default(monkeypatch):
    monkeypatch.setenv("CROSS_RUNNER_AGENT_TIMEOUT_SEC", "not_an_int")
    monkeypatch.delenv("ARC_AGENT_TIMEOUT_SECONDS_ENV", raising=False)
    assert _resolve_arc_agent_timeout() == 600


def test_zero_falls_through_to_default(monkeypatch):
    monkeypatch.setenv("CROSS_RUNNER_AGENT_TIMEOUT_SEC", "0")
    monkeypatch.delenv("ARC_AGENT_TIMEOUT_SECONDS_ENV", raising=False)
    assert _resolve_arc_agent_timeout() == 600
