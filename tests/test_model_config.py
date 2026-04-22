from types import SimpleNamespace

from agent import llm
from domains import harness
from domains.imo import grading_utils, proof_grading_utils, proof_utils


def test_reasoning_kwargs_are_provider_gated(monkeypatch):
    monkeypatch.delenv("HYPERAGENTS_REASONING_EFFORT", raising=False)
    monkeypatch.delenv("OPENAI_REASONING_EFFORT", raising=False)
    monkeypatch.delenv("REASONING_EFFORT", raising=False)

    assert llm._openai_reasoning_effort("openai/o3") == "medium"
    assert llm._openai_reasoning_effort("o3") == "medium"
    assert llm._openai_reasoning_effort("azure/o3-mini") is None
    assert llm._supports_custom_temperature("azure/o3-mini") is True


def test_generic_harness_honors_task_model_override(monkeypatch):
    monkeypatch.setenv("HYPERAGENTS_TASK_MODEL", "openai/gpt-5.4-mini")

    assert harness._domain_model(SimpleNamespace(MODEL="openai/gpt-4o")) == "openai/gpt-5.4-mini"


def test_imo_defaults_use_shared_openai_model():
    assert proof_utils.MODEL == llm.OPENAI_MODEL
    assert grading_utils.MODEL == llm.OPENAI_MODEL
    assert proof_grading_utils.MODEL == llm.OPENAI_MODEL
