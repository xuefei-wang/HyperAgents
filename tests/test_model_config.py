from agent import llm


def _fake_completion(**kwargs):
    _fake_completion.kwargs = kwargs
    return {"choices": [{"message": {"content": "ok"}}], "usage": {}}


def _clear_model_env(monkeypatch):
    for key in (
        "HYPERAGENTS_TASK_MODEL",
        "HYPERAGENTS_POLYGLOT_MODEL",
        "HYPERAGENTS_META_MODEL",
        "HYPERAGENTS_REASONING_EFFORT",
        "OPENAI_REASONING_EFFORT",
        "REASONING_EFFORT",
        "MODEL_PROVIDER",
        "MODEL",
    ):
        monkeypatch.delenv(key, raising=False)


def test_provider_profile_model_is_normalized(monkeypatch):
    _clear_model_env(monkeypatch)
    monkeypatch.setenv("MODEL_PROVIDER", "openai")
    monkeypatch.setenv("MODEL", "gpt-5.4-mini")

    assert llm.provider_profile_model_from_env() == "openai/gpt-5.4-mini"
    assert llm.task_model_from_env() == "openai/gpt-5.4-mini"
    assert llm.meta_model_from_env() == "openai/gpt-5.4-mini"

    monkeypatch.setenv("MODEL_PROVIDER", "anthropic")
    monkeypatch.setenv("MODEL", "claude-haiku-4-5-20251001")
    assert llm.task_model_from_env() == "anthropic/claude-haiku-4-5-20251001"


def test_reasoning_kwargs_are_provider_gated(monkeypatch):
    _clear_model_env(monkeypatch)
    monkeypatch.setenv("REASONING_EFFORT", "medium")

    assert llm._openai_reasoning_effort("openai/o3") == "medium"
    assert llm._openai_reasoning_effort("o3") == "medium"
    assert llm._openai_reasoning_effort("azure/o3-mini") is None
    assert llm._supports_custom_temperature("azure/o3-mini", None) is True


def test_gpt52_temperature_requires_none_reasoning(monkeypatch):
    _clear_model_env(monkeypatch)
    monkeypatch.setattr(llm.litellm, "completion", _fake_completion)

    monkeypatch.setenv("REASONING_EFFORT", "medium")
    llm.get_response_from_llm("hi", model="openai/gpt-5.2", temperature=0.7)
    assert _fake_completion.kwargs["reasoning_effort"] == "medium"
    assert "temperature" not in _fake_completion.kwargs

    monkeypatch.setenv("REASONING_EFFORT", "none")
    llm.get_response_from_llm("hi", model="openai/gpt-5.2", temperature=0.7)
    assert _fake_completion.kwargs["reasoning_effort"] == "none"
    assert _fake_completion.kwargs["temperature"] == 0.7


def test_runtime_env_collectors_forward_profile_model(monkeypatch):
    from domains.arc import harness as arc_harness
    from domains.swebench_pro import harness as swebench_harness
    import generate_loop

    _clear_model_env(monkeypatch)
    monkeypatch.setenv("MODEL_PROVIDER", "openai")
    monkeypatch.setenv("MODEL_AUTH_MODE", "api")
    monkeypatch.setenv("MODEL", "gpt-5.4-mini")

    for collector in (
        generate_loop._runtime_environment,
        arc_harness._runtime_environment,
        swebench_harness._runtime_environment,
    ):
        runtime_env = collector()
        assert runtime_env["MODEL_PROVIDER"] == "openai"
        assert runtime_env["MODEL_AUTH_MODE"] == "api"
        assert runtime_env["MODEL"] == "gpt-5.4-mini"
