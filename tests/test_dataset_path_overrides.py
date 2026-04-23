import importlib


def test_polyglot_constants_honor_env_overrides(monkeypatch):
    monkeypatch.setenv("HYPERAGENTS_POLYGLOT_METADATA_PATH", "/tmp/polyglot-meta.json")
    monkeypatch.setenv("HYPERAGENTS_POLYGLOT_SMALL_TASK_MAP", "/tmp/polyglot-small.json")
    monkeypatch.setenv("HYPERAGENTS_POLYGLOT_MEDIUM_TASK_MAP", "/tmp/polyglot-medium.json")

    constants = importlib.import_module("domains.polyglot.constants")
    constants = importlib.reload(constants)

    assert str(constants.POLYGLOT_METADATA_PATH) == "/tmp/polyglot-meta.json"
    assert str(constants.POLYGLOT_SMALL_TASK_MAP) == "/tmp/polyglot-small.json"
    assert str(constants.POLYGLOT_MEDIUM_TASK_MAP) == "/tmp/polyglot-medium.json"


def test_swebench_constants_honor_env_overrides(monkeypatch):
    monkeypatch.setenv("HYPERAGENTS_SWEBENCH_PRO_DATASET_PATH", "/tmp/swebench.jsonl")
    monkeypatch.setenv("HYPERAGENTS_SWEBENCH_PRO_TASK_MAP", "/tmp/swebench-task-map.json")

    constants = importlib.import_module("domains.swebench_pro.constants")
    constants = importlib.reload(constants)

    assert str(constants.SWEBENCH_PRO_DATASET_PATH) == "/tmp/swebench.jsonl"
    assert str(constants.SWEBENCH_PRO_DEFAULT_TASK_MAP) == "/tmp/swebench-task-map.json"
