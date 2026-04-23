import importlib
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_polyglot_constants_honor_env_overrides(monkeypatch):
    monkeypatch.setenv("HYPERAGENTS_POLYGLOT_METADATA_PATH", "/tmp/polyglot-meta.json")
    monkeypatch.setenv("HYPERAGENTS_POLYGLOT_SMALL_TASK_MAP", "/tmp/polyglot-small.json")
    monkeypatch.setenv("HYPERAGENTS_POLYGLOT_MEDIUM_TASK_MAP", "/tmp/polyglot-medium.json")

    constants = importlib.import_module("domains.polyglot.constants")
    constants = importlib.reload(constants)

    assert str(constants.POLYGLOT_METADATA_PATH) == "/tmp/polyglot-meta.json"
    assert str(constants.POLYGLOT_SMALL_TASK_MAP) == "/tmp/polyglot-small.json"
    assert str(constants.POLYGLOT_MEDIUM_TASK_MAP) == "/tmp/polyglot-medium.json"
