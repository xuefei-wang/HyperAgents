import importlib
import json


def test_polyglot_report_path_accepts_string_output_dir(tmp_path):
    harness = importlib.import_module("domains.polyglot.harness")

    report_path = harness._report_path(str(tmp_path), "eval/run", eval_idx=0)

    assert report_path == tmp_path / "eval__run_0.000.json"


def test_polyglot_full_eval_uses_medium_map_and_requested_workers(monkeypatch, tmp_path):
    generate_loop = importlib.import_module("generate_loop")
    polyglot_harness = importlib.import_module("domains.polyglot.harness")
    polyglot_report = importlib.import_module("domains.polyglot.report")

    small_task_map = tmp_path / "small.json"
    medium_task_map = tmp_path / "medium.json"
    small_task_map.write_text(json.dumps(["small-only"]), encoding="utf-8")
    medium_task_map.write_text(json.dumps(["medium-a", "medium-b"]), encoding="utf-8")

    calls = []

    def fake_harness_polyglot(**kwargs):
        calls.append(
            {
                "test_task_list": list(kwargs["test_task_list"]),
                "max_workers": kwargs["max_workers"],
            }
        )
        return []

    monkeypatch.setattr(generate_loop, "POLYGLOT_SMALL_TASK_MAP", small_task_map)
    monkeypatch.setattr(generate_loop, "POLYGLOT_MEDIUM_TASK_MAP", medium_task_map)
    monkeypatch.setattr(generate_loop, "get_patch_files", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(generate_loop, "get_score", lambda *_args, **_kwargs: 1.0)
    monkeypatch.setattr(generate_loop, "update_node_metadata", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(polyglot_harness, "harness", fake_harness_polyglot)
    monkeypatch.setattr(polyglot_report, "report", lambda **_kwargs: None)

    generate_loop.run_harness_polyglot(
        root_dir=".",
        output_dir=str(tmp_path),
        genid=0,
        skip_staged_eval=False,
        num_samples=2,
        max_workers=4,
    )

    assert calls == [
        {"test_task_list": ["small-only"], "max_workers": 4},
        {"test_task_list": ["medium-a", "medium-b"], "max_workers": 4},
    ]


def test_swebench_harness_uses_requested_workers(monkeypatch, tmp_path):
    generate_loop = importlib.import_module("generate_loop")
    swebench_constants = importlib.import_module("domains.swebench_pro.constants")
    swebench_harness = importlib.import_module("domains.swebench_pro.harness")
    swebench_report = importlib.import_module("domains.swebench_pro.report")

    task_map_path = tmp_path / "swebench-task-map.json"
    task_map_path.write_text(
        json.dumps({"tasks": [{"task_id": "swe-1"}, {"task_id": "swe-2"}]}),
        encoding="utf-8",
    )

    calls = []

    def fake_harness_swebench_pro(**kwargs):
        calls.append(
            {
                "test_task_list": list(kwargs["test_task_list"]),
                "max_workers": kwargs["max_workers"],
            }
        )
        return []

    monkeypatch.setattr(generate_loop, "get_patch_files", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(swebench_constants, "SWEBENCH_PRO_DEFAULT_TASK_MAP", task_map_path)
    monkeypatch.setattr(swebench_harness, "harness", fake_harness_swebench_pro)
    monkeypatch.setattr(swebench_report, "report", lambda **_kwargs: None)

    generate_loop.run_harness_swebench_pro(
        root_dir=".",
        output_dir=str(tmp_path),
        genid=0,
        num_samples=1,
        max_workers=3,
    )

    assert calls == [
        {"test_task_list": ["swe-1", "swe-2"], "max_workers": 3},
    ]
