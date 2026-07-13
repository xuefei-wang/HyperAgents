import os
import subprocess
import sys
import types
import json
from pathlib import Path

import pytest

try:
    import litellm  # noqa: F401
except ModuleNotFoundError:
    sys.modules.setdefault("litellm", types.SimpleNamespace(drop_params=False))

try:
    from docker.models.containers import Container  # noqa: F401
    from docker.types import Mount  # noqa: F401
except ModuleNotFoundError:
    docker_module = types.ModuleType("docker")
    docker_models_module = types.ModuleType("docker.models")
    docker_containers_module = types.ModuleType("docker.models.containers")
    docker_containers_module.Container = object
    docker_types_module = types.ModuleType("docker.types")
    docker_types_module.Mount = object
    sys.modules.setdefault("docker", docker_module)
    sys.modules.setdefault("docker.models", docker_models_module)
    sys.modules.setdefault("docker.models.containers", docker_containers_module)
    sys.modules.setdefault("docker.types", docker_types_module)
sys.modules.setdefault(
    "analysis.plot_progress",
    types.SimpleNamespace(
        plot_progress_single=lambda *args, **kwargs: None,
        plot_progress_together=lambda *args, **kwargs: None,
    ),
)
sys.modules.setdefault(
    "analysis.visualize_archive",
    types.SimpleNamespace(
        visualize_archive_single=lambda *args, **kwargs: None,
        visualize_archive_together=lambda *args, **kwargs: None,
    ),
)

import generate_loop


class _ExecResult:
    def __init__(self, exit_code=0, output=b""):
        self.exit_code = exit_code
        self.output = output


class _FakeContainer:
    def __init__(self, fail_on=None):
        self.fail_on = fail_on
        self.calls = []

    def start(self):
        self.calls.append((["start"], None))

    def exec_run(self, cmd, workdir=None):
        self.calls.append((cmd, workdir))
        rendered = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
        if self.fail_on and self.fail_on in rendered:
            return _ExecResult(exit_code=1, output=b"delete failed")
        return _ExecResult()


def _commands(container):
    return [" ".join(cmd) if isinstance(cmd, list) else str(cmd) for cmd, _ in container.calls]


def test_no_archive_copy_prunes_raw_grader_artifacts_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("KCSI_HA_LEAK_TRAIN_EVAL_ARTIFACTS", raising=False)
    copied = []
    monkeypatch.setattr(
        generate_loop,
        "copy_to_container",
        lambda container, source_path, dest_path: copied.append((source_path, dest_path)),
    )
    prev_gen_dir = tmp_path / "run" / "gen_0"
    prev_gen_dir.mkdir(parents=True)
    container = _FakeContainer()

    dest = generate_loop.copy_no_archive_prev_eval_to_container(
        container,
        str(prev_gen_dir),
        "/container/out",
        current_genid=1,
    )

    assert dest == "/container/out/run/gen_0"
    assert copied == [(str(prev_gen_dir), "/container/out/run/gen_0")]
    rendered = "\n".join(_commands(container))
    assert "*_eval.md" in rendered
    assert "*_docker.log" in rendered
    assert "-delete" in rendered
    assert "-print -quit" in rendered


def test_leak_flag_preserves_raw_grader_artifacts(monkeypatch):
    monkeypatch.setenv("KCSI_HA_LEAK_TRAIN_EVAL_ARTIFACTS", "1")
    container = _FakeContainer()

    generate_loop._prune_copied_eval_tree(container, "/container/out/tree", current_genid=1)

    rendered = "\n".join(_commands(container))
    assert "*_eval.md" not in rendered
    assert "*_docker.log" not in rendered


def test_pruning_failure_raises_instead_of_leaking_artifacts(monkeypatch):
    monkeypatch.delenv("KCSI_HA_LEAK_TRAIN_EVAL_ARTIFACTS", raising=False)
    container = _FakeContainer(fail_on="*_eval.md")

    with pytest.raises(generate_loop.EvalTreePruneError, match="Failed to prune copied eval tree"):
        generate_loop._prune_copied_eval_tree(container, "/container/out/tree", current_genid=1)


class _LocalBashContainer:
    """Runs the prune bash commands against a real on-disk eval tree.

    ``_prune_copied_eval_tree`` only ever shells out via ``exec_run(["bash",
    "-lc", cmd])``; executing those commands for real (against a tmp tree that
    mirrors the harness output layout) gives a behavioral test that the leak
    files are actually removed and the scalar/transcript files survive, rather
    than just asserting on the rendered command strings.
    """

    def __init__(self):
        self.calls = []

    def start(self):
        self.calls.append((["start"], None))

    def exec_run(self, cmd, workdir=None):
        self.calls.append((cmd, workdir))
        proc = subprocess.run(cmd, capture_output=True)
        return _ExecResult(exit_code=proc.returncode, output=proc.stdout + proc.stderr)


def _build_eval_tree(root: Path):
    """Create a run dir mirroring swebench_pro/arc/polyglot harness output.

    Returns (kept_paths, deleted_paths) as absolute Paths.
    """
    kept = []
    deleted = []

    def write(rel, content="x", *, keep):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        (kept if keep else deleted).append(p)

    # Per-gen node metadata + the agent's OWN outputs (in-regime -> KEEP).
    write("gen_0/metadata.json", keep=True)
    write("gen_0/agent_output/model_patch.diff", keep=True)
    write("gen_0/agent_output/meta_agent_chat_history.md", keep=True)

    # SWE-bench Pro: scalar report + per-task scalars/transcripts KEEP; the
    # whole official_eval/ grader tree (raw stdout/stderr/output/entryscript) DELETE.
    write("gen_0/swebench_pro_eval/report.json", keep=True)
    write("gen_0/swebench_pro_eval/eval_run_0/task_results.json", keep=True)
    write("gen_0/swebench_pro_eval/eval_run_0/eval_run_patches.json", keep=True)
    write("gen_0/swebench_pro_eval/eval_run_0/django__django-1.json", keep=True)
    write("gen_0/swebench_pro_eval/eval_run_0/django__django-1.md", keep=True)
    write("gen_0/swebench_pro_eval/eval_run_0/official_eval/eval_results.json", keep=False)
    write("gen_0/swebench_pro_eval/eval_run_0/official_eval/django__django-1/eval_run_stdout.log", keep=False)
    write("gen_0/swebench_pro_eval/eval_run_0/official_eval/django__django-1/eval_run_stderr.log", keep=False)
    write("gen_0/swebench_pro_eval/eval_run_0/official_eval/django__django-1/eval_run_output.json", keep=False)
    write("gen_0/swebench_pro_eval/eval_run_0/official_eval/django__django-1/eval_run_entryscript.sh", keep=False)

    # ARC: scalar report KEEP; official_eval/results.json (per-attempt correctness) DELETE.
    write("gen_0/arc1_eval/report.json", keep=True)
    write("gen_0/arc1_eval/eval_run_0/task_results.json", keep=True)
    write("gen_0/arc1_eval/eval_run_0/official_eval/results.json", keep=False)

    # Polyglot: scalar report + agent chat + per-task scalar KEEP; the raw
    # grader tails (_eval.md, _docker.log) live at top level -> DELETE.
    write("gen_0/polyglot_eval/report.json", keep=True)
    write("gen_0/polyglot_eval/eval_run_0/rust__wordy.md", keep=True)
    write("gen_0/polyglot_eval/eval_run_0/rust__wordy.json", keep=True)
    write("gen_0/polyglot_eval/eval_run_0/rust__wordy_eval.md", keep=False)
    write("gen_0/polyglot_eval/eval_run_0/rust__wordy_docker.log", keep=False)

    return kept, deleted


def test_prune_removes_grader_artifacts_end_to_end(monkeypatch, tmp_path):
    monkeypatch.delenv("KCSI_HA_LEAK_TRAIN_EVAL_ARTIFACTS", raising=False)
    root = tmp_path / "run"
    kept, deleted = _build_eval_tree(root)
    container = _LocalBashContainer()

    # current_genid=99 so the gen_<id> directory prune does not touch gen_0.
    generate_loop._prune_copied_eval_tree(container, str(root), current_genid=99)

    for path in deleted:
        assert not path.exists(), f"leak artifact survived prune: {path}"
    # The whole official_eval/ subtree must be gone, not just its files.
    assert not list(root.rglob("official_eval")), "official_eval dir survived prune"
    for path in kept:
        assert path.exists(), f"in-regime file was wrongly pruned: {path}"


def test_prune_leak_flag_retains_grader_artifacts_end_to_end(monkeypatch, tmp_path):
    monkeypatch.setenv("KCSI_HA_LEAK_TRAIN_EVAL_ARTIFACTS", "1")
    root = tmp_path / "run"
    kept, deleted = _build_eval_tree(root)
    container = _LocalBashContainer()

    generate_loop._prune_copied_eval_tree(container, str(root), current_genid=99)

    # With the escape hatch set, every artifact is retained for faithful repro.
    for path in kept + deleted:
        assert path.exists(), f"leak flag should retain: {path}"


def test_default_prune_commands_cover_new_leak_surfaces(monkeypatch):
    monkeypatch.delenv("KCSI_HA_LEAK_TRAIN_EVAL_ARTIFACTS", raising=False)
    container = _FakeContainer()

    generate_loop._prune_copied_eval_tree(container, "/container/out/tree", current_genid=1)

    rendered = "\n".join(_commands(container))
    for needle in (
        "official_eval",
        "*_stdout.log",
        "*_stderr.log",
        "*_output.json",
        "*_entryscript.sh",
        "*_eval.md",
        "*_docker.log",
    ):
        assert needle in rendered, f"prune does not cover {needle}"
    # Fail-closed re-scan covers both the directory and the file patterns.
    assert "-print -quit" in rendered


def test_leak_flag_omits_official_eval_prune(monkeypatch):
    monkeypatch.setenv("KCSI_HA_LEAK_TRAIN_EVAL_ARTIFACTS", "1")
    container = _FakeContainer()

    generate_loop._prune_copied_eval_tree(container, "/container/out/tree", current_genid=1)

    rendered = "\n".join(_commands(container))
    assert "official_eval" not in rendered
    assert "*_stdout.log" not in rendered


@pytest.mark.parametrize("fail_on", ["official_eval", "*_stdout.log", "*_entryscript.sh"])
def test_new_artifact_prune_failure_raises(monkeypatch, fail_on):
    monkeypatch.delenv("KCSI_HA_LEAK_TRAIN_EVAL_ARTIFACTS", raising=False)
    container = _FakeContainer(fail_on=fail_on)

    with pytest.raises(generate_loop.EvalTreePruneError, match="Failed to prune copied eval tree"):
        generate_loop._prune_copied_eval_tree(container, "/container/out/tree", current_genid=1)


def test_generate_propagates_prune_failures_instead_of_continuing(monkeypatch, tmp_path):
    container = _FakeContainer()
    monkeypatch.setattr(generate_loop, "build_container", lambda *args, **kwargs: container)
    monkeypatch.setattr(generate_loop, "apply_diffs_container", lambda *args, **kwargs: "commit")
    monkeypatch.setattr(generate_loop, "get_patch_files", lambda *args, **kwargs: [])
    monkeypatch.setattr(generate_loop, "is_starting_node", lambda genid: False)
    monkeypatch.setattr(
        generate_loop,
        "copy_prev_eval_to_container",
        lambda *args, **kwargs: (_ for _ in ()).throw(generate_loop.EvalTreePruneError("prune failed")),
    )
    monkeypatch.setattr(generate_loop, "cleanup_container", lambda *args, **kwargs: None)
    monkeypatch.setattr(generate_loop, "get_score", lambda *args, **kwargs: None)

    with pytest.raises(generate_loop.EvalTreePruneError, match="prune failed"):
        generate_loop.generate(
            docker_client=object(),
            domains=["polyglot"],
            output_dir=str(tmp_path),
            run_id="unit",
            current_genid=1,
            parent_genid=0,
            root_dir=str(tmp_path),
            max_generation=2,
        )

    metadata = json.loads((tmp_path / "gen_1" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["run_eval"] is False
    assert metadata["valid_parent"] is False
