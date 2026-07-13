import sys
import types

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

    with pytest.raises(RuntimeError, match="Failed to prune copied eval tree"):
        generate_loop._prune_copied_eval_tree(container, "/container/out/tree", current_genid=1)
