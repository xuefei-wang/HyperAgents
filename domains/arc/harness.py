import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import docker
from docker.errors import ImageNotFound
from dotenv import load_dotenv

from utils.constants import REPO_NAME
from utils.common import load_json_file
from utils.docker_utils import copy_from_container, copy_to_container, setup_logger

from domains.arc.constants import (
    ARC_AGENT_TIMEOUT_SECONDS,
    ARC_BENCHMARKING_SRC,
    ARC_DEFAULT_MANIFESTS,
    ARC_TASK_DIRS,
    ARC_WORKSPACE_UI_DIR,
)


def _load_shared_env() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    for env_path in [
        repo_root / "configs" / "providers" / ".env.shared",
        repo_root / "configs" / "providers" / ".env.haiku",
        repo_root / "configs" / "providers" / ".env.openai",
        repo_root / "configs" / "models" / "shared.env",
    ]:
        if env_path.exists():
            load_dotenv(env_path, override=True)


def _runtime_environment():
    keys = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "OPENROUTER_API_KEY",
        "DEEPSEEK_API_KEY",
        "METAGEN_ACCESS_TOKEN",
        "AWS_REGION",
        "AWS_REGION_NAME",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "HYPERAGENTS_TASK_MODEL",
        "HYPERAGENTS_POLYGLOT_MODEL",
        "HYPERAGENTS_META_MODEL",
        "HYPERAGENTS_REASONING_EFFORT",
        "OPENAI_REASONING_EFFORT",
        "REASONING_EFFORT",
    ]
    return {key: os.environ[key] for key in keys if os.environ.get(key)}


def _ensure_image(client, root_dir: str, image_name: str, logger) -> None:
    try:
        client.images.get(image_name)
        logger.info("Reusing Docker image %s", image_name)
        return
    except ImageNotFound:
        pass

    logger.info("Building Docker image %s from %s", image_name, root_dir)
    image, logs = client.images.build(path=root_dir, tag=image_name, rm=True, network_mode="host")
    del image
    for log_entry in logs:
        if "stream" in log_entry:
            logger.info(log_entry["stream"].strip())


def _start_container(client, image_name: str, container_name: str):
    return client.containers.run(
        image=image_name,
        name=container_name,
        detach=True,
        tty=True,
        stdin_open=True,
        network_mode="host",
        command="tail -f /dev/null",
    )


def _copy_agent_code(container, root_dir: str) -> None:
    copy_to_container(container, os.path.join(root_dir, "task_agent.py"), f"/{REPO_NAME}/task_agent.py", verbose=False)
    copy_to_container(container, os.path.join(root_dir, "run_task_agent.py"), f"/{REPO_NAME}/run_task_agent.py", verbose=False)
    copy_to_container(container, os.path.join(root_dir, "agent/"), f"/{REPO_NAME}/agent/", verbose=False)
    copy_to_container(container, os.path.join(root_dir, "utils/"), f"/{REPO_NAME}/utils/", verbose=False)
    copy_to_container(container, os.path.join(root_dir, "meta_agent.py"), f"/{REPO_NAME}/meta_agent.py", verbose=False)
    copy_to_container(container, os.path.join(root_dir, "run_meta_agent.py"), f"/{REPO_NAME}/run_meta_agent.py", verbose=False)
    copy_to_container(container, os.path.join(root_dir, "README.md"), f"/{REPO_NAME}/README.md", verbose=False)


def _apply_model_patches(container, model_patch_paths) -> None:
    if not model_patch_paths:
        return
    for model_patch_path in model_patch_paths:
        copy_to_container(container, model_patch_path, f"/{REPO_NAME}/parent_patch.txt", verbose=False)
        container.exec_run(f"/bin/sh -c 'patch -p1 -f < /{REPO_NAME}/parent_patch.txt'", workdir=f"/{REPO_NAME}")
        container.exec_run(f"rm /{REPO_NAME}/parent_patch.txt", workdir="/")


def _copy_optional_file(container, source_path: str, dest_path: Path) -> bool:
    try:
        copy_from_container(container, source_path, dest_path, verbose=False)
        return True
    except FileNotFoundError:
        return False


def _init_workspace_repo(workspace_dir: Path) -> str:
    subprocess.run(["git", "init"], cwd=workspace_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "benchmark-prep"], cwd=workspace_dir, check=True)
    subprocess.run(["git", "config", "user.email", "benchmark-prep@example.com"], cwd=workspace_dir, check=True)
    subprocess.run(["git", "add", "."], cwd=workspace_dir, check=True)
    subprocess.run(["git", "commit", "-m", "Initial ARC UI workspace"], cwd=workspace_dir, check=True, capture_output=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workspace_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _zero_grid_like(grid: list[list[int]]) -> list[list[int]]:
    return [[0 for _ in row] for row in grid]


def _prediction_document(payload: dict, prediction_grid: list[list[int]] | None = None) -> dict:
    grid = prediction_grid if prediction_grid is not None else _zero_grid_like(payload["test_input"])
    return {
        "benchmark": payload["benchmark"],
        "task_id": payload["task_id"],
        "pair_index": payload["pair_index"],
        "attempts": [
            {"attempt_index": 1, "prediction": [row[:] for row in grid]},
            {"attempt_index": 2, "prediction": [row[:] for row in grid]},
        ],
    }


def _missing_prediction_document(payload: dict) -> dict:
    return {
        "benchmark": payload["benchmark"],
        "task_id": payload["task_id"],
        "pair_index": payload["pair_index"],
        "missing_prediction": True,
        "attempts": [],
    }


def _grid_shape(grid: list[list[int]]) -> str:
    return f"{len(grid)} rows x {len(grid[0]) if grid else 0} cols"


def _grid_colors(grid: list[list[int]]) -> str:
    colors = sorted({value for row in grid for value in row})
    return ", ".join(str(value) for value in colors)


def _format_grid(grid: list[list[int]]) -> str:
    return "\n".join(" ".join(str(value) for value in row) for row in grid)


def _write_grid_summary(workspace_dir: Path, payload: dict) -> None:
    lines = [
        "# ARC Grid Summary",
        "",
        f"Benchmark: {payload['benchmark']}",
        f"Task id: {payload['task_id']}",
        f"Pair index: {payload['pair_index']}",
        "",
        "This file is derived only from `payload.json`. It does not contain the hidden test output.",
    ]
    for index, pair in enumerate(payload["train"]):
        lines.extend(
            [
                "",
                f"## Train Example {index}",
                "",
                f"Input shape: {_grid_shape(pair['input'])}",
                f"Input colors: {_grid_colors(pair['input'])}",
                "",
                "Input grid:",
                "```text",
                _format_grid(pair["input"]),
                "```",
                "",
                f"Output shape: {_grid_shape(pair['output'])}",
                f"Output colors: {_grid_colors(pair['output'])}",
                "",
                "Output grid:",
                "```text",
                _format_grid(pair["output"]),
                "```",
            ]
        )
    lines.extend(
        [
            "",
            "## Test Input",
            "",
            f"Shape: {_grid_shape(payload['test_input'])}",
            f"Colors: {_grid_colors(payload['test_input'])}",
            "",
            "Grid:",
            "```text",
            _format_grid(payload["test_input"]),
            "```",
        ]
    )
    (workspace_dir / "grid_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_workspace_state_files(workspace_dir: Path, payload: dict) -> None:
    workspace_state = _prediction_document(payload)
    (workspace_dir / "workspace_state.json").write_text(
        json.dumps(workspace_state, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_readme(workspace_dir: Path, payload: dict) -> None:
    text = f"""# ARC Workspace

Solve this ARC test pair using the files in this workspace. This is a
file-backed workspace with optional browser UI files. The hidden test output is
not present.

Task:
- benchmark: {payload["benchmark"]}
- task_id: {payload["task_id"]}
- pair_index: {payload["pair_index"]}

Files:
- `payload.json`: canonical public task data with train examples and test input.
- `grid_summary.md`: readable rendering of the same public grids.
- `workspace_state.json`: UI working state for two editable attempts. This is not graded directly.
- `index.html`, `css/`, and `js/`: optional visual editor.
- `tools/serve_workspace.py`: local server for the UI save/export buttons.
- `tools/validate_prediction.py`: validates `prediction.json` structure without scoring correctness.
- `prediction.json`: final graded artifact created by the agent or UI.

For browser-capable agents or manual inspection:

```bash
python3 tools/serve_workspace.py --host 127.0.0.1 --port 8765
```

Then open `http://127.0.0.1:8765/index.html`. The UI loads `payload.json` and
`workspace_state.json`; `Export Prediction` writes `prediction.json`.

For file-based agents, read `payload.json` or `grid_summary.md`, write
`prediction.json` directly, and validate it:

```bash
python3 tools/validate_prediction.py prediction.json
```

Accepted compact prediction shape:

```json
{{"attempt_1": [[1, 0], [0, 1]], "attempt_2": [[1, 0], [0, 1]]}}
```

The full `attempts` shape with metadata is also accepted. If only one answer is
credible, repeat it for both attempts. Use only integer colors `0` through `9`.
"""
    (workspace_dir / "README.md").write_text(text, encoding="utf-8")


def _prepare_workspace(entry: dict, workspace_root: Path) -> tuple[Path, str]:
    instance_id = entry["instance_id"]
    workspace_dir = workspace_root / instance_id
    if workspace_dir.exists():
        shutil.rmtree(workspace_dir)
    shutil.copytree(ARC_WORKSPACE_UI_DIR, workspace_dir)
    payload = load_json_file(entry["payload_file"])
    (workspace_dir / "payload.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_workspace_state_files(workspace_dir, payload)
    _write_grid_summary(workspace_dir, payload)
    _write_readme(workspace_dir, payload)
    base_commit = _init_workspace_repo(workspace_dir)
    return workspace_dir, base_commit


def _build_problem_statement(payload: dict) -> str:
    return f"""Solve this ARC puzzle test pair.

Benchmark: {payload["benchmark"]}
Task id: {payload["task_id"]}
Pair index: {payload["pair_index"]}

Use the public task files in `/testbed`:
- `payload.json` contains `train` examples and top-level `test_input`.
- `grid_summary.md` is a readable rendering of the same public grids.
- The hidden test output is not included.

Infer the train input -> train output transformation, apply it to `test_input`,
and leave `/testbed/prediction.json` with exactly two attempts. Use integer
colors 0 through 9. If you only have one credible answer, repeat it for both
attempts. Validate with `python3 tools/validate_prediction.py prediction.json`.
Do not modify benchmark files or hardcode task IDs."""


def _normalize_prediction(prediction: dict, entry: dict) -> dict:
    prediction = dict(prediction)
    prediction["benchmark"] = entry["benchmark"]
    prediction["task_id"] = entry["task_id"]
    prediction["pair_index"] = entry["pair_index"]
    if "attempts" in prediction:
        attempts = prediction["attempts"]
        if not isinstance(attempts, list) or not attempts:
            raise ValueError("attempts must be a non-empty list")
        normalized_attempts = []
        for attempt_index, attempt in enumerate(attempts[:2], start=1):
            if isinstance(attempt, dict):
                grid = attempt.get("prediction", attempt.get("output", attempt.get("answer")))
            else:
                grid = attempt
            if grid is None:
                raise ValueError("each attempt must contain prediction")
            _validate_prediction_grid(grid, f"attempt {attempt_index} prediction")
            normalized_attempt = {"attempt_index": attempt_index, "prediction": grid}
            normalized_attempts.append(normalized_attempt)
        if len(normalized_attempts) == 1:
            duplicate = dict(normalized_attempts[0])
            duplicate["attempt_index"] = 2
            normalized_attempts.append(duplicate)
        return {
            "benchmark": prediction["benchmark"],
            "task_id": prediction["task_id"],
            "pair_index": prediction["pair_index"],
            "attempts": normalized_attempts,
        }
    if "attempt_1" in prediction or "attempt_2" in prediction:
        normalized_attempts = []
        for attempt_index, key in enumerate(["attempt_1", "attempt_2"], start=1):
            if key not in prediction:
                continue
            _validate_prediction_grid(prediction[key], key)
            normalized_attempts.append(
                {"attempt_index": attempt_index, "prediction": prediction[key]}
            )
        if not normalized_attempts:
            raise ValueError("prediction.json must contain at least attempt_1 or attempt_2")
        if len(normalized_attempts) == 1:
            duplicate = dict(normalized_attempts[0])
            duplicate["attempt_index"] = 2
            normalized_attempts.append(duplicate)
        prediction["attempts"] = normalized_attempts
        return prediction
    if "prediction" in prediction:
        _validate_prediction_grid(prediction["prediction"], "prediction")
        prediction["attempts"] = [
            {"attempt_index": 1, "prediction": prediction.pop("prediction")},
        ]
        duplicate = dict(prediction["attempts"][0])
        duplicate["attempt_index"] = 2
        prediction["attempts"].append(duplicate)
        return prediction
    raise ValueError("prediction.json must contain either attempts or prediction")


def _validate_prediction_grid(grid, label: str) -> None:
    if not isinstance(grid, list) or not grid:
        raise ValueError(f"{label} must be a non-empty 2D list")
    if not all(isinstance(row, list) and row for row in grid):
        raise ValueError(f"{label} must contain non-empty rows")
    width = len(grid[0])
    if len(grid) > 30 or width > 30:
        raise ValueError(f"{label} dimensions must be at most 30x30")
    for row_index, row in enumerate(grid):
        if len(row) != width:
            raise ValueError(f"{label} row {row_index} has inconsistent width")
        for col_index, value in enumerate(row):
            if not isinstance(value, int) or value < 0 or value > 9:
                raise ValueError(f"{label}[{row_index}][{col_index}] must be an integer 0..9")


def _load_json_object_text(text: str) -> dict:
    decoder = json.JSONDecoder()
    payload, _ = decoder.raw_decode(text.lstrip())
    if not isinstance(payload, dict):
        raise ValueError("prediction payload must be a JSON object")
    return payload


def _load_prediction_file(path: Path) -> dict:
    return _load_json_object_text(path.read_text(encoding="utf-8"))


def _load_prediction_from_patch(patch_text: str) -> dict:
    in_prediction = False
    lines = []
    for line in patch_text.splitlines():
        if line.startswith("diff --git "):
            if in_prediction:
                break
            in_prediction = line.strip() == "diff --git a/prediction.json b/prediction.json"
            continue
        if not in_prediction:
            continue
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            lines.append(line[1:])

    if not lines:
        raise ValueError("prediction.json was not found in patch")
    return _load_json_object_text("\n".join(lines))


def _write_missing_prediction(entry: dict, prediction_path: Path) -> None:
    task_payload = load_json_file(entry["payload_file"])
    payload = _missing_prediction_document(task_payload)
    prediction_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _collect_patch(container, base_commit: str) -> str:
    cmd = [
        "python",
        "-c",
        (
            "from utils.git_utils import diff_versus_commit; "
            f"print(diff_versus_commit('/testbed', '{base_commit}'), end='')"
        ),
    ]
    result = container.exec_run(cmd, workdir=f"/{REPO_NAME}")
    return result.output.decode("utf-8", errors="replace")


def process_entry(entry, out_dname: Path, model_name_or_path: str, model_patch_paths, root_dir: str):
    instance_id = entry["instance_id"]
    out_fname = out_dname / f"{instance_id}.json"
    patch_fname = out_dname / f"{instance_id}.patch.diff"
    chat_history_file = out_dname / f"{instance_id}.md"
    docker_log_file = out_dname / f"{instance_id}_docker.log"
    host_prediction_file = out_dname / "predictions" / f"{instance_id}.json"

    if out_fname.exists():
        with open(out_fname, "r", encoding="utf-8") as handle:
            return json.load(handle)

    container = None
    agent_status = "error"
    patch_text = ""
    started_at = datetime.datetime.now(datetime.timezone.utc)

    try:
        _load_shared_env()
        client = docker.from_env()
        logger = setup_logger(str(docker_log_file))
        _ensure_image(client, root_dir, REPO_NAME, logger)

        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        container_name = f"{REPO_NAME}-arc-{instance_id[:48]}-{run_id}"
        logger.info("Starting ARC task container %s", container_name)
        container = _start_container(client, REPO_NAME, container_name)

        logger.info("Copying agent code into task container")
        _copy_agent_code(container, root_dir)
        _apply_model_patches(container, model_patch_paths)

        workspace_root = out_dname / "workspaces"
        workspace_dir, base_commit = _prepare_workspace(entry, workspace_root)
        copy_to_container(container, workspace_dir, "/testbed", verbose=False)

        chat_history_container = f"/tmp/{instance_id}.md"
        agent_model = os.getenv("HYPERAGENTS_TASK_MODEL", "openai/gpt-5.4-mini")
        problem_statement = _build_problem_statement(load_json_file(entry["payload_file"]))
        cmd = [
            "timeout",
            str(ARC_AGENT_TIMEOUT_SECONDS),
            "python",
            f"/{REPO_NAME}/run_task_agent.py",
            "--domain",
            "arc_ui",
            "--problem_statement",
            problem_statement,
            "--git_dir",
            "/testbed/",
            "--chat_history_file",
            chat_history_container,
            "--base_commit",
            base_commit,
            "--outdir",
            "/tmp/",
            "--model",
            agent_model,
        ]
        logger.info("Running ARC task agent with timeout=%s", ARC_AGENT_TIMEOUT_SECONDS)
        exec_result = container.exec_run(cmd, environment=_runtime_environment(), workdir="/testbed/")
        exit_code = exec_result.exit_code or 0
        logger.info("Task agent finished with exit_code=%s", exit_code)

        if exit_code == 0:
            agent_status = "completed"
        elif exit_code == 124:
            agent_status = "timeout"
        else:
            agent_status = "error"

        patch_text = _collect_patch(container, base_commit)
        patch_fname.write_text(patch_text, encoding="utf-8")
        _copy_optional_file(container, chat_history_container, chat_history_file)

        host_prediction_file.parent.mkdir(parents=True, exist_ok=True)
        prediction_present = _copy_optional_file(container, "/testbed/prediction.json", host_prediction_file)
        prediction_error = None
        prediction = None
        if prediction_present:
            try:
                prediction = _normalize_prediction(_load_prediction_file(host_prediction_file), entry)
            except Exception as exc:
                prediction_error = str(exc)
        if prediction is None:
            try:
                prediction = _normalize_prediction(_load_prediction_from_patch(patch_text), entry)
                prediction_error = None
            except Exception as exc:
                if prediction_error is None:
                    prediction_error = str(exc)

        if prediction is not None:
            host_prediction_file.write_text(json.dumps(prediction, indent=2) + "\n", encoding="utf-8")
            prediction_present = True
        else:
            if not prediction_present:
                prediction_error = None
            _write_missing_prediction(entry, host_prediction_file)

        ended_at = datetime.datetime.now(datetime.timezone.utc)
        result = {
            "instance_id": instance_id,
            "benchmark": entry["benchmark"],
            "task_id": entry["task_id"],
            "pair_index": entry["pair_index"],
            "model_name_or_path": model_name_or_path,
            "agent_status": agent_status,
            "agent_exit_code": exit_code,
            "patch_file": str(patch_fname),
            "patch_bytes": len(patch_text.encode("utf-8")),
            "workspace_repo": str(workspace_dir),
            "prediction_file": str(host_prediction_file) if prediction_present and prediction_error is None else None,
            "prediction_error": prediction_error,
            "chat_history_file": str(chat_history_file),
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "duration_seconds": (ended_at - started_at).total_seconds(),
            "success": True,
        }
        out_fname.write_text(json.dumps(result, indent=4), encoding="utf-8")
        return result

    except Exception as e:
        ended_at = datetime.datetime.now(datetime.timezone.utc)
        try:
            host_prediction_file.parent.mkdir(parents=True, exist_ok=True)
            _write_missing_prediction(entry, host_prediction_file)
        except Exception:
            pass
        result = {
            "instance_id": instance_id,
            "benchmark": entry["benchmark"],
            "task_id": entry["task_id"],
            "pair_index": entry["pair_index"],
            "model_name_or_path": model_name_or_path,
            "agent_status": agent_status,
            "patch_file": str(patch_fname),
            "patch_bytes": len(patch_text.encode("utf-8")),
            "prediction_file": None,
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "duration_seconds": (ended_at - started_at).total_seconds(),
            "success": False,
            "error": str(e),
        }
        out_fname.write_text(json.dumps(result, indent=4), encoding="utf-8")
        return result
    finally:
        if container is not None:
            try:
                container.remove(force=True)
            except Exception:
                pass


def _resolve_manifest_payload_file(path_value: str, manifest_dir: Path) -> Path:
    path = Path(path_value)
    if path.exists():
        return path
    if path.is_absolute():
        candidate = manifest_dir / path.name
        if candidate.exists():
            return candidate
    else:
        candidate = manifest_dir / path
        if candidate.exists():
            return candidate
    return path


def _resolve_manifest_source_file(benchmark: str, task_id: str, path_value: str) -> Path:
    path = Path(path_value)
    if path.exists():
        return path
    return ARC_TASK_DIRS[benchmark] / f"{task_id}.json"


def _select_manifest_items(manifest_items: list[dict], num_samples: int) -> list[dict]:
    """Select entries for the first N unique ARC task IDs.

    ARC's official scorer reports one score per unique task ID. Some tasks have
    multiple test pairs, so positive num_samples must not truncate the flat
    pair list directly or the official denominator will be smaller than the
    requested sample count.
    """
    if num_samples <= 0:
        return manifest_items

    selected = []
    seen_task_ids = set()
    for item in manifest_items:
        task_id = item["task_id"]
        if task_id not in seen_task_ids:
            if len(seen_task_ids) >= num_samples:
                break
            seen_task_ids.add(task_id)
        selected.append(item)
    return selected


def selected_entry_count(manifest_path: Path, num_samples: int) -> int:
    manifest = load_json_file(str(manifest_path))
    return len(_select_manifest_items(manifest["payloads"], num_samples))


def _load_entries(benchmark: str, manifest_path: Path, num_samples: int) -> list[dict]:
    manifest = load_json_file(str(manifest_path))
    manifest_dir = manifest_path.parent
    entries = []
    for item in _select_manifest_items(manifest["payloads"], num_samples):
        instance_id = f"{item['task_id']}_pair{item['pair_index']}"
        payload_file = _resolve_manifest_payload_file(item["payload_file"], manifest_dir)
        source_file = _resolve_manifest_source_file(benchmark, item["task_id"], item["source_file"])
        entries.append(
            {
                "instance_id": instance_id,
                "benchmark": benchmark,
                "task_id": item["task_id"],
                "pair_index": int(item["pair_index"]),
                "payload_file": str(payload_file),
                "source_file": str(source_file),
            }
        )
    return entries


def _run_converter(predictions_dir: Path, submissions_dir: Path, manifest_path: Path | None) -> None:
    wrapper = Path(__file__).resolve().parents[4] / "scripts" / "convert_arc_workspace_predictions.py"
    cmd = [
        sys.executable,
        str(wrapper),
        "--predictions-dir",
        str(predictions_dir),
        "--output-submission-dir",
        str(submissions_dir),
    ]
    if manifest_path is not None:
        cmd.extend(["--manifest", str(manifest_path)])
    subprocess.run(cmd, check=True)


def _attempt_sort_key(name: str) -> tuple[int, str]:
    if name.startswith("attempt_"):
        try:
            return (int(name.rsplit("_", 1)[1]), name)
        except ValueError:
            pass
    return (10_000, name)


def _score_submission_pair(pair_submission: object, expected_output: object) -> tuple[bool, int, int, list[dict]]:
    if not isinstance(pair_submission, dict):
        return False, 0, 0, []

    submitted_trials = 0
    correct_trials = 0
    attempts = []
    for attempt_name in sorted(pair_submission, key=_attempt_sort_key):
        attempt_payload = pair_submission[attempt_name]
        if not isinstance(attempt_payload, dict) or "answer" not in attempt_payload:
            continue
        submitted_trials += 1
        is_correct = attempt_payload["answer"] == expected_output
        correct_trials += int(is_correct)
        attempts.append(
            {
                "attempt": attempt_name,
                "correct": is_correct,
            }
        )
    return correct_trials > 0, submitted_trials, correct_trials, attempts


def _run_local_scorer(benchmark: str, submissions_dir: Path, results_dir: Path) -> None:
    task_dir = ARC_TASK_DIRS[benchmark]
    task_details = []
    task_score = 0
    total_attempts = 0
    total_correct_trials = 0

    for submission_file in sorted(submissions_dir.glob("*.json")):
        task_id = submission_file.stem
        task_file = task_dir / f"{task_id}.json"
        task_payload = load_json_file(task_file)
        submission = load_json_file(submission_file)
        test_cases = task_payload.get("test", [])
        pair_details = []
        task_correct = isinstance(submission, list) and len(submission) >= len(test_cases)

        for pair_index, test_case in enumerate(test_cases):
            expected_output = test_case.get("output")
            pair_submission = submission[pair_index] if isinstance(submission, list) and pair_index < len(submission) else None
            pair_correct, submitted_trials, correct_trials, attempts = _score_submission_pair(
                pair_submission,
                expected_output,
            )
            total_attempts += submitted_trials
            total_correct_trials += correct_trials
            task_correct = task_correct and pair_correct
            pair_details.append(
                {
                    "pair_index": pair_index,
                    "correct": pair_correct,
                    "submitted_trials": submitted_trials,
                    "correct_trials": correct_trials,
                    "attempts": attempts,
                }
            )

        task_score += int(task_correct)
        task_details.append(
            {
                "task_id": task_id,
                "correct": task_correct,
                "pairs": pair_details,
            }
        )

    results = {
        "score": task_score,
        "total_tasks": len(task_details),
        "total_attempts": total_attempts,
        "total_submitted_trials": total_attempts,
        "total_correct_trials": total_correct_trials,
        "scorer": "local_exact_match_fallback",
        "tasks": task_details,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


def _run_official_scorer(benchmark: str, submissions_dir: Path, results_dir: Path) -> None:
    if not ARC_BENCHMARKING_SRC.is_dir():
        _run_local_scorer(benchmark, submissions_dir, results_dir)
        return

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in [str(ARC_BENCHMARKING_SRC), env.get("PYTHONPATH", "")] if part
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "arc_agi_benchmarking.scoring",
            "--task_dir",
            str(ARC_TASK_DIRS[benchmark]),
            "--submission_dir",
            str(submissions_dir),
            "--results_dir",
            str(results_dir),
        ],
        check=True,
        env=env,
    )


def harness(
    benchmark: str,
    test_task_list=None,
    num_samples=-1,
    max_workers=4,
    model_name_or_path=None,
    model_patch_paths=None,
    pred_dname="./outputs",
    output_dir="./outputs",
    root_dir=None,
    manifest_path=None,
):
    del test_task_list, output_dir
    _load_shared_env()
    if benchmark not in {"arc1", "arc2"}:
        raise ValueError("benchmark must be arc1 or arc2")
    manifest_path = Path(manifest_path) if manifest_path else ARC_DEFAULT_MANIFESTS[benchmark]

    if model_name_or_path is None:
        model_name_or_path = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dname = Path(pred_dname) / f"{model_name_or_path}_0"
    out_dname.mkdir(parents=True, exist_ok=True)

    root_dir = root_dir or str(Path(__file__).resolve().parents[2])
    entries = _load_entries(benchmark, manifest_path, num_samples)

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_entry = {
            executor.submit(process_entry, entry, out_dname, model_name_or_path, model_patch_paths, root_dir): entry
            for entry in entries
        }
        for future in as_completed(future_to_entry):
            results.append(future.result())

    (out_dname / "task_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    _run_converter(out_dname / "predictions", out_dname / "submissions", manifest_path if num_samples <= 0 else None)
    official_eval_dir = out_dname / "official_eval"
    official_eval_dir.mkdir(parents=True, exist_ok=True)
    _run_official_scorer(benchmark, out_dname / "submissions", official_eval_dir)
    return [out_dname]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["arc1", "arc2"], required=True)
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--max_workers", type=int, default=2)
    parser.add_argument("--model_name_or_path", type=str, default="eval_run")
    parser.add_argument("--model_patch_paths", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--manifest_path", type=str, default=None)
    args = parser.parse_args()

    model_patch_paths = args.model_patch_paths.split(",") if args.model_patch_paths else None
    harness(
        benchmark=args.benchmark,
        num_samples=args.num_samples,
        max_workers=args.max_workers,
        model_name_or_path=args.model_name_or_path,
        model_patch_paths=model_patch_paths,
        pred_dname=args.output_dir,
        output_dir=args.output_dir,
        root_dir="./",
        manifest_path=args.manifest_path,
    )


if __name__ == "__main__":
    main()
