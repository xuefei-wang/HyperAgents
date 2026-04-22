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
from utils.docker_utils import copy_from_container, copy_to_container, safe_log, setup_logger

from domains.swebench_pro.constants import (
    SWEBENCH_PRO_AGENT_TIMEOUT_SECONDS,
    SWEBENCH_PRO_DATASET_PATH,
    SWEBENCH_PRO_DEFAULT_TASK_MAP,
    SWEBENCH_PRO_EVAL_DOCKERHUB_USERNAME,
    SWEBENCH_PRO_REPO_CACHE_DIR,
    SWEBENCH_PRO_SOURCE_DIR,
)


def _load_shared_env() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    env_paths = [
        repo_root / "configs" / "providers" / ".env.shared",
        repo_root / "configs" / "providers" / ".env.haiku",
        repo_root / "configs" / "providers" / ".env.openai",
        repo_root / "configs" / "models" / "shared.env",
    ]
    for env_path in env_paths:
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
    ]
    return {key: os.environ[key] for key in keys if os.environ.get(key)}


def _load_dataset_rows(dataset_path: str) -> dict[str, dict]:
    rows = {}
    with open(dataset_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[row["instance_id"]] = row
    return rows


def _load_task_ids(test_task_list=None, num_samples=-1):
    if test_task_list:
        task_ids = list(test_task_list)
    else:
        task_map = load_json_file(str(SWEBENCH_PRO_DEFAULT_TASK_MAP))
        task_ids = [task["task_id"] for task in task_map["tasks"]]
    if num_samples > 0:
        task_ids = task_ids[:num_samples]
    return task_ids


def _ensure_task_repo(instance_id: str, workspace_root: Path) -> Path:
    source_repo = SWEBENCH_PRO_REPO_CACHE_DIR / instance_id
    if not (source_repo / ".git").exists():
        raise FileNotFoundError(f"Cached SWE-bench Pro repo missing: {source_repo}")

    task_root = workspace_root / instance_id
    repo_dest = task_root / "repo"
    if task_root.exists():
        shutil.rmtree(task_root)
    task_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_repo, repo_dest, symlinks=True)
    return repo_dest


def _ensure_image(client, root_dir: str, image_name: str, logger) -> None:
    try:
        client.images.get(image_name)
        logger.info("Reusing Docker image %s", image_name)
        return
    except ImageNotFound:
        pass

    logger.info("Building Docker image %s from %s", image_name, root_dir)
    image, logs = client.images.build(
        path=root_dir,
        tag=image_name,
        rm=True,
        network_mode="host",
    )
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
        container.exec_run(
            f"/bin/sh -c 'patch -p1 -f < /{REPO_NAME}/parent_patch.txt'",
            workdir=f"/{REPO_NAME}",
        )
        container.exec_run(f"rm /{REPO_NAME}/parent_patch.txt", workdir="/")


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


def _copy_optional_file(container, source_path: str, dest_path: Path) -> None:
    try:
        copy_from_container(container, source_path, dest_path, verbose=False)
    except FileNotFoundError:
        return


def process_entry(entry, out_dname: Path, model_name_or_path: str, model_patch_paths, root_dir: str):
    instance_id = entry["instance_id"]
    base_commit = entry["base_commit"]
    out_fname = out_dname / f"{instance_id}.json"
    patch_fname = out_dname / f"{instance_id}.patch.diff"
    chat_history_file = out_dname / f"{instance_id}.md"
    docker_log_file = out_dname / f"{instance_id}_docker.log"

    if out_fname.exists():
        with open(out_fname, "r", encoding="utf-8") as handle:
            return json.load(handle)

    client = None
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
        container_name = f"{REPO_NAME}-swebench-pro-{instance_id[:40]}-{run_id}"
        logger.info("Starting task container %s", container_name)
        container = _start_container(client, REPO_NAME, container_name)

        logger.info("Copying agent code into task container")
        _copy_agent_code(container, root_dir)
        _apply_model_patches(container, model_patch_paths)

        task_workspace_root = out_dname / "workspaces"
        logger.info("Creating host task workspace under %s", task_workspace_root)
        repo_path = _ensure_task_repo(instance_id, task_workspace_root)
        logger.info("Copying task repo to /testbed")
        copy_to_container(container, repo_path, "/testbed", verbose=False)

        chat_history_container = f"/tmp/{instance_id}.md"
        runtime_env = _runtime_environment()
        agent_model = os.getenv("HYPERAGENTS_TASK_MODEL", "openai/gpt-5.4-mini")
        cmd = [
            "timeout",
            str(SWEBENCH_PRO_AGENT_TIMEOUT_SECONDS),
            "python",
            f"/{REPO_NAME}/run_task_agent.py",
            "--domain",
            "swebench_pro",
            "--problem_statement",
            entry["problem_statement"],
            "--requirements",
            entry.get("requirements", "") or "",
            "--interface",
            entry.get("interface", "") or "",
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
        logger.info("Running task agent with timeout=%s", SWEBENCH_PRO_AGENT_TIMEOUT_SECONDS)
        exec_result = container.exec_run(cmd, environment=runtime_env, workdir="/testbed/")
        exit_code = exec_result.exit_code or 0
        logger.info("Task agent finished with exit_code=%s", exit_code)

        if exit_code == 0:
            agent_status = "completed"
        elif exit_code == 124:
            agent_status = "timeout"
        else:
            agent_status = "error"

        logger.info("Collecting patch from /testbed")
        patch_text = _collect_patch(container, base_commit)
        patch_fname.write_text(patch_text, encoding="utf-8")
        _copy_optional_file(container, chat_history_container, chat_history_file)

        ended_at = datetime.datetime.now(datetime.timezone.utc)
        result = {
            "instance_id": instance_id,
            "model_name_or_path": model_name_or_path,
            "agent_status": agent_status,
            "agent_exit_code": exit_code,
            "patch_file": str(patch_fname),
            "patch_bytes": len(patch_text.encode("utf-8")),
            "workspace_repo": str(repo_path),
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
        result = {
            "instance_id": instance_id,
            "model_name_or_path": model_name_or_path,
            "agent_status": agent_status,
            "patch_file": str(patch_fname),
            "patch_bytes": len(patch_text.encode("utf-8")),
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


def _write_patch_bundle(results, output_path: Path, prefix: str) -> None:
    patches = []
    for result in results:
        patch_path = Path(result["patch_file"])
        patch_text = patch_path.read_text(encoding="utf-8") if patch_path.exists() else ""
        patches.append(
            {
                "instance_id": result["instance_id"],
                "patch": patch_text,
                "prefix": prefix,
            }
        )
    output_path.write_text(json.dumps(patches, indent=2), encoding="utf-8")


def _run_official_eval(predictions_file: Path, eval_output_dir: Path, num_workers: int) -> None:
    wrapper = Path(__file__).resolve().parents[4] / "scripts" / "run_swebench_pro_eval.py"
    cmd = [
        sys.executable,
        str(wrapper),
        "--patch-path",
        str(predictions_file),
        "--output-dir",
        str(eval_output_dir),
        "--source-dir",
        str(SWEBENCH_PRO_SOURCE_DIR),
        "--raw-sample-path",
        str(SWEBENCH_PRO_DATASET_PATH),
        "--scripts-dir",
        str(SWEBENCH_PRO_SOURCE_DIR / "run_scripts"),
        "--dockerhub-username",
        SWEBENCH_PRO_EVAL_DOCKERHUB_USERNAME,
        "--use-local-docker",
        "--num-workers",
        str(num_workers),
        "--redo",
    ]
    print(f"swebench_pro harness: running official eval {cmd}")
    subprocess.run(cmd, check=True)


def harness(
    test_task_list=None,
    num_samples=-1,
    max_workers=4,
    model_name_or_path=None,
    model_patch_paths=None,
    pred_dname="./outputs",
    output_dir="./outputs",
    root_dir=None,
):
    _load_shared_env()
    rows = _load_dataset_rows(str(SWEBENCH_PRO_DATASET_PATH))
    task_ids = _load_task_ids(test_task_list=test_task_list, num_samples=num_samples)
    entries = [rows[task_id] for task_id in task_ids]

    if model_name_or_path is None:
        model_name_or_path = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dname = Path(pred_dname) / f"{model_name_or_path}_0"
    out_dname.mkdir(parents=True, exist_ok=True)

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_entry = {
            executor.submit(process_entry, entry, out_dname, model_name_or_path, model_patch_paths, root_dir): entry
            for entry in entries
        }
        for future in as_completed(future_to_entry):
            results.append(future.result())

    task_results_file = out_dname / "task_results.json"
    task_results_file.write_text(json.dumps(results, indent=2), encoding="utf-8")

    predictions_file = out_dname / "eval_run_patches.json"
    _write_patch_bundle(results, predictions_file, prefix="eval_run")

    official_eval_dir = out_dname / "official_eval"
    official_eval_dir.mkdir(parents=True, exist_ok=True)
    _run_official_eval(predictions_file, official_eval_dir, num_workers=4)
    return [out_dname]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--max_workers", type=int, default=2)
    parser.add_argument("--model_name_or_path", type=str, default="eval_run")
    parser.add_argument("--model_patch_paths", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    args = parser.parse_args()

    model_patch_paths = args.model_patch_paths.split(",") if args.model_patch_paths else None
    task_map = load_json_file(str(SWEBENCH_PRO_DEFAULT_TASK_MAP))
    task_ids = [task["task_id"] for task in task_map["tasks"]]
    harness(
        test_task_list=task_ids,
        num_samples=args.num_samples,
        max_workers=args.max_workers,
        model_name_or_path=args.model_name_or_path,
        model_patch_paths=model_patch_paths,
        pred_dname=args.output_dir,
        output_dir=args.output_dir,
        root_dir="./",
    )


if __name__ == "__main__":
    main()
