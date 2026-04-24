import argparse
import datetime
import json
import os
import tempfile
from enum import Enum
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import docker
from datasets import load_dataset
from dotenv import load_dotenv

from utils.constants import REPO_NAME
from utils.common import load_json_file
from domains.polyglot.testrepo_prompt import get_test_description
from domains.polyglot.test_spec import make_test_spec
from domains.polyglot.docker_build import build_env_images, build_container, cleanup_container
from domains.polyglot.constants import (
    MAP_REPO_VERSION_TO_SPECS,
    TEST_COMMANDS,
    POLYGLOT_METADATA_PATH,
    POLYGLOT_SOURCE_DIR,
    POLYGLOT_TASK_MAP_DIR,
)
from domains.polyglot.git_utils import filter_patch_by_files, remove_patch_by_files
from domains.polyglot.utils import (
    copy_to_container,
    copy_from_container,
    log_container_output,
    remove_existing_container,
    safe_log,
    setup_logger,
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


def _collect_runtime_env(names):
    env_vars = {}
    for name in names:
        value = os.getenv(name)
        if value:
            env_vars[name] = value
    return env_vars


def _normalize_dataset_repo_paths(dataset):
    """Make metadata-generated repo paths portable across local and remote workspaces."""
    normalized = []
    for entry in dataset:
        entry = dict(entry)
        repo = Path(entry.get("repo", ""))
        if not repo.exists():
            language = entry.get("language")
            task_name = entry.get("task_name")
            candidate = POLYGLOT_SOURCE_DIR / language / "exercises" / "practice" / task_name
            if candidate.exists():
                entry["repo"] = str(candidate)
        normalized.append(entry)
    return normalized


def get_eval_script(commands):
    return "\n".join(["#!/bin/bash", "set -uxo pipefail"] + commands) + "\n"

def process_entry(entry, out_dname, model_name_or_path, model_patch_paths, root_dir):
    """
    Process a single dataset entry. This function encapsulates the main processing logic
    for each entry to make it suitable for parallel execution.
    """
    instance_id = entry['instance_id']
    problem_statement = entry['problem_statement']
    base_commit = entry['base_commit']
    chat_history_file = out_dname / (instance_id + ".md")
    out_fname = out_dname / (instance_id + ".json")
    eval_file = out_dname / f"{instance_id}_eval.sh"
    eval_result_file = out_dname / f"{instance_id}_eval.md"

    # Skip if output result already exists
    if out_fname.exists():
        print(f"Skipping existing entry {instance_id}")
        with open(out_fname) as f:
            result = json.loads(f.read())
        return result

    try:
        _load_shared_env()
        # Create and start the Docker container
        client = docker.from_env()
        run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        # Set up thread-specific logger
        logger = setup_logger(str(out_dname / f"{instance_id}_docker.log"))
        nocache = True
        test_spec = make_test_spec(entry)
        # Remove any existing container with the same name
        container_name = test_spec.get_instance_container_name(run_id)
        remove_existing_container(client, container_name)
        # Now create and start the container
        container = build_container(test_spec, client, run_id, logger, nocache, force_rebuild=False)
        container.start()

        # Copy the necessary files and requirements to the container
        root_dir = root_dir if root_dir is not None else "./"
        task_agent_requirements = os.path.join(root_dir, 'domains/polyglot/task_agent_requirements.txt')
        copy_to_container(container, os.path.join(root_dir, 'task_agent.py'), f'/{REPO_NAME}/task_agent.py')
        copy_to_container(container, os.path.join(root_dir, 'run_task_agent.py'), f'/{REPO_NAME}/run_task_agent.py')
        copy_to_container(container, task_agent_requirements, f'/{REPO_NAME}/requirements.txt')
        copy_to_container(container, os.path.join(root_dir, 'agent/'), f'/{REPO_NAME}/agent/')
        copy_to_container(container, os.path.join(root_dir, 'utils/'), f'/{REPO_NAME}/utils/')
        copy_to_container(container, os.path.join(root_dir, 'meta_agent.py'), f'/{REPO_NAME}/meta_agent.py')
        copy_to_container(container, os.path.join(root_dir, 'run_meta_agent.py'), f'/{REPO_NAME}/run_meta_agent.py')
        copy_to_container(container, os.path.join(root_dir, 'README.md'), f'/{REPO_NAME}/README.md')
        chat_history_file_container = f'/{REPO_NAME}/{chat_history_file.name}'

        # See the checked repo
        exec_result = container.exec_run("ls -R /testbed", workdir='/')
        log_container_output(exec_result)

        # Get test description
        eval_cmd = MAP_REPO_VERSION_TO_SPECS[entry['language']]['test_cmd']
        test_description = get_test_description(eval_cmd, polyglot=True)

        # Apply model patch
        if model_patch_paths:
            safe_log("Applying model patches")
            for model_patch_path in model_patch_paths:
                copy_to_container(container, model_patch_path, f'/{REPO_NAME}/parent_patch.txt')
                exec_result = container.exec_run(f"/bin/sh -c 'patch -p1 -f < /{REPO_NAME}/parent_patch.txt'", workdir=f'/{REPO_NAME}')
                log_container_output(exec_result)
                exec_result = container.exec_run(f"rm /{REPO_NAME}/parent_patch.txt", workdir='/')
                log_container_output(exec_result)

        # Install this repo requirements
        safe_log("Installing more requirements")
        exec_result = container.exec_run(f"python -m pip install -r /{REPO_NAME}/requirements.txt", workdir='/')
        log_container_output(exec_result)

        # Run the agent
        env_vars = _collect_runtime_env([
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            "OPENROUTER_API_KEY",
            "DEEPSEEK_API_KEY",
            "METAGEN_ACCESS_TOKEN",
            "AWS_REGION",
            "AWS_REGION_NAME",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "HYPERAGENTS_TASK_MODEL",
            "HYPERAGENTS_POLYGLOT_MODEL",
            "HYPERAGENTS_REASONING_EFFORT",
            "OPENAI_REASONING_EFFORT",
            "REASONING_EFFORT",
        ])
        safe_log("Running the agent")
        agent_model = os.getenv(
            "HYPERAGENTS_POLYGLOT_MODEL",
            os.getenv("HYPERAGENTS_TASK_MODEL", "openai/gpt-5.4-mini"),
        )
        cmd = [
            "timeout", "600",  # 10 min timeout
            "python", f"/{REPO_NAME}/run_task_agent.py",
            "--problem_statement", problem_statement,
            "--git_dir", "/testbed/",
            "--chat_history_file", chat_history_file_container,
            "--base_commit", base_commit,
            "--outdir", f"/{REPO_NAME}/",
            "--test_description", test_description,
            "--language", entry['language'],
            "--model", agent_model,
        ]
        exec_result = container.exec_run(cmd, environment=env_vars, workdir='/testbed/')
        log_container_output(exec_result)

        # Copy output files back to host
        logger.info("Copying output files back to host")
        copy_from_container(container, chat_history_file_container, chat_history_file)
        # Additional chat history files
        exec_result = container.exec_run(f"find /{REPO_NAME}/ -name '{instance_id}_*.md'", workdir='/')
        chat_history_files_container = exec_result.output.decode().split()
        for chat_history_file_container in chat_history_files_container:
            chat_history_file = out_dname / Path(chat_history_file_container).name
            copy_from_container(container, chat_history_file_container, chat_history_file)

        # Get model_patch
        model_patch = ''
        logger.info("Getting model_patch")
        exec_result = container.exec_run(f"cat /{REPO_NAME}/model_patch.diff")
        log_container_output(exec_result)
        model_patch = exec_result.output.decode()

        # Additional proposed model patches
        proposed_model_patches = []

        # Directly do eval
        eval_result = ''
        if not model_patch:
            eval_result = 'empty_patch'
            result = {
            "instance_id": instance_id,
            "model_name_or_path": model_name_or_path,
            "model_patch": model_patch,
            'proposed_model_patches': proposed_model_patches,
            "eval_result": eval_result,
            "success": True
        }
            out_fname.write_text(json.dumps(result, indent=4))
            return {"success": True, "instance_id": instance_id, "eval_result": eval_result}


        solution_files = entry['files']['solution']
        solution_paths = " ".join(solution_files)

        # Preserve only the benchmark-defined solution artifacts across the
        # transition from base_commit to test_commit. Include untracked files so
        # test-writing tasks can create their solution file from scratch.
        exec_result = container.exec_run(f"git -C /testbed stash push --include-untracked -- {solution_paths}", workdir='/')
        log_container_output(exec_result)
        exec_result = container.exec_run(f"git -C /testbed reset --hard {entry['test_commit']}", workdir='/')
        log_container_output(exec_result)
        exec_result = container.exec_run(f"git -C /testbed clean -fd", workdir='/')
        log_container_output(exec_result)
        exec_result = container.exec_run("git -C /testbed stash list", workdir='/')
        log_container_output(exec_result)
        if exec_result.output.decode().strip():
            exec_result = container.exec_run(f"rm -rf {' '.join('/testbed/' + path for path in solution_files)}", workdir='/')
            log_container_output(exec_result)
            exec_result = container.exec_run("git -C /testbed stash pop", workdir='/')
            log_container_output(exec_result)
        else:
            safe_log("No solution-file changes were stashed before evaluation")

        safe_log("Running the eval")
        language = entry['language']
        test_command = TEST_COMMANDS[language]

        eval_file.write_text(get_eval_script(test_command))

        copy_to_container(container, eval_file, '/testbed/eval.sh')
        exec_result = container.exec_run("ls -R /testbed", workdir='/')
        log_container_output(exec_result)
        exec_result = container.exec_run("chmod +x /testbed/eval.sh", workdir='/')
        log_container_output(exec_result)

        exec_result = container.exec_run("timeout 120 ./eval.sh", workdir='/testbed')
        log_container_output(exec_result, raise_error=False)
        eval_result_file.write_text(exec_result.output.decode())
        if exec_result.exit_code == 0:
            eval_result = 'resolved'
        else:
            eval_result = 'unresolved'

        # Write result to file
        result = {
            "instance_id": instance_id,
            "model_name_or_path": model_name_or_path,
            "model_patch": model_patch,
            'proposed_model_patches': proposed_model_patches,
            "eval_result": eval_result,
            "success": True
        }
        out_fname.write_text(json.dumps(result, indent=4))

        return {"success": True, "instance_id": instance_id, "eval_result": eval_result}

    except Exception as e:

        # Check if eval_result exists in local scope
        if 'eval_result' not in locals():
            eval_result = 'incomplete'
        else:
            eval_result = 'error'
        if 'model_patch' not in locals():
            model_patch = ''
        if 'proposed_model_patches' not in locals():
            proposed_model_patches = ''

        # Write result to file
        result = {
            "instance_id": instance_id,
            "model_name_or_path": model_name_or_path,
            "model_patch": model_patch,
            'proposed_model_patches': proposed_model_patches,
            "eval_result": eval_result,
            "success": False
        }
        out_fname.write_text(json.dumps(result, indent=4))

        print(f"Error processing entry {instance_id}: {str(e)}")
        return {"success": False, "instance_id": instance_id, "eval_result": eval_result}

    finally:
        # Clean up docker container
        try:
            cleanup_container(client, container, logger)
        except Exception as e:
            print(f"Error cleaning up Docker container for {instance_id}: {e}")

def harness(
        dataset_path=str(POLYGLOT_METADATA_PATH),
        test_task_list=None,
        num_samples=-1,
        max_workers=4,
        model_name_or_path=None,
        model_patch_paths=None,
        num_evals=1,
        num_evals_parallel=1,
        pred_dname='./outputs',
        output_dir='./outputs',
        root_dir=None,
    ):
    """
    _load_shared_env()
    Parallel processing harness using ThreadPoolExecutor.

    Args:
        test_task_list: List of task IDs to process (None for all)
        num_samples: Number of samples to process (-1 for all)
        max_workers: Maximum number of concurrent threads
        model_name_or_path: Model name or path
        model_patch_paths: Paths to the model patches for dgm
        num_evals: Repeated number of swe evaluations
    """
    if model_patch_paths:
        for model_patch_path in model_patch_paths:
            # Read and modify model patch
            with open(model_patch_path, 'r') as f:
                patch_content = f.read()
            patch_content = remove_patch_by_files(patch_content)
            # Placeholder for any patch modifications if needed
            with open(model_patch_path, 'w') as f:
                f.write(patch_content + '\n')

    if num_evals > 1:
        raise ValueError("Multiple evaluations (num_evals > 1) is not supported with polyglot")

    # Load dataset
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    with open(dataset_path) as f:
        dataset = json.load(f)
    dataset = _normalize_dataset_repo_paths(dataset)

    # Ensure that necessary directories exist
    if model_name_or_path is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        model_name_or_path = timestamp
    pred_dname = Path(pred_dname)
    pred_dname.mkdir(exist_ok=True)
    out_dnames = []

    # Prepare the dataset entries
    entries = list(dataset)
    # Capture the full benchmark cardinality BEFORE subset filtering so the
    # report can distinguish "instances attempted" vs "benchmark size".
    benchmark_total_instances = len(entries)
    if test_task_list:
        entries = [entry for entry in entries if entry['instance_id'] in test_task_list]
    if num_samples > 0:
        entries = entries[:num_samples]

    # Build the environment images
    client = docker.from_env()
    build_env_images(client, dataset=entries, max_workers=max_workers, force_rebuild=False)

    # Define a function to handle a single evaluation for all specified issues
    def process_evaluation(eval_idx):
        model_name_or_path_inst = f"{model_name_or_path}_{eval_idx}"
        out_dname = pred_dname / model_name_or_path_inst
        out_dname.mkdir(exist_ok=True)

        print(f"Starting evaluation {eval_idx} for model {model_name_or_path}")

        # Process entries in parallel
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_entry = {
                executor.submit(process_entry, entry, out_dname, model_name_or_path_inst, model_patch_paths, root_dir): entry
                for entry in entries
            }

            # Process completed tasks as they finish
            for future in as_completed(future_to_entry):
                result = future.result()
                results.append(future.result())
                if result["success"]:
                    print(f"Successfully processed entry {result['instance_id']} for eval {eval_idx}")
                else:
                    print(f"Failed to process entry {result['instance_id']} for eval {eval_idx}: {result.get('error', 'Unknown error')}")
        # Get final results from completed futures

        return out_dname, results

    out_dname, results = process_evaluation(0)
    print(f"All evaluations completed for model {model_name_or_path}")

    # Directly generate report
    # write report to file
    incomplete_ids = [result["instance_id"] for result in results if not result["success"]]
    completed_ids = [result["instance_id"] for result in results if result["success"]]
    # Get resolved/unresolved/error/empty patch IDs from results
    resolved_ids = []
    unresolved_ids = []
    error_ids = []
    empty_patch_ids = []
    unstopped_containers = []
    unremoved_images = []

    for result in results:
        if result["success"]:
            if result.get("eval_result") == "resolved":
                resolved_ids.append(result["instance_id"])
            elif result.get("eval_result") == "unresolved":
                unresolved_ids.append(result["instance_id"])
            elif result.get("eval_result") == "empty_patch":
                empty_patch_ids.append(result["instance_id"])
            else:
                error_ids.append(result["instance_id"])

    # NOTE: `total_instances` refers to the instances actually attempted in
    # this run (the filtered `entries` subset). `benchmark_total_instances`
    # preserves the full benchmark cardinality (size of the loaded dataset
    # prior to subset filtering) so downstream analyses that want the full
    # denominator can still compute it. This aligns with the ARC / SWE-bench
    # Pro report convention.
    report = {
        "total_instances": len(entries),
        "benchmark_total_instances": benchmark_total_instances,
        "submitted_instances": len(results),
        "completed_instances": len(completed_ids),
        "resolved_instances": len(resolved_ids),
        "unresolved_instances": len(unresolved_ids),
        "empty_patch_instances": len(empty_patch_ids),
        "error_instances": len(error_ids),
        "unstopped_instances": len(unstopped_containers),
        "completed_ids": list(sorted(completed_ids)),
        "incomplete_ids": list(sorted(incomplete_ids)),
        "empty_patch_ids": list(sorted(empty_patch_ids)),
        "submitted_ids": list(sorted(result["instance_id"] for result in results)),
        "resolved_ids": list(sorted(resolved_ids)),
        "unresolved_ids": list(sorted(unresolved_ids)),
        "error_ids": list(sorted(error_ids)),
        "unstopped_containers": list(sorted(unstopped_containers)),
        "unremoved_images": list(sorted(unremoved_images)),
        "schema_version": 2,
    }

    print(report)
    report_file = output_dir / Path(
        model_name_or_path.replace("/", "__") + f"_{0}"
        + f".{000}"
        + ".json"
    )
    with open(report_file, "w") as f:
        print(json.dumps(report, indent=4), file=f)
    print(f"Report written to {report_file}")

    return out_dnames

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=-1, help="Number of samples to process")
    parser.add_argument("--max_workers", type=int, default=5, help="Maximum number of concurrent threads")
    parser.add_argument("--model_name_or_path", type=str, default=None, help="Model name or path")
    parser.add_argument("--model_patch_paths", type=str, default=None, help="Paths to the model patches")
    parser.add_argument("--num_evals", type=int, default=1, help="Repeated number of swe evaluations")
    parser.add_argument("--num_evals_parallel", type=int, default=1, help="Number of parallel repeated evaluations")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--subset", type=str, default="small", help="Dataset subset")
    args = parser.parse_args()

    if args.subset == "small":
        benchmark_subset = POLYGLOT_TASK_MAP_DIR / "small.json"
        fallback_subset = "./domains/polyglot/subsets/small.json"
        task_list = load_json_file(str(benchmark_subset if benchmark_subset.exists() else fallback_subset))
    elif args.subset == "medium":
        benchmark_subset = POLYGLOT_TASK_MAP_DIR / "medium.json"
        fallback_subset = "./domains/polyglot/subsets/medium.json"
        task_list = load_json_file(str(benchmark_subset if benchmark_subset.exists() else fallback_subset))
    else:
        with open(POLYGLOT_METADATA_PATH) as f:
            metadata = json.loads(f.read())
            language_task_list = [entry["instance_id"] for entry in metadata if entry["instance_id"].startswith("python")]
            # Create a list of all tasks from metadata
            task_list = [entry["instance_id"] for entry in metadata]

    model_patch_paths = args.model_patch_paths.split(',') if args.model_patch_paths is not None else None

    # Run the parallel harness
    harness(
        dataset_path=str(POLYGLOT_METADATA_PATH),
        test_task_list=task_list,
        num_samples=args.num_samples,
        max_workers=args.max_workers,
        model_name_or_path=args.model_name_or_path,
        model_patch_paths=model_patch_paths,
        num_evals=args.num_evals,
        num_evals_parallel=args.num_evals_parallel,
        pred_dname=args.output_dir,
        output_dir=args.output_dir,
    )

if __name__ == "__main__":
    main()
