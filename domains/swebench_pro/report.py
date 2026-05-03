import argparse
import glob
import json
import os


def _aggregate_llm_usage(run_dir):
    """Aggregate LLM_USAGE records from per-instance markdown files under run_dir.

    Unlike the ARC variant (which walks chat_history_file paths from task_results),
    SWE-bench Pro writes chat histories alongside the task_results.json in run_dir,
    so we glob *.md there directly — this avoids depending on the relative-path
    quirks of chat_history_file, which are written relative to the hyperagents CWD.
    """
    aggregate = {
        "calls": 0,
        "malformed_records": 0,
        "total_tokens": 0,
        "prompt_tokens": 0,
        "cached_prompt_tokens": 0,
        "uncached_prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "cost_usd": 0.0,
    }
    saw_cost = False

    md_files = sorted(glob.glob(os.path.join(run_dir, "*.md")))
    for md_path in md_files:
        try:
            with open(md_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    if "LLM_USAGE:" not in line:
                        continue
                    try:
                        usage_record = json.loads(line.split("LLM_USAGE:", 1)[1].strip())
                    except json.JSONDecodeError:
                        aggregate["malformed_records"] += 1
                        continue

                    usage = usage_record.get("usage") or {}
                    prompt_tokens = int(usage.get("prompt_tokens") or 0)
                    completion_tokens = int(usage.get("completion_tokens") or 0)
                    total_tokens = int(usage.get("total_tokens") or 0)
                    prompt_details = usage.get("prompt_tokens_details") or {}
                    completion_details = usage.get("completion_tokens_details") or {}
                    output_details = usage.get("output_tokens_details") or {}
                    cached_tokens = int(prompt_details.get("cached_tokens") or 0)
                    reasoning_tokens = int(
                        completion_details.get("reasoning_tokens")
                        or output_details.get("reasoning_tokens")
                        or 0
                    )

                    aggregate["calls"] += 1
                    aggregate["prompt_tokens"] += prompt_tokens
                    aggregate["cached_prompt_tokens"] += cached_tokens
                    aggregate["uncached_prompt_tokens"] += max(prompt_tokens - cached_tokens, 0)
                    aggregate["completion_tokens"] += completion_tokens
                    aggregate["total_tokens"] += total_tokens or prompt_tokens + completion_tokens
                    aggregate["reasoning_tokens"] += reasoning_tokens

                    cost = usage_record.get("cost_usd")
                    if isinstance(cost, (int, float)):
                        aggregate["cost_usd"] += float(cost)
                        saw_cost = True
        except OSError:
            continue

    if not saw_cost:
        aggregate["cost_usd"] = None
    return aggregate


def _resolve_run_dir(output_dir, run_keyword):
    direct_task_results_path = os.path.join(output_dir, "task_results.json")
    direct_eval_results_path = os.path.join(output_dir, "official_eval", "eval_results.json")
    if os.path.exists(direct_task_results_path) and os.path.exists(direct_eval_results_path):
        return output_dir, output_dir

    run_dir = os.path.join(output_dir, f"{run_keyword}_0")
    return run_dir, output_dir


def report(output_dir, run_keyword, expected_num_tasks=None):
    run_dir, report_dir = _resolve_run_dir(output_dir, run_keyword)
    task_results_path = os.path.join(run_dir, "task_results.json")
    eval_results_path = os.path.join(run_dir, "official_eval", "eval_results.json")

    with open(task_results_path, "r", encoding="utf-8") as handle:
        task_results = json.load(handle)
    with open(eval_results_path, "r", encoding="utf-8") as handle:
        eval_results = json.load(handle)

    total_instances = expected_num_tasks if expected_num_tasks is not None else len(task_results)
    resolved_ids = sorted(instance_id for instance_id, resolved in eval_results.items() if resolved)
    unresolved_ids = sorted(instance_id for instance_id, resolved in eval_results.items() if not resolved)
    timeout_ids = sorted(result["instance_id"] for result in task_results if result.get("agent_status") == "timeout")
    error_ids = sorted(result["instance_id"] for result in task_results if result.get("agent_status") == "error")
    completed_ids = sorted(result["instance_id"] for result in task_results if result.get("agent_status") == "completed")
    empty_patch_ids = sorted(result["instance_id"] for result in task_results if result.get("patch_bytes", 0) == 0)
    llm_usage = _aggregate_llm_usage(run_dir)

    report_data = {
        "accuracy_score": len(resolved_ids) / total_instances if total_instances > 0 else 0.0,
        "total_instances": total_instances,
        "submitted_instances": len(task_results),
        "resolved_instances": len(resolved_ids),
        "unresolved_instances": len(unresolved_ids),
        "empty_patch_instances": len(empty_patch_ids),
        "error_instances": len(error_ids),
        "timeout_instances": len(timeout_ids),
        "resolved_ids": resolved_ids,
        "unresolved_ids": unresolved_ids,
        "empty_patch_ids": empty_patch_ids,
        "error_ids": error_ids,
        "timeout_ids": timeout_ids,
        "completed_ids": completed_ids,
        "llm_usage": llm_usage,
        "task_results_file": task_results_path,
        "official_eval_results_file": eval_results_path,
    }

    report_file = os.path.join(report_dir, "report.json")
    with open(report_file, "w", encoding="utf-8") as handle:
        json.dump(report_data, handle, indent=4)
    print(f"Report written to {report_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, default="eval_run")
    parser.add_argument("--expected_num_tasks", type=int, default=None)
    args = parser.parse_args()
    report(args.output_dir, args.model_name_or_path, expected_num_tasks=args.expected_num_tasks)
