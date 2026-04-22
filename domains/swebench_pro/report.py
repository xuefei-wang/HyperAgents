import argparse
import json
import os


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
