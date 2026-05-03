import argparse
import json
import os


def _aggregate_llm_usage(task_results):
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

    for result in task_results:
        chat_history_file = result.get("chat_history_file")
        if not chat_history_file or not os.path.exists(chat_history_file):
            continue
        with open(chat_history_file, "r", encoding="utf-8") as handle:
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
                cached_tokens = int(prompt_details.get("cached_tokens") or 0)
                reasoning_tokens = int(completion_details.get("reasoning_tokens") or 0)

                aggregate["calls"] += 1
                aggregate["prompt_tokens"] += prompt_tokens
                aggregate["cached_prompt_tokens"] += cached_tokens
                aggregate["uncached_prompt_tokens"] += max(prompt_tokens - cached_tokens, 0)
                aggregate["completion_tokens"] += completion_tokens
                aggregate["total_tokens"] += total_tokens
                aggregate["reasoning_tokens"] += reasoning_tokens

                cost = usage_record.get("cost_usd")
                if isinstance(cost, (int, float)):
                    aggregate["cost_usd"] += float(cost)
                    saw_cost = True

    if not saw_cost:
        aggregate["cost_usd"] = None
    return aggregate


def _resolve_run_dir(output_dir, run_keyword):
    direct_task_results_path = os.path.join(output_dir, "task_results.json")
    direct_results_path = os.path.join(output_dir, "official_eval", "results.json")
    if os.path.exists(direct_task_results_path) and os.path.exists(direct_results_path):
        return output_dir, output_dir
    run_dir = os.path.join(output_dir, f"{run_keyword}_0")
    return run_dir, output_dir


def report(output_dir, run_keyword="eval_run", expected_num_items=None):
    run_dir, report_dir = _resolve_run_dir(output_dir, run_keyword)
    task_results_path = os.path.join(run_dir, "task_results.json")
    official_results_path = os.path.join(run_dir, "official_eval", "results.json")

    with open(task_results_path, "r", encoding="utf-8") as handle:
        task_results = json.load(handle)
    with open(official_results_path, "r", encoding="utf-8") as handle:
        official_results = json.load(handle)

    timeout_ids = sorted(result["instance_id"] for result in task_results if result.get("agent_status") == "timeout")
    error_ids = sorted(result["instance_id"] for result in task_results if result.get("agent_status") == "error")
    missing_ids = sorted(result["instance_id"] for result in task_results if not result.get("prediction_file"))
    prediction_error_ids = sorted(result["instance_id"] for result in task_results if result.get("prediction_error"))
    completed_ids = sorted(result["instance_id"] for result in task_results if result.get("agent_status") == "completed")
    llm_usage = _aggregate_llm_usage(task_results)

    total_tasks = int(official_results.get("total_tasks", 0))
    score = float(official_results.get("score", 0.0))
    accuracy = score / total_tasks if total_tasks > 0 else 0.0
    scorer = official_results.get("scorer", "official_arc_agi_benchmarking")

    report_data = {
        "accuracy_score": accuracy,
        "score": score,
        "scorer": scorer,
        "official_scorer": scorer != "local_exact_match_fallback",
        "total_tasks": total_tasks,
        "total_pair_items": expected_num_items if expected_num_items is not None else len(task_results),
        "submitted_pair_items": len(task_results),
        "total_attempts": official_results.get("total_attempts", 0),
        "timeout_instances": len(timeout_ids),
        "error_instances": len(error_ids),
        "missing_prediction_instances": len(missing_ids),
        "prediction_error_instances": len(prediction_error_ids),
        "timeout_ids": timeout_ids,
        "error_ids": error_ids,
        "missing_prediction_ids": missing_ids,
        "prediction_error_ids": prediction_error_ids,
        "completed_ids": completed_ids,
        "llm_usage": llm_usage,
        "task_results_file": task_results_path,
        "official_results_file": official_results_path,
    }

    report_file = os.path.join(report_dir, "report.json")
    with open(report_file, "w", encoding="utf-8") as handle:
        json.dump(report_data, handle, indent=4)
    print(f"Report written to {report_file}")
    return report_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, default="eval_run")
    parser.add_argument("--expected_num_items", type=int, default=None)
    args = parser.parse_args()
    report(args.output_dir, args.model_name_or_path, expected_num_items=args.expected_num_items)
