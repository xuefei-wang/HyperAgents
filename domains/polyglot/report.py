import argparse
import glob
import os
import json

from utils.common import load_json_file


def _aggregate_llm_usage(results_dir):
    """Aggregate LLM_USAGE records from per-task markdown chat histories.

    Polyglot writes per-task chat histories under ``eval_run_*/``; each ``.md``
    file is the chat_history for one Polyglot instance. This mirrors the
    aggregator in ``domains/swebench_pro/report.py`` but walks the
    ``eval_run_*`` subdirectories so runs with multiple eval passes still
    aggregate cleanly.
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

    md_files = sorted(glob.glob(os.path.join(results_dir, "eval_run_*", "*.md")))
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


def get_all_performance(run_keyword, results_dir='./outputs', expected_num_tasks=None):
    """
    Retrieve performance results for all runs based on the provided keyword.

    Args:
        run_keyword (str): A keyword used to identify the target runs' evaluation results.

    Returns:
        list: A list of dictionaries, each containing performance results for a matching run.
    """
    # Find all JSON files in eval_results_dir matching the keyword
    matching_files = [
        f for f in os.listdir(results_dir)
        if f.endswith('.json') and run_keyword in f
    ]

    # Return an empty list if no matches are found
    if not matching_files:
        print(f"No evaluation files found matching the keyword '{run_keyword}'.")
        return None, None

    # Process each matching file
    performance_results = []
    total_resolved_instances = 0
    total_submitted_instances = 0
    total_unresolved_ids = []
    total_resolved_ids = []
    total_emptypatch_ids = []
    total_reported_instances = 0
    total_benchmark_instances = 0
    for file_name in matching_files:
        eval_agent_path = os.path.join(results_dir, file_name)
        eval_results = load_json_file(eval_agent_path)
        resolved_instances = eval_results.get('resolved_instances', 0)
        submitted_instances = eval_results.get('submitted_instances', 0)
        # `total_instances` is subset-aware after the fix (counts attempted
        # instances for this run). `benchmark_total_instances` is the new
        # field that preserves the full benchmark cardinality. For legacy
        # artifacts written before the fix, `benchmark_total_instances` is
        # absent, so we fall back to the old `total_instances` value so
        # historical reports still aggregate cleanly.
        reported_instances = eval_results.get('total_instances', 0)
        benchmark_instances = eval_results.get(
            'benchmark_total_instances',
            reported_instances if isinstance(reported_instances, int) else 0,
        )
        total_resolved_instances += resolved_instances
        total_submitted_instances += submitted_instances
        total_reported_instances += reported_instances if isinstance(reported_instances, int) else 0
        total_benchmark_instances += benchmark_instances if isinstance(benchmark_instances, int) else 0
        accuracy_score = resolved_instances / submitted_instances if submitted_instances > 0 else 0
        performance_results.append({'file': file_name, 'accuracy_score': accuracy_score, **eval_results})
        total_unresolved_ids.extend(eval_results.get('unresolved_ids', []))
        total_emptypatch_ids.extend(eval_results.get('empty_patch_ids', []))
        total_resolved_ids.extend(eval_results.get('resolved_ids', []))

    # Calculate the overall accuracy score.
    #
    # `total_instances` is subset-aware (counts instances actually attempted
    # in this run). `benchmark_total_instances` carries the full benchmark
    # cardinality for downstream analyses that need the original denominator.
    #
    # When `expected_num_tasks` is supplied, cap `total_instances` at the
    # smaller of (expected, total_reported) so a caller over-stating the
    # expected count can't inflate the denominator past what was actually
    # attempted. With the fix, per-file `total_instances` is the attempted
    # count, so this `min()` resolves to the true attempted count.
    overall_performance = {}
    if expected_num_tasks is not None:
        total_instances = expected_num_tasks
        if total_reported_instances > 0:
            total_instances = min(total_instances, total_reported_instances)
    else:
        total_instances = total_reported_instances or total_submitted_instances
    overall_performance['accuracy_score'] = total_resolved_instances / total_instances if total_instances > 0 else 0
    overall_performance['total_resolved_instances'] = total_resolved_instances
    overall_performance['total_submitted_instances'] = total_submitted_instances
    overall_performance['total_instances'] = total_instances
    overall_performance['benchmark_total_instances'] = total_benchmark_instances
    overall_performance['files'] = matching_files
    overall_performance['total_unresolved_ids'] = total_unresolved_ids
    overall_performance['total_emptypatch_ids'] = total_emptypatch_ids
    overall_performance['total_resolved_ids'] = total_resolved_ids
    overall_performance['llm_usage'] = _aggregate_llm_usage(results_dir)

    return performance_results, overall_performance

def report(output_dir, run_keyword, expected_num_tasks=None):
    # Get performance results
    performances, overall_performance = get_all_performance(run_keyword, results_dir=output_dir, expected_num_tasks=expected_num_tasks)

    # Write report
    report_file = os.path.join(output_dir, "report.json")
    with open(report_file, "w") as f:
        print(json.dumps(overall_performance, indent=4), file=f)
    print(f"Report written to {report_file}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./outputs/initial_polyglot_0", help="Output directory")
    parser.add_argument("--model_name_or_path", type=str, default="eval_run", help="Dataset subset")
    args = parser.parse_args()

    report(args.output_dir, args.model_name_or_path)
