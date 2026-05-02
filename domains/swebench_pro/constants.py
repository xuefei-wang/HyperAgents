import os
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
HYPERAGENTS_ROOT = Path(__file__).resolve().parents[2]

SWEBENCH_PRO_BENCHMARK_DIR = WORKSPACE_ROOT / "benchmarks" / "swebench_pro"
SWEBENCH_PRO_SOURCE_DIR = SWEBENCH_PRO_BENCHMARK_DIR / "source"
SWEBENCH_PRO_TASK_MAP_DIR = SWEBENCH_PRO_BENCHMARK_DIR / "task_maps"
SWEBENCH_PRO_REPO_CACHE_DIR = SWEBENCH_PRO_BENCHMARK_DIR / "repo_cache"
SWEBENCH_PRO_DATASET_PATH = SWEBENCH_PRO_BENCHMARK_DIR / "dataset" / "test.jsonl"
SWEBENCH_PRO_DEFAULT_TASK_MAP = SWEBENCH_PRO_TASK_MAP_DIR / "swebench_pro_test_50_seed0_v1.json"


def _resolve_swebench_pro_timeout(default: int = 600) -> int:
    """Read CROSS_RUNNER_AGENT_TIMEOUT_SEC first (cross-runner sweep override),
    then SWEBENCH_PRO_AGENT_TIMEOUT_SECONDS_ENV (HA-specific override),
    falling back to ``default``. Returns a positive int seconds value."""
    for var in ("CROSS_RUNNER_AGENT_TIMEOUT_SEC", "SWEBENCH_PRO_AGENT_TIMEOUT_SECONDS_ENV"):
        raw = os.environ.get(var, "").strip()
        if not raw:
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        if value > 0:
            return value
    return default


SWEBENCH_PRO_AGENT_TIMEOUT_SECONDS = _resolve_swebench_pro_timeout()
SWEBENCH_PRO_EVAL_DOCKERHUB_USERNAME = "jefzda"
