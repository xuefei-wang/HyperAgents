import json
import os
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
HYPERAGENTS_ROOT = Path(__file__).resolve().parents[2]

ARC_BENCHMARK_DIR = WORKSPACE_ROOT / "benchmarks" / "arc"
ARC_BENCHMARKING_SRC = ARC_BENCHMARK_DIR / "benchmarking" / "src"
ARC_WORKSPACE_UI_DIR = ARC_BENCHMARK_DIR / "workspace_ui"
ARC_WORKSPACE_PAYLOADS_DIR = ARC_BENCHMARK_DIR / "workspace_payloads"

ARC_DEFAULTS_CONFIG = Path(
    os.environ.get(
        "SWARMS_ARC_DEFAULTS_CONFIG",
        WORKSPACE_ROOT / "configs" / "benchmarks" / "arc_defaults.json",
    )
)


def _load_arc_defaults() -> dict:
    try:
        return json.loads(ARC_DEFAULTS_CONFIG.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _default_path(benchmark: str, field: str, fallback: Path) -> Path:
    value = _ARC_DEFAULTS.get(benchmark, {}).get(field)
    if not value:
        return fallback
    path = Path(value)
    return path if path.is_absolute() else WORKSPACE_ROOT / path


_ARC_DEFAULTS = _load_arc_defaults()

ARC1_TASK_DIR = _default_path(
    "arc1",
    "task_dir",
    WORKSPACE_ROOT / "benchmarks" / "arc1" / "source" / "data" / "training",
)
ARC2_TASK_DIR = _default_path(
    "arc2",
    "task_dir",
    WORKSPACE_ROOT / "benchmarks" / "arc2" / "source" / "data" / "training",
)

ARC_DEFAULT_MANIFESTS = {
    "arc1": Path(
        os.environ.get(
            "HYPERAGENTS_ARC1_MANIFEST",
            _default_path(
                "arc1",
                "payload_manifest",
                ARC_WORKSPACE_PAYLOADS_DIR / "arc1" / "arc1_train_50_seed0" / "manifest.json",
            ),
        )
    ),
    "arc2": Path(
        os.environ.get(
            "HYPERAGENTS_ARC2_MANIFEST",
            _default_path(
                "arc2",
                "payload_manifest",
                ARC_WORKSPACE_PAYLOADS_DIR / "arc2" / "arc2_train_50_seed0" / "manifest.json",
            ),
        )
    ),
}

ARC_TASK_DIRS = {
    "arc1": ARC1_TASK_DIR,
    "arc2": ARC2_TASK_DIR,
}

ARC_AGENT_TIMEOUT_SECONDS = 600
