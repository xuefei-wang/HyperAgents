import os
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
HYPERAGENTS_ROOT = Path(__file__).resolve().parents[2]

POLYGLOT_BENCHMARK_DIR = WORKSPACE_ROOT / "benchmarks/polyglot"
POLYGLOT_SOURCE_DIR = POLYGLOT_BENCHMARK_DIR / "source"
POLYGLOT_TASK_MAP_DIR = POLYGLOT_BENCHMARK_DIR / "task_maps"
POLYGLOT_METADATA_PATH = POLYGLOT_SOURCE_DIR / "polyglot_benchmark_metadata.json"

# Constants - Evaluation Log Directories
BASE_IMAGE_BUILD_DIR = HYPERAGENTS_ROOT / "domains/polyglot/logs/build_images/base"
ENV_IMAGE_BUILD_DIR = HYPERAGENTS_ROOT / "domains/polyglot/logs/build_images/env"
INSTANCE_IMAGE_BUILD_DIR = HYPERAGENTS_ROOT / "domains/polyglot/logs/build_images/instances"
RUN_EVALUATION_LOG_DIR = HYPERAGENTS_ROOT / "domains/polyglot/logs/run_evaluation"

NPM_TEST_SH = """#!/bin/bash

# exit when any command fails
set -e

# Create symlinks if they don't exist
[ ! -e node_modules ] && ln -s /npm-install/node_modules .
[ ! -e package-lock.json ] && ln -s /npm-install/package-lock.json .


sed -i 's/\bxtest(/test(/g' *.spec.js
npm run test
"""

CPP_TEST_SH = """#!/bin/bash

# exit when any command fails
set -e

[ ! -d "build" ] && mkdir build
cd build
cmake -DEXERCISM_RUN_ALL_TESTS=1 -G "Unix Makefiles" ..
make
"""

NPM_TEST_COMMANDS = [
    "set -e",
    "[ ! -e node_modules ] && ln -s /npm-install/node_modules .",
    "[ ! -e package-lock.json ] && ln -s /npm-install/package-lock.json .",
    "sed -i 's/\\bxtest(/test(/g' *.spec.js",
    "npm run test"
]

CPP_TEST_COMMANDS = [
    "set -e",
    "[ ! -d \"build\" ] && mkdir build",
    "cd build",
    "cmake -DEXERCISM_RUN_ALL_TESTS=1 -G \"Unix Makefiles\" ..",
    "make",
    "cd ../"
]


TEST_COMMANDS = {
    "python": ["pytest -rA --tb=long"],
    "rust": ["cargo test -- --include-ignored"],
    "go": ["go test ./..."],
    "javascript": NPM_TEST_COMMANDS,
    "cpp": CPP_TEST_COMMANDS,
    "java": ["./gradlew test"],
}

PYTHON_SPECS = {
    "python": "3.11",
    "pip_packages": [
        "flake8~=5.0.4",
        "pylint~=2.17.1",
        "black==22.3.0",
        "yapf~=0.32.0",
        "tomli>=1.1.0",
        "pytest",
        "pytest-env",
        "pytest-asyncio",
        "async_timeout",
        "trio",
        "testpath"
    ],
    "test_cmd": " ".join(TEST_COMMANDS["python"]),
}

GENERIC_SPECS = {
    "rust": {
        "python": "3.11",
        "test_cmd": " ".join(TEST_COMMANDS["rust"])
    },
    "go": {
        "python": "3.11",
        "test_cmd": " ".join(TEST_COMMANDS["go"])
    },
    "javascript": {
        "python": "3.11",
        "test_cmd": "\n".join(TEST_COMMANDS["javascript"])
    },
    "cpp": {
        "python": "3.11",
        "test_cmd": "\n".join(TEST_COMMANDS["cpp"])
    },
    "java": {
        "python": "3.11",
        "test_cmd": " ".join(TEST_COMMANDS["java"])
    }
}

MAP_REPO_VERSION_TO_SPECS = {
    "python": PYTHON_SPECS,
    "rust": GENERIC_SPECS["rust"],
    "go": GENERIC_SPECS["go"],
    "javascript": GENERIC_SPECS["javascript"],
    "cpp": GENERIC_SPECS["cpp"],
    "java": GENERIC_SPECS["java"]
}

MAP_REPO_TO_INSTALL = {}

USE_X86 = {}


def _resolve_polyglot_timeout(default: int = 600) -> int:
    """Read CROSS_RUNNER_AGENT_TIMEOUT_SEC first (cross-runner sweep override),
    then POLYGLOT_AGENT_TIMEOUT_SECONDS_ENV (HA-specific override), falling back
    to ``default`` (the legacy 600s / 10-min budget). Returns a positive int
    seconds value. Mirrors domains.swebench_pro.constants._resolve_swebench_pro_timeout
    so the polyglot harness honors the same sweep-level timeout (kcsi #1125)."""
    for var in ("CROSS_RUNNER_AGENT_TIMEOUT_SEC", "POLYGLOT_AGENT_TIMEOUT_SECONDS_ENV"):
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


POLYGLOT_AGENT_TIMEOUT_SECONDS = _resolve_polyglot_timeout()
