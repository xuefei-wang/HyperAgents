"""Docker-free unit tests for the polyglot /testbed history scrub (LEAK 2).

Reproduces the real prepare -> clone -> reset sequence in a tmp git repo and runs
the IDENTICAL git command sequences the harness runs in-container (via
domains.polyglot.history_scrub), asserting the hidden test file + .meta reference
solution are unrecoverable after the scrub, base_commit is intact, and grading
can still obtain the tests via the re-injection bundle.
"""

import subprocess

import pytest

from domains.polyglot import history_scrub

# Distinctive markers we assert never leak (hidden test) / can be restored.
_HIDDEN_TEST_MARKER = "SECRET_ASSERT_EXPECTED_42"
_REFERENCE_SOLUTION_MARKER = "REFERENCE_SOLUTION_IS_42"


def _bash(cmd, check=True):
    # errors="replace": some probes (git cat-file --batch) emit raw object bytes.
    return subprocess.run(
        ["bash", "-lc", cmd], capture_output=True, text=True, errors="replace", check=check
    )


def _git(cwd, *args, check=True):
    return subprocess.run(
        ["git", "-C", str(cwd), *args], capture_output=True, text=True, check=check
    )


def _prepare_source(src):
    """Mirror prepare_polyglot_dataset.register_git: solution-only base_commit,
    then `git add . && git commit --amend` -> test_commit (all files)."""
    src.mkdir(parents=True)
    _git(src, "init", "-q")
    _git(src, "config", "user.email", "t@example.com")
    _git(src, "config", "user.name", "prep")
    (src / "solution.py").write_text("def solve():\n    pass\n")
    (src / ".docs").mkdir()
    (src / ".docs" / "instructions.md").write_text("Implement solve().\n")
    _git(src, "add", "solution.py", ".docs")
    _git(src, "commit", "-qm", "Initial commit")
    base_commit = _git(src, "rev-parse", "HEAD").stdout.strip()

    # Everything else (the amend): hidden tests + .meta reference solution.
    (src / "solution_test.py").write_text(f"assert solve() == '{_HIDDEN_TEST_MARKER}'\n")
    (src / ".meta").mkdir()
    (src / ".meta" / "example.py").write_text(f"def solve():\n    return '{_REFERENCE_SOLUTION_MARKER}'\n")
    _git(src, "add", ".")
    _git(src, "commit", "--amend", "-qm", "all files")
    test_commit = _git(src, "rev-parse", "HEAD").stdout.strip()
    return base_commit, test_commit


def _clone_testbed(src, testbed, base_commit):
    """Mirror make_repo_script_list: local clone then reset --hard base_commit."""
    _bash(f"git clone -q {src} {testbed}")
    _git(testbed, "config", "user.email", "t@example.com")
    _git(testbed, "config", "user.name", "agent")
    _git(testbed, "reset", "--hard", base_commit)


def _run_probes(testbed, probes):
    return {cmd: (_bash(cmd, check=False).stdout + _bash(cmd, check=False).stderr) for cmd in probes}


def _content_probes(testbed, test_commit):
    """Recovery vectors keyed on the hidden-test/reference-solution CONTENT."""
    return _run_probes(
        testbed,
        [
            f"git -C {testbed} show origin/HEAD:solution_test.py",
            f"git -C {testbed} show origin/master:solution_test.py",
            f"git -C {testbed} cat-file -p {test_commit}:solution_test.py",
            f"git -C {testbed} cat-file -p {test_commit}:.meta/example.py",
            f"git -C {testbed} log --all --source -p",
            f"git -C {testbed} cat-file --batch-all-objects --batch",
        ],
    )


def _enumeration_probes(testbed):
    """Object/ref enumerations that must not even NAME test_commit (no sha in cmd)."""
    return _run_probes(
        testbed,
        [
            f"git -C {testbed} log --all --pretty=%H",
            f"git -C {testbed} reflog --all",
            f"git -C {testbed} rev-list --all --objects",
            f"git -C {testbed} fsck --unreachable --no-reflogs",
            f"git -C {testbed} for-each-ref",
        ],
    )


def test_scrub_makes_hidden_tests_unrecoverable_then_reinject_restores(tmp_path):
    src = tmp_path / "repo_source"
    base_commit, test_commit = _prepare_source(src)
    testbed = tmp_path / "testbed"
    _clone_testbed(src, testbed, base_commit)
    host_bundle = tmp_path / "host" / "tests.bundle"
    host_bundle.parent.mkdir()

    # Pre-condition: the leak is real before scrub.
    pre = _bash(f"git -C {testbed} cat-file -p {test_commit}:solution_test.py")
    assert _HIDDEN_TEST_MARKER in pre.stdout

    # 1) Bundle test_commit out to a host path (outside /testbed).
    for cmd in history_scrub.bundle_create_commands(
        test_commit, testbed=str(testbed), bundle=str(host_bundle)
    ):
        _bash(cmd)
    assert host_bundle.exists()

    # 2) Scrub /testbed down to base_commit and delete /repo_source.
    for cmd in history_scrub.scrub_commands(
        base_commit, testbed=str(testbed), repo_source=str(src)
    ):
        _bash(cmd)

    # 3) Fail-closed verification must PASS (exit 0).
    verify = _bash(
        history_scrub.verify_scrub_command(
            test_commit, base_commit, testbed=str(testbed), repo_source=str(src)
        ),
        check=False,
    )
    assert verify.returncode == 0, f"verify failed: {verify.stdout}{verify.stderr}"

    # No content-bearing vector recovers the hidden test or reference solution.
    for cmd, out in _content_probes(testbed, test_commit).items():
        assert _HIDDEN_TEST_MARKER not in out, f"hidden test leaked via: {cmd}"
        assert _REFERENCE_SOLUTION_MARKER not in out, f"reference solution leaked via: {cmd}"
    # No enumeration even names test_commit (these do not embed the sha).
    for cmd, out in _enumeration_probes(testbed).items():
        assert test_commit not in out, f"test_commit reachable via: {cmd}"

    # /repo_source is gone; the hidden test is not on disk during the agent run.
    assert not src.exists()
    assert not (testbed / "solution_test.py").exists()
    assert not (testbed / ".meta").exists()

    # base_commit content is intact (working tree + git object).
    assert (testbed / "solution.py").exists()
    assert "def solve" in _bash(f"git -C {testbed} cat-file -p {base_commit}:solution.py").stdout
    assert _bash(f"git -C {testbed} rev-parse HEAD").stdout.strip() == base_commit

    # 4) Grade-time re-injection restores test_commit, so `git reset --hard
    #    <test_commit>` (the harness grade step) obtains the hidden tests again.
    for cmd in history_scrub.reinject_commands(
        test_commit, testbed=str(testbed), bundle=str(host_bundle)
    ):
        _bash(cmd)
    _git(testbed, "reset", "--hard", test_commit)
    assert (testbed / "solution_test.py").exists()
    assert _HIDDEN_TEST_MARKER in (testbed / "solution_test.py").read_text()
    assert _REFERENCE_SOLUTION_MARKER in (testbed / ".meta" / "example.py").read_text()


def test_verify_fails_closed_when_scrub_not_applied(tmp_path):
    """The leak guard must reject an un-scrubbed clone (test_commit recoverable)."""
    src = tmp_path / "repo_source"
    base_commit, test_commit = _prepare_source(src)
    testbed = tmp_path / "testbed"
    _clone_testbed(src, testbed, base_commit)

    verify = _bash(
        history_scrub.verify_scrub_command(
            test_commit, base_commit, testbed=str(testbed), repo_source=str(src)
        ),
        check=False,
    )
    assert verify.returncode != 0
    assert "LEAK-GUARD" in (verify.stdout + verify.stderr)
