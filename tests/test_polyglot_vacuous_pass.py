"""Unit tests for the polyglot vacuous-pass guard (kcsi #1137 parity).

`is_vacuous_pass` only ever turns an exit-0 eval that ran ZERO tests into a
fail; it must never flag a genuine passing run and must never fire for the
languages it does not guard (cpp/js/java). These cases pin the go/rust/python
markers so a future edit that breaks the regex is caught here rather than in a
live campaign.
"""

import pytest

from domains.polyglot.harness import is_vacuous_pass


# --- vacuous (zero tests ran on an exit-0 run) -> True ---------------------
@pytest.mark.parametrize(
    "language, output",
    [
        # go: "[no test files]" and no "ok\t<pkg>" line means nothing ran.
        ("go", "?   \texample/foo\t[no test files]\n"),
        # rust: a genuine pass reports "ok. <n>=1.. passed"; 0 passed == vacuous.
        ("rust", "test result: ok. 0 passed; 0 failed; 0 ignored\n"),
        ("rust", ""),
        # python: pytest forced to exit 0 while collecting nothing.
        ("python", "collected 0 items\n\nno tests ran in 0.01s\n"),
        ("python", "===== no tests ran in 0.00s =====\n"),
    ],
)
def test_vacuous_outputs_flagged(language, output):
    assert is_vacuous_pass(language, output) is True


# --- genuine passing runs -> False (never flag a real pass) ----------------
@pytest.mark.parametrize(
    "language, output",
    [
        # go: at least one "ok\t<pkg>" line means a package ran a test, even if
        # another package reports "[no test files]".
        ("go", "ok  \texample/foo\t0.02s\n?   \texample/bar\t[no test files]\n"),
        ("go", "ok  \texample/foo\t0.02s\n"),
        # rust: >=1 passed.
        ("rust", "test result: ok. 3 passed; 0 failed; 0 ignored\n"),
        ("rust", "running 1 test\ntest tests::it_works ... ok\ntest result: ok. 1 passed; 0 failed\n"),
        # python: real collection.
        ("python", "collected 5 items\n\ntest_foo.py .....  [100%]\n5 passed in 0.10s\n"),
    ],
)
def test_genuine_pass_not_flagged(language, output):
    assert is_vacuous_pass(language, output) is False


# --- unguarded languages -> always False -----------------------------------
@pytest.mark.parametrize("language", ["cpp", "javascript", "java", "unknown"])
@pytest.mark.parametrize("output", ["", "anything at all", "collected 0 items"])
def test_unguarded_languages_never_flagged(language, output):
    assert is_vacuous_pass(language, output) is False


# --- None / empty eval_output is handled without raising -------------------
@pytest.mark.parametrize(
    "language, expected",
    [
        # rust with no output at all == no "N passed" line == vacuous.
        ("rust", True),
        # go/python require a positive marker that an empty string lacks.
        ("go", False),
        ("python", False),
        ("cpp", False),
    ],
)
def test_none_output_does_not_raise(language, expected):
    assert is_vacuous_pass(language, None) is expected
