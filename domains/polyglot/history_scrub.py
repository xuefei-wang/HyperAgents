# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Close the polyglot /testbed + /repo_source hidden-test leak (matched-info regime).

The instance image ships ``/testbed`` as a ``git clone`` of ``/repo_source`` and
then ``git reset --hard``'s it to ``base_commit`` (domains/polyglot/test_spec.py:
``make_repo_script_list``). ``/repo_source``'s HEAD is ``test_commit`` -- the
``git commit --amend`` "all files" commit created in
``prepare_polyglot_dataset.py`` that contains the HIDDEN TESTS and the Exercism
``.meta`` REFERENCE-SOLUTION/example files. Two problems follow:

  * A LOCAL ``git clone`` hardlinks the whole object store, so ``/testbed``
    keeps the ``test_commit`` object AND ``origin`` refs pointing at it even
    after the reset to ``base_commit``.
  * ``/repo_source`` itself stays in the container with every file on disk.

The task agent runs as root with an unrestricted bash tool and
``--git_dir /testbed`` (harness.py), so it can recover the graded answer
offline, defeating the matched-information regime::

    git -C /testbed show origin/HEAD:<testfile>
    git -C /testbed cat-file -p <test_commit>:<testfile>
    git -C /testbed log --all           # / reflog / rev-list --all / fsck
    cat /repo_source/<testfile>          # or /repo_source/.meta/<solution>

Grading still needs ``test_commit`` -- harness.py restores the official test
files with ``git -C /testbed reset --hard <test_commit>`` (wrapped by a stash
push/pop of the agent's solution). So we cannot simply delete it. Instead:

  * BEFORE the agent runs: bundle ``test_commit``, pull the bundle OUT to the
    host (the running agent container never holds it), then scrub ``/testbed``
    down to ``base_commit`` only and delete ``/repo_source``. A fail-closed
    verification asserts ``test_commit`` is unrecoverable and aborts the task
    otherwise (scored as an error -- never a leak).
  * AT GRADE TIME: copy the bundle back in and fetch it, restoring
    ``test_commit`` so the existing reset/stash flow is byte-for-byte unchanged.

Mirrors ``runtime_runner/src/workspace.ts:sanitizeRepoHistory`` (kcsi issue
#924): drop remotes, delete refs, expire reflogs, absorb alternates, gc-prune,
verify. The command sequences here are pure data (lists of shell strings) so the
container harness and the Docker-free unit test exercise the IDENTICAL git ops.
"""

import shlex

# In-container paths. ``/repo_source`` is the clone SOURCE baked into the image
# (dockerfiles.py: ``COPY ./repo_source /repo_source``); it is unused after the
# clone, so we delete it. The bundle lives under ``/tmp`` only transiently while
# it is pulled to the host.
TESTBED = "/testbed"
REPO_SOURCE = "/repo_source"
CONTAINER_BUNDLE = "/tmp/eval-tests.bundle"
# A private ref namespace the agent's own commits never touch; the bundle
# carries exactly this ref so grade-time re-injection is deterministic.
BUNDLE_REF = "refs/eval/testcommit"


def bundle_create_commands(test_commit, testbed=TESTBED, bundle=CONTAINER_BUNDLE, ref=BUNDLE_REF):
    """Commands that bundle ``test_commit`` (still present in the fresh clone)."""
    testbed = shlex.quote(testbed)
    bundle = shlex.quote(bundle)
    ref = shlex.quote(ref)
    test_commit = shlex.quote(test_commit)
    return [
        f"git -C {testbed} update-ref {ref} {test_commit}",
        f"git -C {testbed} bundle create {bundle} {ref}",
        f"git -C {testbed} update-ref -d {ref}",
    ]


def scrub_commands(base_commit, testbed=TESTBED, repo_source=REPO_SOURCE):
    """Commands that reduce ``/testbed`` to ``base_commit`` only and drop ``/repo_source``."""
    q_testbed = shlex.quote(testbed)
    q_repo_source = shlex.quote(repo_source)
    q_base = shlex.quote(base_commit)
    alternates = shlex.quote(f"{testbed}/.git/objects/info/alternates")
    return [
        # Detach at base_commit so deleting branch refs cannot orphan it; the
        # working tree is left exactly at base_commit for the agent.
        f"git -C {q_testbed} checkout -q --detach {q_base}",
        # Drop all remotes (also kills ``git fetch origin <future-sha>`` and the
        # refs/remotes/origin/* tracking refs that reach test_commit).
        f'for r in $(git -C {q_testbed} remote); do git -C {q_testbed} remote remove "$r"; done',
        # Delete every remaining ref; the detached HEAD keeps base_commit reachable.
        f"git -C {q_testbed} for-each-ref --format='%(refname)' | "
        f'while read ref; do git -C {q_testbed} update-ref -d "$ref"; done',
        # Expire reflogs so they cannot keep test_commit reachable.
        f"git -C {q_testbed} reflog expire --expire=now --all",
        # A ``--shared`` clone would keep future objects in an alternate that gc
        # cannot prune; absorb them locally then drop the link (defensive -- a
        # plain local clone hardlinks instead, but this costs nothing).
        f"if [ -f {alternates} ]; then git -C {q_testbed} repack -a -d && rm -f {alternates}; fi",
        # Consolidate packs (drops hardlinked packs that still carry test_commit)
        # and physically prune the now-unreachable objects.
        f"git -C {q_testbed} repack -a -d -q",
        f"git -C {q_testbed} gc --prune=now --quiet",
        # Remove the on-disk clone source (holds the tests, .meta, and full git).
        f"rm -rf {q_repo_source}",
    ]


def verify_scrub_command(test_commit, base_commit, testbed=TESTBED, repo_source=REPO_SOURCE):
    """A single shell command that exits non-zero if any leak surface remains.

    Authoritative fail-closed gate: a security control must not silently no-op,
    so this is checked with ``raise_error=True`` after the scrub.
    """
    q_testbed = shlex.quote(testbed)
    q_repo_source = shlex.quote(repo_source)
    q_base = shlex.quote(base_commit)
    q_test = shlex.quote(test_commit)
    return "; ".join(
        [
            # base_commit must survive intact.
            f"if ! git -C {q_testbed} cat-file -e {q_base}^{{commit}} 2>/dev/null; then "
            'echo "LEAK-GUARD: base_commit missing after scrub" >&2; exit 1; fi',
            # test_commit object must be gone.
            f"if git -C {q_testbed} cat-file -e {q_test}^{{commit}} 2>/dev/null; then "
            'echo "LEAK-GUARD: test_commit object still present" >&2; exit 1; fi',
            # No remotes.
            f'if [ -n "$(git -C {q_testbed} remote)" ]; then '
            'echo "LEAK-GUARD: remote still present" >&2; exit 1; fi',
            # No refs (detached HEAD is not a ref, so this stays empty).
            f'if [ -n "$(git -C {q_testbed} for-each-ref)" ]; then '
            'echo "LEAK-GUARD: refs still present" >&2; exit 1; fi',
            # Nothing reachable names test_commit.
            f"if git -C {q_testbed} rev-list --all --objects 2>/dev/null | grep -q {q_test}; then "
            'echo "LEAK-GUARD: test_commit reachable" >&2; exit 1; fi',
            # No dangling/unreachable copy of test_commit survived the prune.
            f"if git -C {q_testbed} fsck --unreachable --no-reflogs 2>/dev/null | grep -q {q_test}; then "
            'echo "LEAK-GUARD: test_commit recoverable via fsck" >&2; exit 1; fi',
            # The on-disk clone source is gone.
            f"if [ -e {q_repo_source} ]; then "
            'echo "LEAK-GUARD: repo_source still present" >&2; exit 1; fi',
        ]
    )


def reinject_commands(test_commit, testbed=TESTBED, bundle=CONTAINER_BUNDLE, ref=BUNDLE_REF):
    """Commands that restore ``test_commit`` from the host-held bundle at grade time."""
    q_testbed = shlex.quote(testbed)
    q_bundle = shlex.quote(bundle)
    ref = shlex.quote(ref)
    q_test = shlex.quote(test_commit)
    return [
        f"git -C {q_testbed} fetch --no-tags {q_bundle} {ref}:{ref}",
        # Confirm the object is back before the caller's reset --hard <test_commit>.
        f"git -C {q_testbed} cat-file -e {q_test}^{{commit}}",
    ]
