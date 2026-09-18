"""Harness code version identification for run manifests.

Issue #6 leftover: RunManifest carries a code version so a run can be
traced to the exact harness code that produced it. The value is the
git HEAD of the repository checkout plus a "+dirty" marker when the
working tree has uncommitted changes (a dirty tree is recorded, not
hidden); outside a git checkout the value is "unknown".
"""

from __future__ import annotations

import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _git(args: list[str]) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def code_version() -> str:
    """git HEAD (with +dirty marker) or "unknown" outside a checkout."""
    head = _git(["rev-parse", "HEAD"])
    if head is None:
        return "unknown"
    status = _git(["status", "--porcelain"])
    if status:
        return f"{head}+dirty"
    return head
