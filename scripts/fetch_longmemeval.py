#!/usr/bin/env python
"""Fetch and verify the pinned LongMemEval-S cleaned data file (issue #7).

The pinned source record lives in eval/datasets/longmemeval.py (source
URL, revision, sha256, size, license, verification date). This script:

1. skips the download when the destination already matches the pinned
   sha256 (idempotent; safe to re-run after any failure);
2. otherwise streams the revision-pinned URL to <dest>.part, verifies the
   sha256 (and size) and moves it into place atomically;
3. writes data/longmemeval/provenance.json recording the source link,
   revision, checksum, license and verification date of this fetch;
4. runs the full dataset validation (500 questions, 30 abstention,
   pinned field list) and prints the summary.

The data file never enters Git (data/ is gitignored).

Usage:
    uv run python scripts/fetch_longmemeval.py            # fetch + verify
    uv run python scripts/fetch_longmemeval.py --validate-only
    uv run python scripts/fetch_longmemeval.py --force     # re-download
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Any, Callable

from eval.contracts.common import ContractError, now_utc
from eval.datasets.longmemeval import (
    DATASET_LICENSE,
    DATASET_PLAN,
    DEFAULT_LONGMEMEVAL_S_PATH,
    FILE_NAME,
    FILE_SHA256,
    FILE_SIZE_BYTES,
    SOURCE_FILE_URL,
    SOURCE_REPO_URL,
    SOURCE_REVISION,
    UPSTREAM_CODE_URL,
    VERIFIED_ON,
    check_pinned_expectations,
    validate_dataset,
)
from eval.datasets.longmemeval import LongMemEvalDataset

PROVENANCE_SCHEMA_VERSION = 1

Downloader = Callable[[str, Path], None]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_downloader(url: str, dest: Path) -> None:
    """Stream the pinned URL to dest (follows the CDN redirect)."""
    with urllib.request.urlopen(url, timeout=120) as response, dest.open("wb") as fh:
        shutil.copyfileobj(response, fh, length=1 << 20)


def _write_provenance(
    dest: Path,
    *,
    downloaded: bool,
    checksum: str,
    size_bytes: int,
    validation: dict[str, Any] | None,
    fetched_at: str,
) -> Path:
    record = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "dataset_plan": DATASET_PLAN,
        "source_url": SOURCE_FILE_URL,
        "source_repo": SOURCE_REPO_URL,
        "upstream_code": UPSTREAM_CODE_URL,
        "revision": SOURCE_REVISION,
        "file": FILE_NAME,
        "sha256": checksum,
        "size_bytes": size_bytes,
        "license": DATASET_LICENSE,
        "verified_on": VERIFIED_ON,
        "fetched_at": fetched_at,
        "downloaded_this_run": downloaded,
        "validation": validation,
    }
    path = dest.parent / "provenance.json"
    path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def fetch(
    dest: str | Path = DEFAULT_LONGMEMEVAL_S_PATH,
    *,
    downloader: Downloader = default_downloader,
    url: str = SOURCE_FILE_URL,
    expected_sha256: str = FILE_SHA256,
    expected_size: int | None = FILE_SIZE_BYTES,
    validate: bool = True,
    force: bool = False,
    validate_only: bool = False,
    now: Callable[[], str] = now_utc,
) -> dict[str, Any]:
    """Fetch/verify the pinned file; returns a JSON-serializable outcome.

    `downloader`, `url`, `expected_*` and `now` are injection points for
    offline tests; production callers use the module pins.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    downloaded = False
    skipped_reason: str | None = None

    if dest.exists() and not force:
        actual = sha256_file(dest)
        if actual == expected_sha256:
            skipped_reason = "already-present-and-verified"
    if dest.exists() and validate_only and skipped_reason is None:
        raise ContractError(
            code="validate_only_checksum_mismatch",
            message=(
                f"{dest} hashes to {sha256_file(dest)} but the pin expects "
                f"{expected_sha256}; --validate-only refuses to download"
            ),
            location=str(dest),
        )

    if skipped_reason is None and not validate_only:
        tmp = dest.with_name(dest.name + ".part")
        if tmp.exists():
            tmp.unlink()  # idempotent re-run after an interrupted download
        downloader(url, tmp)
        actual = sha256_file(tmp)
        problems: list[str] = []
        if actual != expected_sha256:
            problems.append(f"sha256 {actual} != pinned {expected_sha256}")
        if expected_size is not None and tmp.stat().st_size != expected_size:
            problems.append(
                f"size {tmp.stat().st_size} != pinned {expected_size}"
            )
        if problems:
            tmp.unlink()
            raise ContractError(
                code="download_checksum_mismatch",
                message=(
                    "the downloaded file does not match the pinned revision: "
                    + "; ".join(problems)
                ),
                location=str(tmp),
            )
        tmp.replace(dest)
        downloaded = True

    if not dest.exists():
        raise ContractError(
            code="dataset_missing",
            message=f"{dest} is still missing after fetch",
            location=str(dest),
        )

    validation: dict[str, Any] | None = None
    if validate:
        summary = validate_dataset(LongMemEvalDataset.from_file(dest))
        check_pinned_expectations(summary)
        validation = json.loads(summary.model_dump_json())

    provenance = _write_provenance(
        dest,
        downloaded=downloaded,
        checksum=expected_sha256,
        size_bytes=dest.stat().st_size,
        validation=validation,
        fetched_at=now(),
    )
    return {
        "dest": str(dest),
        "downloaded": downloaded,
        "skipped_reason": skipped_reason,
        "sha256": expected_sha256,
        "provenance": str(provenance),
        "validation": validation,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fetch_longmemeval",
        description=(
            "Fetch and verify the pinned LongMemEval-S cleaned file "
            "(data stays out of Git; provenance is recorded next to it)"
        ),
    )
    parser.add_argument(
        "--dest",
        default=None,
        metavar="PATH",
        help=f"destination file (default: {DEFAULT_LONGMEMEVAL_S_PATH})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download even when the destination already verifies",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="verify an existing file without downloading",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="skip the full dataset validation pass (checksum still enforced)",
    )
    args = parser.parse_args(argv)
    try:
        outcome = fetch(
            args.dest or DEFAULT_LONGMEMEVAL_S_PATH,
            validate=not args.no_validate,
            force=args.force,
            validate_only=args.validate_only,
        )
    except ContractError as exc:
        print(
            f"error: {exc.code} at {exc.location or '(root)'}: {exc.message}",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(outcome, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
