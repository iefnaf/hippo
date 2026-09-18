#!/usr/bin/env python
"""Fetch and verify the pinned DeepSeek offline tokenizer (issue #8).

The pin record lives in eval/prepare/tokens.py (repo, revision, sha256,
size, license, verification date). Exact token counting
(counting_mode='exact') loads this file; loading re-verifies the sha256.

1. skips the download when the destination already matches the pinned
   sha256 (idempotent; safe to re-run after any failure);
2. otherwise streams the revision-pinned URL to <dest>.part, verifies
   sha256 and size, then moves it into place atomically;
3. writes provenance.json next to the file recording source link,
   revision, checksum, license and fetch time;
4. loads the tokenizer through the harness path (sha check + a smoke
   count) so the fetch is verified end to end.

The tokenizer file never enters Git (data/ is gitignored).

Usage:
    uv run python scripts/fetch_deepseek_tokenizer.py             # fetch + verify
    uv run python scripts/fetch_deepseek_tokenizer.py --verify-only
    uv run python scripts/fetch_deepseek_tokenizer.py --force      # re-download
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

from eval.contracts.common import now_utc
from eval.prepare.tokens import (
    DEEPSEEK_TOKENIZER_LICENSE,
    DEEPSEEK_TOKENIZER_REPO,
    DEEPSEEK_TOKENIZER_REVISION,
    DEEPSEEK_TOKENIZER_SHA256,
    DEEPSEEK_TOKENIZER_SIZE_BYTES,
    DEEPSEEK_TOKENIZER_URL,
    DEEPSEEK_TOKENIZER_VERIFIED_ON,
    DEFAULT_DEEPSEEK_TOKENIZER_PATH,
    DeepSeekOfflineTokenizer,
)

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
    fetched_at: str,
) -> Path:
    doc: dict[str, Any] = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "source_repo": f"https://huggingface.co/{DEEPSEEK_TOKENIZER_REPO}",
        "source_url": DEEPSEEK_TOKENIZER_URL,
        "revision": DEEPSEEK_TOKENIZER_REVISION,
        "file": dest.name,
        "sha256": checksum,
        "size_bytes": size_bytes,
        "license": DEEPSEEK_TOKENIZER_LICENSE,
        "verified_on": DEEPSEEK_TOKENIZER_VERIFIED_ON,
        "downloaded_this_run": downloaded,
        "fetched_at": fetched_at,
    }
    provenance = dest.parent / "provenance.json"
    provenance.write_text(
        json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return provenance


def fetch(
    dest: Path | None = None,
    *,
    force: bool = False,
    verify_only: bool = False,
    downloader: Downloader = default_downloader,
) -> dict[str, Any]:
    dest = dest or DEFAULT_DEEPSEEK_TOKENIZER_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    downloaded = False

    if dest.exists() and not force:
        checksum = sha256_file(dest)
        if checksum == DEEPSEEK_TOKENIZER_SHA256:
            provenance = _write_provenance(
                dest,
                downloaded=False,
                checksum=checksum,
                size_bytes=dest.stat().st_size,
                fetched_at=now_utc(),
            )
            return {"status": "present", "dest": dest, "provenance": provenance}
        if verify_only:
            raise SystemExit(
                f"error: {dest} exists with sha256 {checksum} but the pin "
                f"expects {DEEPSEEK_TOKENIZER_SHA256}; re-run with --force"
            )
        print(f"checksum mismatch for {dest}; re-downloading", file=sys.stderr)

    if verify_only:
        raise SystemExit(
            f"error: pinned tokenizer missing at {dest}; run without "
            "--verify-only to fetch it"
        )

    part = dest.with_suffix(dest.suffix + ".part")
    downloader(DEEPSEEK_TOKENIZER_URL, part)
    checksum = sha256_file(part)
    if checksum != DEEPSEEK_TOKENIZER_SHA256:
        part.unlink(missing_ok=True)
        raise SystemExit(
            f"error: downloaded tokenizer has sha256 {checksum} but the "
            f"pin expects {DEEPSEEK_TOKENIZER_SHA256}; the upstream file "
            "moved — update the pin record in eval/prepare/tokens.py"
        )
    size = part.stat().st_size
    if size != DEEPSEEK_TOKENIZER_SIZE_BYTES:
        part.unlink(missing_ok=True)
        raise SystemExit(
            f"error: downloaded tokenizer is {size} bytes but the pin "
            f"expects {DEEPSEEK_TOKENIZER_SIZE_BYTES}; update the pin "
            "record in eval/prepare/tokens.py"
        )
    part.replace(dest)
    downloaded = True
    provenance = _write_provenance(
        dest,
        downloaded=downloaded,
        checksum=checksum,
        size_bytes=size,
        fetched_at=now_utc(),
    )
    return {"status": "downloaded", "dest": dest, "provenance": provenance}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="never download; verify an existing file against the pin",
    )
    args = parser.parse_args(argv)

    result = fetch(force=args.force, verify_only=args.verify_only)
    dest: Path = result["dest"]

    # End-to-end verification through the harness path (sha + load + count).
    tokenizer = DeepSeekOfflineTokenizer(dest)
    sample_counts = {
        "hello_world": tokenizer.count("Hello world, this is a test."),
        "empty": tokenizer.count(""),
    }
    summary = {
        "status": result["status"],
        "dest": str(dest),
        "provenance": str(result["provenance"]),
        "tokenizer_id": tokenizer.tokenizer_id,
        "smoke_counts": sample_counts,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
