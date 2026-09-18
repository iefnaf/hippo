"""Fetch script: idempotent download, checksum enforcement, provenance.

All tests run offline with an injected downloader; the production pins
themselves are verified against the real file by the skipped-without-
data tests in test_longmemeval_real.py.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from eval.contracts.common import ContractError
from scripts.fetch_longmemeval import fetch, sha256_file


def _synthetic_dataset_bytes() -> bytes:
    # Minimal structurally valid entry (same shape as the adapter tests).
    entry = {
        "question_id": "0a0a0a0a",
        "question_type": "single-session-user",
        "question": "Which package manager?",
        "question_date": "2023/05/21 (Sun) 14:32",
        "answer": "pnpm",
        "haystack_session_ids": ["answer_s1"],
        "haystack_dates": ["2023/05/21 (Sun) 08:00"],
        "haystack_sessions": [[{"role": "user", "content": "pnpm", "has_answer": True}]],
        "answer_session_ids": ["answer_s1"],
    }
    return json.dumps([entry]).encode("utf-8")


class RecordingDownloader:
    """Writes fixed bytes and counts how often it was called."""

    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.calls = 0

    def __call__(self, url: str, dest: Path) -> None:
        self.calls += 1
        dest.write_bytes(self.payload)


class ExplodingDownloader:
    def __call__(self, url: str, dest: Path) -> None:
        raise AssertionError("download must be skipped when the file verifies")


def _fetch_ok_args(payload: bytes, **overrides):
    args = dict(
        downloader=RecordingDownloader(payload),
        url="https://example.test/pinned/longmemeval_s_cleaned.json",
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_size=len(payload),
        validate=False,
        now=lambda: "2026-09-18T00:00:00.000000Z",
    )
    args.update(overrides)
    return args


class TestFetchAndVerify:
    def test_download_verifies_and_records_provenance(self, tmp_path: Path):
        payload = _synthetic_dataset_bytes()
        dest = tmp_path / "data" / "longmemeval_s_cleaned.json"
        outcome = fetch(dest, **_fetch_ok_args(payload))
        assert outcome["downloaded"] is True
        assert dest.read_bytes() == payload
        prov = json.loads(Path(outcome["provenance"]).read_text(encoding="utf-8"))
        assert prov["sha256"] == hashlib.sha256(payload).hexdigest()
        assert prov["source_url"].endswith("longmemeval_s_cleaned.json")
        assert prov["revision"]
        assert prov["license"] == "MIT"
        assert prov["verified_on"]
        assert prov["fetched_at"] == "2026-09-18T00:00:00.000000Z"
        assert prov["downloaded_this_run"] is True
        assert not list(dest.parent.glob("*.part"))

    def test_second_run_is_idempotent_no_download(self, tmp_path: Path):
        payload = _synthetic_dataset_bytes()
        dest = tmp_path / "data" / "longmemeval_s_cleaned.json"
        args = _fetch_ok_args(payload)
        fetch(dest, **args)
        outcome = fetch(
            dest,
            downloader=ExplodingDownloader(),  # must not be called
            expected_sha256=args["expected_sha256"],
            expected_size=args["expected_size"],
            validate=False,
            now=args["now"],
        )
        assert outcome["downloaded"] is False
        assert outcome["skipped_reason"] == "already-present-and-verified"
        prov = json.loads(Path(outcome["provenance"]).read_text(encoding="utf-8"))
        assert prov["downloaded_this_run"] is False

    def test_checksum_mismatch_cleans_up_and_fails(self, tmp_path: Path):
        payload = _synthetic_dataset_bytes()
        dest = tmp_path / "longmemeval_s_cleaned.json"
        with pytest.raises(ContractError) as excinfo:
            fetch(
                dest,
                **_fetch_ok_args(payload, expected_sha256="0" * 64),
            )
        assert excinfo.value.code == "download_checksum_mismatch"
        assert not dest.exists()
        assert not list(tmp_path.glob("*.part"))

    def test_size_mismatch_fails_too(self, tmp_path: Path):
        payload = _synthetic_dataset_bytes()
        dest = tmp_path / "longmemeval_s_cleaned.json"
        with pytest.raises(ContractError, match="size"):
            fetch(dest, **_fetch_ok_args(payload, expected_size=len(payload) + 1))
        assert not dest.exists()

    def test_corrupt_existing_file_is_redownloaded(self, tmp_path: Path):
        payload = _synthetic_dataset_bytes()
        dest = tmp_path / "longmemeval_s_cleaned.json"
        dest.write_bytes(b"corrupted prefix")
        outcome = fetch(dest, **_fetch_ok_args(payload))
        assert outcome["downloaded"] is True
        assert sha256_file(dest) == hashlib.sha256(payload).hexdigest()

    def test_force_redownloads_even_when_verified(self, tmp_path: Path):
        payload = _synthetic_dataset_bytes()
        dest = tmp_path / "longmemeval_s_cleaned.json"
        args = _fetch_ok_args(payload)
        fetch(dest, **args)
        assert args["downloader"].calls == 1
        fetch(dest, **_fetch_ok_args(payload, force=True))
        assert args["downloader"].calls == 1  # same downloader instance unused
        # force with a fresh downloader triggers a second download
        fresh = RecordingDownloader(payload)
        fetch(
            dest,
            downloader=fresh,
            expected_sha256=args["expected_sha256"],
            expected_size=args["expected_size"],
            validate=False,
            force=True,
            now=args["now"],
        )
        assert fresh.calls == 1

    def test_validate_only_never_downloads(self, tmp_path: Path):
        payload = _synthetic_dataset_bytes()
        dest = tmp_path / "longmemeval_s_cleaned.json"
        args = _fetch_ok_args(payload)
        fetch(dest, **args)
        outcome = fetch(
            dest,
            downloader=ExplodingDownloader(),
            expected_sha256=args["expected_sha256"],
            validate=False,
            validate_only=True,
            now=args["now"],
        )
        assert outcome["downloaded"] is False

    def test_validate_only_rejects_mismatched_existing_file(self, tmp_path: Path):
        dest = tmp_path / "longmemeval_s_cleaned.json"
        dest.write_bytes(b"not the pinned file")
        with pytest.raises(ContractError) as excinfo:
            fetch(
                dest,
                downloader=ExplodingDownloader(),
                expected_sha256="1" * 64,
                validate=False,
                validate_only=True,
            )
        assert excinfo.value.code == "validate_only_checksum_mismatch"

    def test_missing_dest_validate_only_fails_with_dataset_missing(
        self, tmp_path: Path
    ):
        with pytest.raises(ContractError) as excinfo:
            fetch(
                tmp_path / "nope.json",
                downloader=ExplodingDownloader(),
                expected_sha256="1" * 64,
                validate=False,
                validate_only=True,
            )
        assert excinfo.value.code == "dataset_missing"

    def test_stale_part_file_is_replaced(self, tmp_path: Path):
        payload = _synthetic_dataset_bytes()
        dest = tmp_path / "longmemeval_s_cleaned.json"
        part = tmp_path / "longmemeval_s_cleaned.json.part"
        part.write_bytes(b"interrupted download")
        fetch(dest, **_fetch_ok_args(payload))
        assert dest.read_bytes() == payload
        assert not part.exists()


class TestValidationIntegration:
    def test_full_validation_records_summary_in_provenance(
        self, tmp_path: Path, monkeypatch
    ):
        # The pin check itself compares against the REAL pinned counts
        # (500/30); a synthetic 1-question file can never pass it, so this
        # positive-path test stubs only that final gate (its behaviour is
        # covered by the adapter and real-data tests).
        import scripts.fetch_longmemeval as fetch_mod

        monkeypatch.setattr(fetch_mod, "check_pinned_expectations", lambda s: None)
        payload = _synthetic_dataset_bytes()
        dest = tmp_path / "longmemeval_s_cleaned.json"
        outcome = fetch(dest, **_fetch_ok_args(payload, validate=True))
        assert outcome["validation"]["total_questions"] == 1
        assert outcome["validation"]["observed_turn_fields"] == [
            "content",
            "has_answer",
            "role",
        ]
        prov = json.loads(Path(outcome["provenance"]).read_text(encoding="utf-8"))
        assert prov["validation"]["total_questions"] == 1

    def test_no_validation_leaves_null_summary(self, tmp_path: Path):
        payload = _synthetic_dataset_bytes()
        dest = tmp_path / "longmemeval_s_cleaned.json"
        outcome = fetch(dest, **_fetch_ok_args(payload, validate=False))
        assert outcome["validation"] is None
        prov = json.loads(Path(outcome["provenance"]).read_text(encoding="utf-8"))
        assert prov["validation"] is None

    def test_pin_mismatch_fails_the_fetch(self, tmp_path: Path):
        # A structurally valid but off-pin file must not pass the gate;
        # the downloaded file stays for inspection but no provenance is
        # recorded for a failed verification.
        payload = _synthetic_dataset_bytes()
        dest = tmp_path / "longmemeval_s_cleaned.json"
        with pytest.raises(ContractError) as excinfo:
            fetch(dest, **_fetch_ok_args(payload, validate=True))
        assert excinfo.value.code == "dataset_pin_mismatch"
        assert dest.exists()  # bytes verified against the injected sha
        assert not (dest.parent / "provenance.json").exists()
