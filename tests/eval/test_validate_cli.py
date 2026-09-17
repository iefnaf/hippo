"""Tests for the validate command: structured, locatable errors.

Acceptance criterion: validate reports missing fields, illegal status
values, illegal time formats and wrong ranges with locatable errors
instead of raising bare exceptions.
"""

from __future__ import annotations

import json

import pytest

from eval.cli import main
from eval.config import load_config_toml
from eval.validate import validate_config, validate_fixture

EXEMPLAR_DIR = "eval/fixtures/manual"
INVALID_DIR = f"{EXEMPLAR_DIR}/invalid"


def run_cli(*argv: str, capsys) -> tuple[int, str]:
    code = main(list(argv))
    return code, capsys.readouterr().out


def run_cli_capture(*argv: str, capsys) -> tuple[int, str, str]:
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


class TestValidateCommand:
    def test_valid_config_and_fixtures_pass(self, capsys):
        code, out = run_cli(
            "validate",
            "--config", "eval/configs/examples/offline_fake.toml",
            "--fixture", f"{EXEMPLAR_DIR}/raw_evidence.exemplar.json",
            "--fixture", f"{EXEMPLAR_DIR}/result.exemplar.json",
            "--fixture", f"{EXEMPLAR_DIR}/valid.session.json",
            capsys=capsys,
        )
        assert code == 0
        assert "0 failed" in out

    def test_invalid_fixture_directory_fails_with_locations(self, capsys):
        code, out = run_cli(
            "validate",
            "--config", "eval/configs/examples/offline_fake.toml",
            "--fixture", INVALID_DIR,
            capsys=capsys,
        )
        assert code == 1
        # Every invalid fixture is listed with a located error line.
        assert "bad_offset.raw_evidence.json" in out
        assert "/evidence/0/extractive_span" in out
        assert "bad_time.retrieval_request.json" in out
        assert "/payload/question_date" in out
        assert "bad_effect.mutation_receipt.json" in out
        assert "/payload/error/effect" in out
        assert "bad_version.raw_evidence.json" in out
        assert "/schema_version" in out

    def test_invalid_config_reports_location(self, capsys):
        code, out = run_cli(
            "validate",
            "--config", "eval/configs/examples/invalid_bad_family.toml",
            capsys=capsys,
        )
        assert code == 1
        assert "model_family" in out or "families" in out

    def test_missing_config_file_structured(self, capsys):
        code, out = run_cli(
            "validate", "--config", "/nonexistent/config.toml", capsys=capsys
        )
        assert code == 1
        assert "config_missing" in out

    def test_validate_never_raises_bare_exceptions(self, tmp_path):
        # A totally broken JSON file still yields a DocumentReport, not a
        # traceback.
        broken = tmp_path / "session.broken.json"
        broken.write_text("{not json", encoding="utf-8")
        report = validate_fixture(broken)
        assert not report.ok
        assert report.errors[0].code in ("invalid_json", "unsupported_schema_version")


class TestLocatableErrors:
    def test_missing_field_is_located(self, tmp_path):
        fixture = tmp_path / "session.missing_field.json"
        fixture.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "session",
                    "payload": {
                        "session_id": "s1",
                        "messages": [],
                        # occurred_at missing
                    },
                }
            ),
            encoding="utf-8",
        )
        report = validate_fixture(fixture)
        assert not report.ok
        assert "occurred_at" in report.errors[0].message

    def test_illegal_status_is_located(self, tmp_path):
        fixture = tmp_path / "mutation_receipt.bad_status.json"
        fixture.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "mutation_receipt",
                    "payload": {
                        "operation_id": "op1",
                        "status": "maybe",
                        "memory_ids": [],
                        "sources": [],
                        "error": None,
                        "usage": None,
                    },
                }
            ),
            encoding="utf-8",
        )
        report = validate_fixture(fixture)
        assert not report.ok
        assert "/payload/status" in report.errors[0].location

    def test_illegal_time_format_is_located(self, tmp_path):
        fixture = tmp_path / "session.bad_time.json"
        fixture.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "session",
                    "payload": {
                        "session_id": "s1",
                        "occurred_at": "2026-09-03 12:00",
                        "messages": [],
                    },
                }
            ),
            encoding="utf-8",
        )
        report = validate_fixture(fixture)
        assert not report.ok
        assert "/payload/occurred_at" in report.errors[0].location

    def test_wrong_range_is_located(self, tmp_path):
        fixture = tmp_path / "evidence.bad_range.json"
        payload = {
            "kind": "extractive",
            "text": "abc",
            "extractive_span": {
                "session_id": "s1",
                "msg_id": "m1",
                "start": 4,
                "end": 2,
            },
            "derivation_sources": [],
            "source_times": ["2026-09-03"],
            "retrieval_score": None,
        }
        fixture.write_text(
            json.dumps(
                {"schema_version": 1, "kind": "evidence", "payload": payload}
            ),
            encoding="utf-8",
        )
        report = validate_fixture(fixture)
        assert not report.ok
        assert "/payload" in report.errors[0].location
        assert "greater than start" in report.errors[0].message


class TestOtherCommands:
    def test_run_prints_config_fingerprint_offline(self, capsys):
        code, out = run_cli(
            "run", "--config", "eval/configs/examples/offline_fake.toml",
            capsys=capsys,
        )
        assert code == 0
        payload = json.loads(out)
        assert payload["command"] == "run"
        assert payload["metrics_registry_version"] == "1"
        assert len(payload["config_fingerprint"]) == 64

    def test_compare_takes_run_directories_not_configs(self, capsys, tmp_path):
        # compare moved from config-level checks to run-directory
        # comparison (issue #6); pointing it at a config is refused.
        config = "eval/configs/examples/offline_fake.toml"
        code, out, err = run_cli_capture("compare", config, config, capsys=capsys)
        assert code == 2
        assert "run manifest missing" in err

    def test_compare_requires_two_positional_run_dirs(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            main(["compare", "only-one"])
        assert excinfo.value.code == 2

    def test_resume_offline(self, capsys):
        # resume is wired to run directories now: --run is required and
        # the offline plan-only stub is gone (issue #5).
        with pytest.raises(SystemExit) as excinfo:
            run_cli(
                "resume", "--config", "eval/configs/examples/offline_fake.toml",
                capsys=capsys,
            )
        assert excinfo.value.code == 2
        config = load_config_toml("eval/configs/examples/offline_fake.toml")
        assert len(config.fingerprint()) == 64


class TestProjectStructure:
    def test_required_directories_exist(self):
        from pathlib import Path

        root = Path(".")
        for rel in (
            "eval",
            "eval/datasets",
            "eval/memories",
            "eval/prepare",
            "eval/readers",
            "eval/judges",
            "eval/scorers",
            "eval/configs",
            "tests/eval",
        ):
            assert (root / rel).is_dir(), f"missing directory {rel}"
