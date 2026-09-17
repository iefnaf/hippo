"""Tests for shared contract conventions: schema_version and time formats."""

from __future__ import annotations

import json

import pytest

from eval.contracts.common import (
    SCHEMA_VERSION,
    ContractError,
    SchemaVersionedModel,
    parse_dataset_time,
    parse_run_timestamp,
)
from eval.contracts.internal import RawEvidenceArtifact


class TestSchemaVersion:
    def test_first_version_is_one(self):
        assert SCHEMA_VERSION == 1

    def test_artifacts_default_to_version_one(self):
        artifact = RawEvidenceArtifact(
            run_id="r1", sample_handle="s1", evidence=[]
        )
        assert artifact.schema_version == 1
        dumped = json.loads(artifact.dump_json())
        assert dumped["schema_version"] == 1

    def test_wrong_schema_version_is_refused_with_location(self):
        raw = json.dumps(
            {
                "schema_version": 2,
                "run_id": "r1",
                "sample_handle": "s1",
                "evidence": [],
            }
        )
        with pytest.raises(ContractError) as excinfo:
            RawEvidenceArtifact.load_json(raw)
        assert excinfo.value.code == "unsupported_schema_version"
        assert excinfo.value.location == "/schema_version"

    def test_missing_schema_version_is_refused(self):
        raw = json.dumps({"run_id": "r1", "sample_handle": "s1", "evidence": []})
        with pytest.raises(ContractError) as excinfo:
            RawEvidenceArtifact.load_json(raw)
        assert excinfo.value.code == "unsupported_schema_version"

    def test_round_trip_preserves_schema_version(self):
        artifact = RawEvidenceArtifact(
            run_id="r1", sample_handle="s1", evidence=[]
        )
        loaded = RawEvidenceArtifact.load_json(artifact.dump_json())
        assert loaded == artifact


class TestDatasetTimeParsing:
    @pytest.mark.parametrize(
        "value",
        ["2026-09-06", "2026-09-06T12:30:00Z", "2026-09-06T12:30:00+08:00"],
    )
    def test_valid_formats(self, value):
        parsed, mode = parse_dataset_time(value)
        assert mode in ("date", "datetime")

    @pytest.mark.parametrize(
        "value",
        [
            "2026/09/06",        # wrong separator
            "09-06-2026",        # wrong order
            "2026-9-6",          # non-padded
            "2026-09-06 12:30",  # space separator, naive
            "2026-09-06T12:30:00",  # naive datetime (no timezone)
            "not-a-date",
            "",
        ],
    )
    def test_invalid_formats_are_rejected(self, value):
        with pytest.raises(ValueError):
            parse_dataset_time(value)


class TestRunTimestampParsing:
    def test_valid_utc_timestamp(self):
        parsed = parse_run_timestamp("2026-09-06T00:00:01.250000Z")
        assert parsed.utcoffset().total_seconds() == 0

    def test_offset_timestamp_is_accepted(self):
        parsed = parse_run_timestamp("2026-09-06T08:00:00+08:00")
        assert parsed.utcoffset().total_seconds() == 8 * 3600

    def test_naive_timestamp_is_rejected(self):
        with pytest.raises(ValueError) as excinfo:
            parse_run_timestamp("2026-09-06T00:00:01")
        assert "UTC-only" in str(excinfo.value)

    def test_garbage_is_rejected(self):
        with pytest.raises(ValueError):
            parse_run_timestamp("yesterday")


class TestContractErrorStructure:
    def test_fields_and_rendering(self):
        err = ContractError(
            code="missing",
            message="field is required",
            location="/a/b",
            details={"extra": "unexpected key 'c'"},
        )
        assert err.code == "missing"
        assert err.location == "/a/b"
        assert "missing at /a/b" in str(err)
        assert err.to_dict()["code"] == "missing"

    def test_is_value_error_not_bare_runtime(self):
        # The CLI catches ContractError; validate never lets raw
        # exceptions escape as crashes.
        assert issubclass(ContractError, ValueError)
