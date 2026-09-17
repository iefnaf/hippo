"""Tests for experiment config loading and config fingerprinting."""

from __future__ import annotations

import json

import pytest

from eval.config import (
    ConfigArtifact,
    ExperimentConfig,
    load_config_dict,
    load_config_toml,
)
from eval.contracts.common import ContractError
from eval.metrics import REGISTRY_CONTENT_VERSION, REGISTRY_VERSION

CONFIG_ROOT = "eval/configs/examples"


def _valid_config_dict() -> dict:
    return {
        "name": "offline-fake-smoke",
        "dataset_plan": "manual-fixtures@1",
        "sample_plan_id": "smoke-offline-8",
        "sample_ids": [f"sample_{i:04d}" for i in range(8)],
        "smoke_subset_ids": [f"sample_{i:04d}" for i in range(8)],
        "memory": {
            "name": "fake-memory",
            "baseline_kind": "adapter",
            "capabilities": ["extractive_evidence"],
        },
        "reader": {
            "model": "fake-reader",
            "model_family": "family-r",
            "base_url": "offline://fake",
            "temperature": 0.0,
            "max_output_tokens": 1024,
            "tokenizer_id": "test:char-v1",
            "counting_mode": "test",
        },
        "judge": {
            "model": "fake-judge",
            "model_family": "family-j",
            "base_url": "offline://fake",
            "temperature": 0.0,
            "protocol_id": "longmemeval-yes-no@1",
            "protocol_source_commit": "0" * 40,
        },
    }


class TestConfigLoading:
    def test_example_toml_loads(self):
        config = load_config_toml(f"{CONFIG_ROOT}/offline_fake.toml")
        assert config.evidence_token_budget == 4096
        assert len(config.smoke_subset_ids) == 8

    def test_invalid_toml_gives_located_error(self, tmp_path):
        bad = tmp_path / "bad.toml"
        bad.write_text("name = [unterminated", encoding="utf-8")
        with pytest.raises(ContractError) as excinfo:
            load_config_toml(bad)
        assert excinfo.value.code == "config_invalid_toml"

    def test_missing_file_gives_located_error(self, tmp_path):
        with pytest.raises(ContractError) as excinfo:
            load_config_toml(tmp_path / "nope.toml")
        assert excinfo.value.code == "config_missing"

    def test_missing_field_gives_located_error(self):
        data = _valid_config_dict()
        del data["reader"]
        with pytest.raises(ContractError) as excinfo:
            load_config_dict(data)
        assert excinfo.value.code.startswith("config_validation")
        assert "reader" in excinfo.value.message

    def test_wrong_field_range_gives_located_error(self):
        data = _valid_config_dict()
        data["evidence_token_budget"] = 0
        with pytest.raises(ContractError) as excinfo:
            load_config_dict(data)
        assert "evidence_token_budget" in excinfo.value.message

    def test_same_family_reader_judge_rejected(self):
        data = _valid_config_dict()
        data["judge"]["model_family"] = data["reader"]["model_family"]
        with pytest.raises(ContractError) as excinfo:
            load_config_dict(data)
        assert "families" in excinfo.value.message

    def test_duplicate_sample_ids_rejected(self):
        data = _valid_config_dict()
        data["sample_ids"] = ["a", "a"]
        with pytest.raises(ContractError) as excinfo:
            load_config_dict(data)
        assert "duplicate" in excinfo.value.message

    def test_smoke_subset_must_have_eight(self):
        data = _valid_config_dict()
        data["smoke_subset_ids"] = data["sample_ids"][:6]
        with pytest.raises(ContractError) as excinfo:
            load_config_dict(data)
        assert "8" in excinfo.value.message

    def test_smoke_subset_must_be_subset_of_samples(self):
        data = _valid_config_dict()
        data["smoke_subset_ids"] = [f"other_{i}" for i in range(8)]
        with pytest.raises(ContractError) as excinfo:
            load_config_dict(data)
        assert "smoke_subset_ids" in excinfo.value.message

    def test_unknown_top_level_key_rejected(self):
        data = _valid_config_dict() | {"unknown_key": 1}
        with pytest.raises(ContractError):
            load_config_dict(data)


class TestConfigFingerprint:
    def test_same_config_same_fingerprint(self):
        one = load_config_toml(f"{CONFIG_ROOT}/offline_fake.toml")
        two = load_config_toml(f"{CONFIG_ROOT}/offline_fake.toml")
        assert one.fingerprint() == two.fingerprint()

    def test_fingerprint_changes_with_content(self):
        one = load_config_toml(f"{CONFIG_ROOT}/offline_fake.toml")
        data = json.loads(one.model_dump_json())
        data["evidence_token_budget"] = 2048
        two = ExperimentConfig.model_validate(data)
        assert one.fingerprint() != two.fingerprint()

    def test_fingerprint_includes_metric_registry_version(self):
        payload = load_config_toml(
            f"{CONFIG_ROOT}/offline_fake.toml"
        ).canonical_payload()
        assert payload["metrics_registry_version"] == REGISTRY_VERSION
        assert (
            payload["metrics_registry_content_version"]
            == REGISTRY_CONTENT_VERSION
        )

    def test_registry_version_change_breaks_fingerprint(self):
        one = load_config_toml(f"{CONFIG_ROOT}/offline_fake.toml")
        payload = one.canonical_payload()
        payload["metrics_registry_version"] = "2"
        import hashlib

        tampered = hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        assert tampered != one.fingerprint()

    def test_fingerprint_is_sha256_hex(self):
        fingerprint = load_config_toml(
            f"{CONFIG_ROOT}/offline_fake.toml"
        ).fingerprint()
        assert len(fingerprint) == 64
        int(fingerprint, 16)  # hex


class TestConfigArtifact:
    def test_snapshot_round_trip(self):
        config = load_config_toml(f"{CONFIG_ROOT}/offline_fake.toml")
        artifact = ConfigArtifact(
            config=config, config_fingerprint=config.fingerprint()
        )
        dumped = json.loads(artifact.model_dump_json())
        assert dumped["schema_version"] == 1
        assert dumped["metrics_registry_version"] == REGISTRY_VERSION

    def test_snapshot_rejects_stale_fingerprint(self):
        config = load_config_toml(f"{CONFIG_ROOT}/offline_fake.toml")
        with pytest.raises(Exception, match="mismatch"):
            ConfigArtifact(config=config, config_fingerprint="0" * 64)
