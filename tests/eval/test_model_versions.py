"""Model version records, drift probes and drift-aware comparison
(issue #8 AC4 + #6 leftover: RunManifest code version).

Offline throughout: probes run through the fake reader (the machinery,
artifacts and comparability rules are identical for the real client;
the live probe smoke lives in test_live_smoke.py and skips without
credentials).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.compare import compare_runs
from eval.config import load_config_dict, load_config_toml
from eval.datasets.manual import ManualDataset
from eval.judges.fake import FakeJudge, FakeJudgeSpec
from eval.memories import build_memory_for_plan
from eval.readers.fake import FakeReader, FakeReaderSpec
from eval.runner import OfflineRunner
from eval.runs import RunStore
from eval.versioning import code_version

EXAMPLE_CONFIG = "eval/configs/examples/offline_fake.toml"
HANDLE = "smoke_single_session_user_0001"


class ObservedModelReader(FakeReader):
    """Fake reader that pretends a server-observed model field."""

    def __init__(self, spec: FakeReaderSpec, observed: str) -> None:
        super().__init__(spec)
        self.observed_models = {observed}


def _config(**overrides):
    data = {
        "name": "versions-test",
        "dataset_plan": "manual-fixtures@1",
        "sample_plan_id": "versions-tests",
        "sample_ids": [HANDLE],
        "smoke_subset_ids": [],
        "memory": {
            "name": "fake-memory",
            "baseline_kind": "adapter",
            "capabilities": ["extractive_evidence", "state_inspection"],
            "config": {"mutation_mode": "sync", "evidence_kinds": ["extractive"]},
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
        "evidence_token_budget": 4096,
    }
    data.update(overrides)
    return load_config_dict(data)


def _run(tmp_path: Path, config, run_id: str, reader=None):
    runner = OfflineRunner(
        config=config,
        dataset=ManualDataset.load_default(),
        adapter=build_memory_for_plan(config.memory),
        reader=reader or FakeReader(FakeReaderSpec.from_reader_plan(config.reader)),
        judge=FakeJudge(FakeJudgeSpec.from_judge_plan(config.judge)),
        store=RunStore(tmp_path / "runs", run_id),
        run_id=run_id,
    )
    return runner.run()


class TestProbesAndVersionRecord:
    def test_probe_set_config_runs_probes_and_archives_outputs(self, tmp_path):
        config = _config(
            reader={
                "model": "fake-reader",
                "model_family": "family-r",
                "base_url": "offline://fake",
                "temperature": 0.0,
                "max_output_tokens": 1024,
                "tokenizer_id": "test:char-v1",
                "counting_mode": "test",
                "probe_set_id": "reader-drift-probe@1",
            }
        )
        outcome = _run(tmp_path, config, "run-probe")
        probe_path = outcome.run_dir / "artifacts" / "model_probes.json"
        assert probe_path.exists()
        probe = json.loads(probe_path.read_text(encoding="utf-8"))
        assert probe["probe_set_id"] == "reader-drift-probe@1"
        assert len(probe["calls"]) == 10  # first-version probe set size
        assert all("expected_behavior" in c for c in probe["calls"])
        assert all(c["error"] is None for c in probe["calls"])
        # Every probe carries the fixed declared question date
        assert probe["calls"][0]["prompt"]
        # version record binds the probe digest
        versions = json.loads(
            (outcome.run_dir / "model_versions.json").read_text(encoding="utf-8")
        )
        assert versions["probe"]["probe_digest"] == probe["probe_digest"]

    def test_probes_are_deterministic_for_the_fake_reader(self, tmp_path):
        config = _config()
        config = config.model_copy(
            update={
                "reader": config.reader.model_copy(
                    update={"probe_set_id": "reader-drift-probe@1"}
                )
            }
        )
        a = _run(tmp_path, config, "run-probe-a")
        b = _run(tmp_path, config, "run-probe-b")
        pa = json.loads(
            ((a.run_dir / "artifacts" / "model_probes.json").read_text())
        )
        pb = json.loads(
            ((b.run_dir / "artifacts" / "model_probes.json").read_text())
        )
        assert [c["output"] for c in pa["calls"]] == [c["output"] for c in pb["calls"]]

    def test_manifest_carries_code_version(self, tmp_path):
        outcome = _run(tmp_path, _config(), "run-codever")
        manifest = json.loads(
            (outcome.run_dir / "run.json").read_text(encoding="utf-8")
        )
        assert manifest["code_version"]  # non-empty (git sha or unknown)
        assert manifest["code_version"] == code_version()

    def test_version_record_declares_all_four_identifiers(self, tmp_path):
        config = _config()
        outcome = _run(tmp_path, config, "run-fourids")
        versions = json.loads(
            (outcome.run_dir / "model_versions.json").read_text(encoding="utf-8")
        )
        for role, model_name in (("reader", "fake-reader"), ("judge", "fake-judge")):
            record = versions[role]
            assert record["alias"] == model_name
            assert record["response_model"] is None  # fake: nothing observed
            assert record["vendor_documented_version"] == ""
            assert record["run_date"]
        assert versions["probe"] is None
        assert versions["code_version"] == code_version()

    def test_resume_keeps_original_probe_archive(self, tmp_path):
        config = _config()
        config = config.model_copy(
            update={
                "reader": config.reader.model_copy(
                    update={"probe_set_id": "reader-drift-probe@1"}
                )
            }
        )
        _run(tmp_path, config, "run-probe-resume")
        before = (tmp_path / "runs" / "run-probe-resume" / "artifacts" / "model_probes.json").read_bytes()
        runner = OfflineRunner(
            config=config,
            dataset=ManualDataset.load_default(),
            adapter=build_memory_for_plan(config.memory),
            reader=FakeReader(FakeReaderSpec.from_reader_plan(config.reader)),
            judge=FakeJudge(FakeJudgeSpec.from_judge_plan(config.judge)),
            store=RunStore(tmp_path / "runs", "run-probe-resume"),
            run_id="run-probe-resume",
        )
        runner.resume()  # probes are archived once; no error, no rewrite
        after = (tmp_path / "runs" / "run-probe-resume" / "artifacts" / "model_probes.json").read_bytes()
        assert before == after


class TestDriftComparability:
    def _observed_reader(self, config, observed: str) -> ObservedModelReader:
        return ObservedModelReader(
            FakeReaderSpec.from_reader_plan(config.reader), observed
        )

    def test_identical_observed_models_stay_same_condition(self, tmp_path):
        config = _config()
        a = _run(tmp_path, config, "run-drift-a", self._observed_reader(config, "m-1"))
        b = _run(tmp_path, config, "run-drift-b", self._observed_reader(config, "m-1"))
        payload = compare_runs(a.run_dir, b.run_dir)
        assert payload["comparability"]["same_condition"] is True

    def test_repointed_alias_breaks_same_condition(self, tmp_path):
        config = _config()
        a = _run(tmp_path, config, "run-drift-c", self._observed_reader(config, "m-1"))
        b = _run(tmp_path, config, "run-drift-d", self._observed_reader(config, "m-2"))
        payload = compare_runs(a.run_dir, b.run_dir)
        comp = payload["comparability"]
        assert comp["same_condition"] is False
        drift = [d for d in comp["differences"] if "response_model" in d["field"]]
        assert drift and drift[0]["field"] == "/model_versions/reader/response_model"
        assert payload["comparison_kind"] == "not_same_condition"

    def test_probe_digest_mismatch_breaks_same_condition(self, tmp_path):
        # Two runs with the same probe-set id but different digests can
        # only happen if the set itself changed: refuse to align.
        config = _config()
        a = _run(tmp_path, config, "run-probe-diff-a")
        b = _run(tmp_path, config, "run-probe-diff-b")
        for run_id, digest in (("run-probe-diff-a", "d1"), ("run-probe-diff-b", "d2")):
            path = tmp_path / "runs" / run_id / "model_versions.json"
            doc = json.loads(path.read_text(encoding="utf-8"))
            doc["probe"] = {
                "run_id": run_id,
                "probe_set_id": "reader-drift-probe@1",
                "probe_digest": digest,
                "created_at": doc["created_at"],
                "calls": [
                    {
                        "prompt_id": "p",
                        "prompt": "p",
                        "expected_behavior": "p",
                        "output": "ok",
                    }
                ],
            }
            path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        payload = compare_runs(a.run_dir, b.run_dir)
        comp = payload["comparability"]
        assert comp["same_condition"] is False
        assert any(
            d["field"] == "/model_versions/probe/probe_digest"
            for d in comp["differences"]
        )


class TestCountingModeSeparation:
    def test_estimated_and_test_counting_never_compare_as_same_condition(
        self, tmp_path
    ):
        base_reader = {
            "model": "fake-reader",
            "model_family": "family-r",
            "base_url": "offline://fake",
            "temperature": 0.0,
            "max_output_tokens": 1024,
        }
        test_cfg = _config(
            reader={**base_reader, "tokenizer_id": "test:char-v1", "counting_mode": "test"}
        )
        est_cfg = _config(
            name="versions-test-estimated",
            reader={
                **base_reader,
                "tokenizer_id": "estimated:chars4-v1",
                "counting_mode": "estimated",
            },
        )
        a = _run(tmp_path, test_cfg, "run-count-test")
        b = _run(tmp_path, est_cfg, "run-count-est")
        payload = compare_runs(a.run_dir, b.run_dir)
        comp = payload["comparability"]
        assert comp["same_condition"] is False
        fields = {d["field"] for d in comp["differences"]}
        assert "/reader/counting_mode" in fields
        assert "/reader/tokenizer_id" in fields

    def test_report_marks_estimated_mode_limitation(self, tmp_path):
        est_cfg = _config(
            reader={
                "model": "fake-reader",
                "model_family": "family-r",
                "base_url": "offline://fake",
                "temperature": 0.0,
                "max_output_tokens": 1024,
                "tokenizer_id": "estimated:chars4-v1",
                "counting_mode": "estimated",
            }
        )
        outcome = _run(tmp_path, est_cfg, "run-est-limit")
        report = json.loads(
            (outcome.run_dir / "report.json").read_text(encoding="utf-8")
        )
        assert report["header"]["counting_mode"] == "estimated"
        assert any("估算计数模式" in note for note in report["limitations"])
