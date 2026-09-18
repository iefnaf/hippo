"""Real-data gate for issue #7: pinned file, split artifacts, smoke
resolution and isolation on the actual LongMemEval-S bytes.

The whole module skips when the pinned data file has not been fetched
(CI runs without the 277MB download):

    uv run python scripts/fetch_longmemeval.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.config import load_config_toml
from eval.datasets.longmemeval import (
    DATASET_PLAN,
    DEFAULT_LONGMEMEVAL_S_PATH,
    FILE_SHA256,
    FILE_SIZE_BYTES,
    LongMemEvalDataset,
    check_pinned_expectations,
    validate_dataset,
)
from eval.datasets.split import SPLIT_SEED, SplitPlan
from scripts.fetch_longmemeval import sha256_file

pytestmark = pytest.mark.skipif(
    not DEFAULT_LONGMEMEVAL_S_PATH.exists(),
    reason=(
        "pinned LongMemEval-S data not fetched; run "
        "`uv run python scripts/fetch_longmemeval.py` to enable real-data tests"
    ),
)

SPLIT_JSON = (
    Path(__file__).resolve().parents[2]
    / "eval"
    / "datasets"
    / "splits"
    / "longmemeval_s_cleaned"
    / "split.json"
)
SMOKE_JSON = SPLIT_JSON.parent / "smoke.json"
REAL_SMOKE_CONFIG = "eval/configs/examples/real_smoke_offline.toml"

#: Adapter-visible session/message key whitelist (dataset contracts).
SESSION_KEYS = {"session_id", "occurred_at", "messages"}
MESSAGE_KEYS = {"msg_id", "role", "content"}


@pytest.fixture(scope="module")
def dataset() -> LongMemEvalDataset:
    return LongMemEvalDataset.load_default()


@pytest.fixture(scope="module")
def summary(dataset: LongMemEvalDataset):
    return validate_dataset(dataset)


class TestPinnedFile:
    def test_checksum_matches_pin(self):
        assert sha256_file(DEFAULT_LONGMEMEVAL_S_PATH) == FILE_SHA256
        assert DEFAULT_LONGMEMEVAL_S_PATH.stat().st_size == FILE_SIZE_BYTES

    def test_pin_expectations_hold(self, summary):
        check_pinned_expectations(summary)  # raises on any mismatch
        assert summary.total_questions == 500
        assert summary.abstention_questions == 30
        assert summary.question_type_counts == {
            "knowledge-update": 78,
            "multi-session": 133,
            "single-session-assistant": 56,
            "single-session-preference": 30,
            "single-session-user": 70,
            "temporal-reasoning": 133,
        }
        # The upstream has_answer annotation is still present in the file
        # (the cleaning whitelist must drop it, not the file).
        assert summary.turns_with_has_answer > 0

    def test_provenance_record_exists_next_to_data(self):
        prov = json.loads(
            (DEFAULT_LONGMEMEVAL_S_PATH.parent / "provenance.json").read_text(
                encoding="utf-8"
            )
        )
        assert prov["sha256"] == FILE_SHA256
        assert prov["revision"]
        assert prov["source_url"]
        assert prov["license"] == "MIT"
        assert prov["verified_on"]


class TestIsolationOnRealData:
    def test_no_official_id_or_marker_in_any_written_object(self, dataset):
        """AC: answer-relevant upstream ids and annotation fields stay out
        of every adapter-visible object, across ALL 500 questions.

        The structural guarantee is the role/content key whitelist (an
        annotation FIELD cannot exist on a forwarded object); the opaque
        upstream ids are additionally asserted absent from the content,
        which holds on the pinned file (natural text never contains them).
        """
        for handle in dataset.sample_handles:
            entry = dataset._entries[handle]
            official_ids = {handle, *entry["haystack_session_ids"]}
            official_ids.update(entry["answer_session_ids"])
            sessions = dataset.iter_sessions(handle)
            for session in sessions:
                dumped = session.model_dump(mode="json")
                assert set(dumped) == SESSION_KEYS
                for message in dumped["messages"]:
                    assert set(message) == MESSAGE_KEYS
            blob = json.dumps(
                [s.model_dump(mode="json") for s in sessions],
                ensure_ascii=False,
            )
            for official in official_ids:
                assert official not in blob, (handle, official)

    def test_gold_and_id_mapping_complete_for_every_question(self, dataset):
        """AC: non-abstention gold sources and ID mappings resolve; a
        missing one would raise (invalid_input semantics), never skip."""
        abstention = 0
        for handle in dataset.sample_handles:
            scoring = dataset.get_scoring_data(handle)
            internal = {s.session_id for s in dataset.iter_sessions(handle)}
            assert set(scoring.gold_source_ids) <= internal
            assert set(scoring.internal_to_official_session) == internal
            if scoring.is_abstention:
                abstention += 1
                continue  # recall N/A by flag; gold may be present or empty
            assert scoring.gold_source_ids, handle
        assert abstention == 30


class TestCommittedSplitArtifacts:
    def test_split_artifacts_regenerate_identically(self):
        """AC: the committed split + smoke manifests are reproducible from
        the pinned data with the committed seed."""
        from scripts.make_split import CONFIG_PATH, SPLIT_DIR, build_plans, _render

        split, smoke = build_plans(DEFAULT_LONGMEMEVAL_S_PATH)
        committed_split = SplitPlan.load_json(
            (SPLIT_DIR / "split.json").read_text(encoding="utf-8")
        )
        from eval.datasets.split import SmokePlan

        committed_smoke = SmokePlan.load_json(
            (SMOKE_JSON).read_text(encoding="utf-8")
        )
        assert committed_split == split
        assert committed_smoke == smoke
        # and the on-disk bytes match a fresh render (formatting included)
        rendered = _render(split, smoke)
        assert (SPLIT_DIR / "split.json").read_text(encoding="utf-8") == rendered[
            "split.json"
        ]
        assert SMOKE_JSON.read_text(encoding="utf-8") == rendered["smoke.json"]
        assert CONFIG_PATH.read_text(encoding="utf-8") == rendered[
            CONFIG_PATH.name
        ]

    def test_split_invariants_and_stratification(self, dataset):
        split = SplitPlan.load_json(SPLIT_JSON.read_text(encoding="utf-8"))
        assert split.dataset_plan == DATASET_PLAN
        assert split.seed == SPLIT_SEED
        assert split.source_sha256 == FILE_SHA256
        assert len(split.dev_ids) == 50
        assert len(split.holdout_ids) == 450
        dev, hold = set(split.dev_ids), set(split.holdout_ids)
        assert not dev & hold
        assert dev | hold == set(dataset.sample_handles)
        # stratified by question_type and abstention on BOTH sides
        items = {i.handle: i for i in dataset.iter_split_items()}
        all_types = {i.question_type for i in items.values()}
        assert {items[h].question_type for h in dev} == all_types
        assert {items[h].question_type for h in hold} == all_types
        assert any(items[h].is_abstention for h in dev)
        assert any(items[h].is_abstention for h in hold)
        for key, count in split.strata.items():
            assert count.total == sum(
                1
                for i in items.values()
                if (
                    i.question_type,
                    "abs" if i.is_abstention else "ans",
                )
                == tuple(key.rsplit("|", 1))
            )

    def test_smoke_ids_resolvable_and_semantically_aligned(self, dataset):
        """AC: the 8 smoke ids resolve on the real dataset with the M1
        smoke semantics: six question types + two abstention, dev-only."""
        from eval.datasets.split import SmokePlan

        smoke = SmokePlan.load_json(SMOKE_JSON.read_text(encoding="utf-8"))
        split = SplitPlan.load_json(SPLIT_JSON.read_text(encoding="utf-8"))
        dataset.require_handles(list(smoke.smoke_ids))
        assert set(smoke.smoke_ids) <= set(split.dev_ids)
        types = []
        for handle in smoke.smoke_ids:
            scoring = dataset.get_scoring_data(handle)
            sessions = dataset.iter_sessions(handle)
            question = dataset.get_question(handle)
            assert sessions, handle
            assert question.question_date  # contract-parses by construction
            if scoring.is_abstention:
                continue
            internal = {s.session_id for s in sessions}
            assert set(scoring.gold_source_ids) <= internal
            types.append(scoring.question_type)
        assert sorted(types) == sorted(
            {
                "single-session-user",
                "single-session-assistant",
                "single-session-preference",
                "temporal-reasoning",
                "knowledge-update",
                "multi-session",
            }
        )
        abst = [
            h for h in smoke.smoke_ids if dataset.get_scoring_data(h).is_abstention
        ]
        assert len(abst) == 2


class TestRealSmokeRun:
    def test_offline_fake_run_over_real_smoke_ids(self, dataset, tmp_path):
        """The 8 real smoke questions flow through the full offline loop
        (ingest -> reopen -> retrieve -> prepare -> read -> judge) with
        fake components, and the adapter-visible objects of the run stay
        free of upstream ids and annotation fields."""
        from eval.judges.fake import FakeJudge, FakeJudgeSpec
        from eval.memories.fake import FakeMemoryAdapter, FakeMemorySpec
        from eval.readers.fake import FakeReader, FakeReaderSpec
        from eval.runner import OfflineRunner
        from eval.runs import RunStore

        config = load_config_toml(REAL_SMOKE_CONFIG)
        runner = OfflineRunner(
            config=config,
            dataset=dataset,
            adapter=FakeMemoryAdapter(FakeMemorySpec.from_memory_plan(config.memory)),
            reader=FakeReader(FakeReaderSpec.from_reader_plan(config.reader)),
            judge=FakeJudge(FakeJudgeSpec.from_judge_plan(config.judge)),
            store=RunStore(tmp_path / "runs", "run-real-smoke"),
            run_id="run-real-smoke",
        )
        outcome = runner.run()
        assert outcome.failed == 0
        assert outcome.invalid_input == 0
        assert len(outcome.results) == 8

        # AC: no answer-relevant upstream id in the written objects (ingest
        # inputs) or any adapter-visible object (namespaces, operation ids,
        # retrieval requests) of the run; annotation fields cannot exist
        # structurally (whitelisted session/message keys).
        official = set()
        for handle in config.sample_ids:
            entry = dataset._entries[handle]
            official.add(handle)
            official.update(entry["haystack_session_ids"])
            official.update(entry["answer_session_ids"])
        for artifacts_dir in sorted((outcome.run_dir / "artifacts").iterdir()):
            log = json.loads(
                (artifacts_dir / "attempts.json").read_text(encoding="utf-8")
            )
            for entry in log["entries"]:
                if entry["stage"] not in ("ingest", "await_ready", "retrieve"):
                    continue  # judge/scoring inputs are private by design
                blob = json.dumps(entry, ensure_ascii=False)
                for oid in official:
                    assert oid not in blob, (artifacts_dir.name, oid)
                if entry["stage"] == "ingest":
                    session = entry["input"]["session"]
                    assert set(session) == SESSION_KEYS
                    for message in session["messages"]:
                        assert set(message) == MESSAGE_KEYS
