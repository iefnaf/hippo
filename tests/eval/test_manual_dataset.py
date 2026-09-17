"""Manual dataset adapter: cleaning, anonymization and isolation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.contracts.common import ContractError
from eval.datasets.manual import (
    ABSTENTION_SUFFIX,
    DEFAULT_DATASET_PATH,
    ManualDataset,
    internal_msg_id,
    internal_session_id,
    namespace_for,
)

FORBIDDEN_MARKERS = (
    "has_answer",
    "answer_session_ids",
    "evidence_session_ids",
    "question_type",
    "answer",
    "_abs",
)


@pytest.fixture()
def dataset() -> ManualDataset:
    return ManualDataset.load_default()


class TestFixtureLoading:
    def test_default_fixture_has_eight_samples(self, dataset):
        assert len(dataset.sample_handles) == 8
        assert "smoke_single_session_user_0001" in dataset.sample_handles

    def test_official_ids_carry_pollution(self):
        # The fixture itself is polluted: the upstream markers exist.
        raw = json.loads(DEFAULT_DATASET_PATH.read_text(encoding="utf-8"))
        assert any(s["question_id"].endswith("_abs") for s in raw["samples"])
        assert any("has_answer" in m for s in raw["samples"] for sess in s["sessions"] for m in sess["messages"])
        assert any("answer" in sess["session_id"] for s in raw["samples"] for sess in s["sessions"])
        assert all("answer_session_ids" in s for s in raw["samples"])

    def test_invalid_json_is_located(self, tmp_path: Path):
        bad = tmp_path / "bad.json"
        bad.write_text("{oops", encoding="utf-8")
        with pytest.raises(ContractError) as excinfo:
            ManualDataset.from_file(bad)
        assert excinfo.value.code == "invalid_json"

    def test_missing_file_is_located(self, tmp_path: Path):
        with pytest.raises(ContractError) as excinfo:
            ManualDataset.from_file(tmp_path / "nope.json")
        assert excinfo.value.code == "dataset_missing"

    def test_unknown_sample_handle_is_located(self, dataset):
        with pytest.raises(ContractError) as excinfo:
            dataset.get_question("smoke_nope_0001")
        assert excinfo.value.code == "unknown_sample_handle"

    def test_require_handles_rejects_unknown(self, dataset):
        with pytest.raises(ContractError) as excinfo:
            dataset.require_handles(["smoke_single_session_user_0001", "ghost"])
        assert excinfo.value.code == "unknown_sample_handle"


class TestCleaningAndAnonymization:
    def test_sessions_sorted_by_date_with_internal_ids(self, dataset):
        sessions = dataset.iter_sessions("smoke_single_session_user_0001")
        assert len(sessions) == 4
        dates = [s.occurred_at for s in sessions]
        assert dates == sorted(dates)
        assert all(s.session_id.startswith("s_") for s in sessions)
        assert all(m.msg_id.startswith("m_") for s in sessions for m in s.messages)

    def test_message_whitelist_drops_upstream_annotations(self, dataset):
        for handle in dataset.sample_handles:
            for session in dataset.iter_sessions(handle):
                dumped = session.model_dump(mode="json")
                assert set(dumped) == {"session_id", "occurred_at", "messages"}
                for message in dumped["messages"]:
                    assert set(message) == {"msg_id", "role", "content"}

    def test_no_upstream_marker_in_adapter_visible_objects(self, dataset):
        official_ids = set()
        raw = json.loads(DEFAULT_DATASET_PATH.read_text(encoding="utf-8"))
        for sample in raw["samples"]:
            official_ids.add(sample["question_id"])
            official_ids.update(sess["session_id"] for sess in sample["sessions"])
        blob = json.dumps(
            [
                s.model_dump(mode="json")
                for handle in dataset.sample_handles
                for s in dataset.iter_sessions(handle)
            ],
            ensure_ascii=False,
        )
        for marker in FORBIDDEN_MARKERS:
            assert marker not in blob, marker
        for official in official_ids:
            assert official not in blob, official

    def test_internal_ids_stable_and_seed_independent_of_gold(self):
        a = internal_session_id("session_12_answer_7ab9")
        b = internal_session_id("session_13_haystack_2f8e")
        assert a == internal_session_id("session_12_answer_7ab9")
        assert a != b
        # A gold session and a haystack session anonymize through the
        # same pure function: no gold-dependent branching.
        assert a.startswith("s_") and len(a) == 2 + 12
        m1 = internal_msg_id("session_12_answer_7ab9", 0)
        m2 = internal_msg_id("session_12_answer_7ab9", 1)
        assert m1 != m2

    def test_namespaces_carry_no_sample_semantics(self):
        ns = namespace_for("smoke-offline-8", "smoke_abstention_0001")
        assert ns.startswith("ns_")
        assert "abstention" not in ns
        assert ns == namespace_for("smoke-offline-8", "smoke_abstention_0001")
        assert ns != namespace_for("smoke-offline-8", "smoke_multi_session_0001")


class TestQuestionAndScoringViews:
    def test_question_view_uses_dataset_question_date(self, dataset):
        ctx = dataset.get_question("smoke_single_session_user_0001")
        assert ctx.query == "这个项目现在使用什么包管理器？"
        assert ctx.question_date == "2026-09-06"

    def test_retrieval_request_carries_date_and_budget(self, dataset):
        request = dataset.build_retrieval_request(
            "smoke_knowledge_update_0001", 4096
        )
        assert request.question_date == "2026-09-10"
        assert request.evidence_token_budget == 4096
        assert request.query == dataset.get_question(
            "smoke_knowledge_update_0001"
        ).query

    def test_scoring_view_is_private_and_complete(self, dataset):
        scoring = dataset.get_scoring_data("smoke_single_session_user_0001")
        assert scoring.expected_answer == "pnpm"
        assert scoring.gold_source_ids  # non-abstention must have gold
        assert scoring.is_abstention is False
        assert scoring.question_type == "single-session-user"
        internal_ids = {
            s.session_id
            for s in dataset.iter_sessions("smoke_single_session_user_0001")
        }
        assert set(scoring.gold_source_ids) <= internal_ids
        assert set(scoring.internal_to_official_session) == internal_ids
        assert scoring.internal_to_official_session[
            scoring.gold_source_ids[0]
        ] == "session_12_answer_7ab9"

    def test_abstention_flag_lives_only_in_private_view(self, dataset):
        scoring = dataset.get_scoring_data("smoke_abstention_0001")
        assert scoring.is_abstention is True
        assert scoring.official_fields["question_id"].endswith(ABSTENTION_SUFFIX)
        # The abstention marker never reaches the adapter-visible handle or
        # sessions.
        assert "smoke_abstention_0001" in dataset.sample_handles
        blob = json.dumps(
            [
                s.model_dump(mode="json")
                for s in dataset.iter_sessions("smoke_abstention_0001")
            ],
            ensure_ascii=False,
        )
        assert "_abs" not in blob
