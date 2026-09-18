"""LongMemEval-S adapter: cleaning, anonymization, isolation (issue #7).

All tests run offline on a synthetic file that mirrors the pinned
upstream shape verified against the real data: parallel haystack
arrays, `YYYY/MM/DD (Www) HH:MM` timestamps, turn-level `has_answer`
pollution, `answer_`-prefixed gold session ids, `_abs` question ids,
repeated filler sessions and integer counting answers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.config import load_config_dict
from eval.contracts.common import ContractError
from eval.datasets import load_dataset_for_config
from eval.datasets.longmemeval import (
    ABSTENTION_SUFFIX,
    DEFAULT_LONGMEMEVAL_S_PATH,
    EXPECTED_ABSTENTION_QUESTIONS,
    EXPECTED_SAMPLE_FIELDS,
    EXPECTED_TOTAL_QUESTIONS,
    FILE_SHA256,
    SOURCE_FILE_URL,
    SOURCE_REVISION,
    VERIFIED_ON,
    DATASET_LICENSE,
    LongMemEvalDataset,
    check_pinned_expectations,
    internal_msg_id,
    internal_session_id,
    normalize_lme_time,
    validate_dataset,
)
from eval.datasets.longmemeval import SplitItem


def turn(role: str, content: str, has_answer: bool | None = None) -> dict:
    t = {"role": role, "content": content}
    if has_answer is not None:
        t["has_answer"] = has_answer
    return t


def question(
    question_id: str,
    *,
    question_type: str = "single-session-user",
    answer: str | int = "pnpm",
    gold: tuple[str, ...] = ("answer_s1",),
    extra_sessions: tuple[dict, ...] = (),
    include_answer_session: bool = True,
) -> dict:
    """Build one upstream entry in the real (parallel-array) shape."""
    sessions: list[dict] = [
        {
            "sid": "sharegpt_hay_0",
            "date": "2023/05/20 (Sat) 10:00",
            "turns": [turn("user", "Tell me a story."), turn("assistant", "Once upon a time...")],
        },
        {
            "sid": "answer_s1",
            "date": "2023/05/21 (Sun) 08:00",
            "turns": [
                turn("user", "We moved the project to pnpm.", has_answer=True),
                turn("assistant", "Noted, pnpm it is.", has_answer=True),
            ],
        },
        {
            "sid": "sharegpt_hay_1",
            "date": "2023/05/21 (Sun) 12:00",
            "turns": [turn("user", "Another filler turn.")],
        },
        *extra_sessions,
    ]
    if not include_answer_session:
        sessions = [s for s in sessions if not s["sid"].startswith("answer_")]
    return {
        "question_id": question_id,
        "question_type": question_type,
        "question": "Which package manager does the project use?",
        "question_date": "2023/05/21 (Sun) 14:32",
        "answer": answer,
        "haystack_session_ids": [s["sid"] for s in sessions],
        "haystack_dates": [s["date"] for s in sessions],
        "haystack_sessions": [s["turns"] for s in sessions],
        "answer_session_ids": list(gold) if gold is not None else [],
    }


@pytest.fixture()
def dataset() -> LongMemEvalDataset:
    return LongMemEvalDataset.from_json(
        json.dumps([question("0a0a0a0a"), question("1b1b1b1b_abs", answer="You did not mention this.", gold=("answer_s2",), extra_sessions=({"sid": "answer_s2", "date": "2023/05/22 (Mon) 09:00", "turns": [turn("user", "I have a cat.")]},))])
    )


class TestNormalizeLmeTime:
    def test_timestamp_and_date_only(self):
        assert normalize_lme_time("2023/05/21 (Sun) 14:32") == "2023-05-21"
        assert normalize_lme_time("2023/05/21") == "2023-05-21"
        assert normalize_lme_time("2023/12/01 (Fri) 00:00") == "2023-12-01"

    @pytest.mark.parametrize(
        "bad",
        [
            "2023-05-21",
            "2023/5/21 (Sun) 14:32",
            "2023/05/21 (Sunday) 14:32",
            "2023/05/21 14:32",
            "23/05/2021 (Sun) 14:32",
            "not a date",
        ],
    )
    def test_rejects_other_formats(self, bad):
        with pytest.raises(ValueError):
            normalize_lme_time(bad)

    def test_rejects_impossible_calendar_date(self):
        with pytest.raises(ValueError, match="real calendar date"):
            normalize_lme_time("2023/02/30 (Thu) 10:00")


class TestLoadingAndHandles:
    def test_handles_are_official_question_ids_verbatim(self, dataset):
        # Including the abstention suffix: the real data reuses base ids
        # for abstention twins, so stripping would be ambiguous.
        assert dataset.sample_handles == ("0a0a0a0a", "1b1b1b1b_abs")

    def test_stripped_handles_would_collide(self):
        twin_a = question("0862e8bf")
        twin_b = question("0862e8bf_abs", answer="no info", gold=("answer_s2",), extra_sessions=({"sid": "answer_s2", "date": "2023/05/22 (Mon) 09:00", "turns": [turn("user", "x")]},))
        ds = LongMemEvalDataset.from_json(json.dumps([twin_a, twin_b]))
        assert len(ds.sample_handles) == 2

    def test_duplicate_question_id_rejected(self):
        raw = json.dumps([question("0a0a0a0a"), question("0a0a0a0a")])
        with pytest.raises(ContractError) as excinfo:
            LongMemEvalDataset.from_json(raw)
        assert excinfo.value.code == "duplicate_sample_handle"

    def test_invalid_json_is_located(self, tmp_path: Path):
        bad = tmp_path / "bad.json"
        bad.write_text("{oops", encoding="utf-8")
        with pytest.raises(ContractError) as excinfo:
            LongMemEvalDataset.from_file(bad)
        assert excinfo.value.code == "invalid_json"

    def test_missing_file_hints_at_fetch_script(self, tmp_path: Path):
        with pytest.raises(ContractError) as excinfo:
            LongMemEvalDataset.from_file(tmp_path / "nope.json")
        assert excinfo.value.code == "dataset_missing"
        assert "fetch_longmemeval" in excinfo.value.message

    def test_non_array_root_rejected(self):
        with pytest.raises(ContractError) as excinfo:
            LongMemEvalDataset.from_json(json.dumps({"samples": []}))
        assert excinfo.value.code == "invalid_type"

    def test_unknown_handle_is_located(self, dataset):
        with pytest.raises(ContractError) as excinfo:
            dataset.get_question("ffffffff")
        assert excinfo.value.code == "unknown_sample_handle"

    def test_require_handles_rejects_unknown(self, dataset):
        with pytest.raises(ContractError) as excinfo:
            dataset.require_handles(["0a0a0a0a", "ffffffff"])
        assert excinfo.value.code == "unknown_sample_handle"


class TestCleaningAndIsolation:
    def test_sessions_sorted_by_normalized_date(self, dataset):
        sessions = dataset.iter_sessions("0a0a0a0a")
        dates = [s.occurred_at for s in sessions]
        assert dates == sorted(dates) == ["2023-05-20", "2023-05-21", "2023-05-21"]
        # Within one day the upstream order is preserved (stable sort).
        assert sessions[1].messages[0].content == "We moved the project to pnpm."

    def test_message_whitelist_drops_upstream_annotations(self, dataset):
        for handle in dataset.sample_handles:
            for session in dataset.iter_sessions(handle):
                dumped = session.model_dump(mode="json")
                assert set(dumped) == {"session_id", "occurred_at", "messages"}
                for message in dumped["messages"]:
                    assert set(message) == {"msg_id", "role", "content"}

    def test_no_upstream_marker_or_id_in_adapter_visible_objects(self, dataset):
        official_ids = {"0a0a0a0a", "1b1b1b1b_abs", "sharegpt_hay_0", "sharegpt_hay_1", "answer_s1", "answer_s2"}
        blob = json.dumps(
            [
                s.model_dump(mode="json")
                for handle in dataset.sample_handles
                for s in dataset.iter_sessions(handle)
            ],
            ensure_ascii=False,
        )
        for marker in ("has_answer", "answer_session_ids", "haystack"):
            assert marker not in blob, marker
        for official in official_ids:
            assert official not in blob, official

    def test_abstention_marker_never_in_sessions_or_query_context(self, dataset):
        ctx = dataset.get_question("1b1b1b1b_abs")
        assert ABSTENTION_SUFFIX not in json.dumps(ctx.model_dump(), ensure_ascii=False)
        blob = json.dumps(
            [s.model_dump() for s in dataset.iter_sessions("1b1b1b1b_abs")],
            ensure_ascii=False,
        )
        assert ABSTENTION_SUFFIX not in blob

    def test_internal_ids_stable_and_slot_scoped(self):
        # The pinned file repeats filler session ids at two timestamps;
        # identity must be per haystack slot.
        a = internal_session_id("sharegpt_hay_0", 0)
        b = internal_session_id("sharegpt_hay_0", 2)
        assert a.startswith("s_") and b.startswith("s_") and a != b
        assert a == internal_session_id("sharegpt_hay_0", 0)
        assert internal_msg_id("answer_s1", 1, 0) != internal_msg_id("answer_s1", 1, 1)
        assert internal_msg_id("answer_s1", 1, 0) != internal_msg_id("answer_s1", 2, 0)

    def test_duplicated_filler_session_kept_as_two_occurrences(self):
        dup = {
            "sid": "sharegpt_hay_0",
            "date": "2023/05/22 (Mon) 09:00",
            "turns": [turn("user", "Tell me a story."), turn("assistant", "Once upon a time...")],
        }
        ds = LongMemEvalDataset.from_json(
            json.dumps([question("0a0a0a0a", extra_sessions=(dup,))])
        )
        sessions = ds.iter_sessions("0a0a0a0a")
        assert len(sessions) == 4  # both occurrences stay in the history
        sids = [s.session_id for s in sessions]
        assert len(set(sids)) == 4
        # The mapping folds the twins onto the shared official id.
        scoring = ds.get_scoring_data("0a0a0a0a")
        assert sorted(scoring.internal_to_official_session.values()).count(
            "sharegpt_hay_0"
        ) == 2

    def test_namespace_carries_no_sample_semantics(self, dataset):
        ns = dataset.namespace_for("real-smoke-8", "1b1b1b1b_abs")
        assert ns.startswith("ns_")
        assert "abs" not in ns
        assert ns == dataset.namespace_for("real-smoke-8", "1b1b1b1b_abs")

    def test_retrieval_request_uses_normalized_question_date(self, dataset):
        request = dataset.build_retrieval_request("0a0a0a0a", 4096)
        assert request.question_date == "2023-05-21"
        assert request.evidence_token_budget == 4096
        assert request.query == dataset.get_question("0a0a0a0a").query

    def test_lazy_cleaning_is_consistent_across_cache_hits(self, dataset):
        first = dataset.iter_sessions("0a0a0a0a")
        second = dataset.iter_sessions("0a0a0a0a")
        assert [s.model_dump() for s in first] == [s.model_dump() for s in second]


class TestScoringView:
    def test_gold_maps_to_internal_ids(self, dataset):
        scoring = dataset.get_scoring_data("0a0a0a0a")
        assert scoring.expected_answer == "pnpm"
        assert scoring.is_abstention is False
        internal_ids = {s.session_id for s in dataset.iter_sessions("0a0a0a0a")}
        assert set(scoring.gold_source_ids) <= internal_ids
        assert scoring.internal_to_official_session[scoring.gold_source_ids[0]] == "answer_s1"
        assert scoring.official_fields["question_date_raw"] == "2023/05/21 (Sun) 14:32"

    def test_integer_answer_normalized_to_string(self):
        ds = LongMemEvalDataset.from_json(json.dumps([question("0a0a0a0a", answer=3)]))
        assert ds.get_scoring_data("0a0a0a0a").expected_answer == "3"

    def test_abstention_flag_and_gold_live_in_private_view(self, dataset):
        scoring = dataset.get_scoring_data("1b1b1b1b_abs")
        assert scoring.is_abstention is True
        assert scoring.official_fields["question_id"] == "1b1b1b1b_abs"

    def test_missing_gold_on_non_abstention_is_a_validation_error(self):
        raw = json.dumps([question("0a0a0a0a", gold=("answer_ghost",))])
        ds = LongMemEvalDataset.from_json(raw)
        with pytest.raises(ContractError) as excinfo:
            ds.get_scoring_data("0a0a0a0a")
        assert excinfo.value.code == "gold_source_unknown"

    def test_missing_gold_on_abstention_is_tolerated(self):
        raw = json.dumps(
            [question("1b1b1b1b_abs", answer="no info", gold=("answer_ghost",))]
        )
        ds = LongMemEvalDataset.from_json(raw)
        scoring = ds.get_scoring_data("1b1b1b1b_abs")
        assert scoring.gold_source_ids == []

    def test_empty_gold_on_non_abstention_rejected_by_contract(self):
        raw = json.dumps([question("0a0a0a0a", gold=())])
        ds = LongMemEvalDataset.from_json(raw)
        with pytest.raises(ContractError) as excinfo:
            ds.get_scoring_data("0a0a0a0a")
        # ScoringData rejects empty gold on non-abstention samples.
        assert excinfo.value.code == "dataset_validation"


class TestUpstreamShapeValidation:
    def _clean_error(self, entry: dict):
        ds = LongMemEvalDataset.from_json(json.dumps([entry]))
        with pytest.raises(ContractError) as excinfo:
            ds.get_scoring_data(entry["question_id"])
        return excinfo.value

    def test_misaligned_parallel_arrays(self):
        entry = question("0a0a0a0a")
        entry["haystack_dates"].append("2023/06/01 (Thu) 01:00")
        assert self._clean_error(entry).code == "dataset_validation"

    def test_unknown_role(self):
        entry = question("0a0a0a0a")
        entry["haystack_sessions"][0][0]["role"] = "robot"
        assert self._clean_error(entry).code == "dataset_validation"

    def test_unknown_question_type(self):
        entry = question("0a0a0a0a", question_type="trivia")
        assert self._clean_error(entry).code == "dataset_validation"

    def test_bad_question_date(self):
        entry = question("0a0a0a0a")
        entry["question_date"] = "May 21 2023"
        assert self._clean_error(entry).code == "dataset_validation"

    def test_empty_haystack(self):
        entry = question("0a0a0a0a")
        entry["haystack_session_ids"] = []
        entry["haystack_dates"] = []
        entry["haystack_sessions"] = []
        assert self._clean_error(entry).code == "dataset_validation"

    def test_empty_session_turns(self):
        entry = question("0a0a0a0a")
        entry["haystack_sessions"][2] = []
        assert self._clean_error(entry).code == "dataset_validation"


class TestSplitMetadata:
    def test_iter_split_items_flags_abstention_and_types(self, dataset):
        items = {i.handle: i for i in dataset.iter_split_items()}
        assert items["0a0a0a0a"].question_type == "single-session-user"
        assert items["0a0a0a0a"].is_abstention is False
        assert items["1b1b1b1b_abs"].is_abstention is True

    def test_split_item_rejects_unknown_type(self):
        with pytest.raises(Exception):
            SplitItem(handle="x", question_type="trivia", is_abstention=False)


class TestDatasetSummaryAndPins:
    def test_validate_dataset_summarizes_structure(self, dataset):
        summary = validate_dataset(dataset)
        assert summary.total_questions == 2
        assert summary.abstention_questions == 1
        assert summary.observed_turn_fields == ("content", "has_answer", "role")
        assert summary.observed_sample_fields == EXPECTED_SAMPLE_FIELDS
        assert summary.turns_with_has_answer == 4  # 2 marked turns per fixture question

    def test_pin_check_reports_every_mismatch(self, dataset):
        summary = validate_dataset(dataset)
        with pytest.raises(ContractError) as excinfo:
            check_pinned_expectations(summary)
        assert excinfo.value.code == "dataset_pin_mismatch"
        msg = excinfo.value.message
        assert "total questions 2 != pinned 500" in msg
        assert "abstention questions 1 != pinned 30" in msg

    def test_pin_check_rejects_unknown_turn_fields(self, dataset):
        from eval.datasets.longmemeval import DatasetSummary

        ok = validate_dataset(dataset)
        drifted = ok.model_copy(
            update={
                "total_questions": EXPECTED_TOTAL_QUESTIONS,
                "abstention_questions": EXPECTED_ABSTENTION_QUESTIONS,
                "observed_turn_fields": ("content", "role", "tool_payload"),
            }
        )
        with pytest.raises(ContractError, match="tool_payload"):
            check_pinned_expectations(drifted)

    def test_committed_pin_record_is_well_formed(self):
        import datetime as dt
        import re

        assert re.fullmatch(r"[0-9a-f]{64}", FILE_SHA256)
        assert re.fullmatch(r"[0-9a-f]{40}", SOURCE_REVISION)
        assert SOURCE_REVISION in SOURCE_FILE_URL
        assert DATASET_LICENSE == "MIT"
        dt.date.fromisoformat(VERIFIED_ON)
        assert EXPECTED_TOTAL_QUESTIONS == 500
        assert EXPECTED_ABSTENTION_QUESTIONS == 30


class TestDatasetRegistry:
    def _config(self, dataset_plan: str, sample_ids: tuple[str, ...] = ("0a0a0a0a",)):
        from eval.config import load_config_toml

        base = load_config_toml(
            "eval/configs/examples/offline_fake.toml"
        ).model_dump(mode="python")
        base["dataset_plan"] = dataset_plan
        base["sample_ids"] = list(sample_ids)
        base["smoke_subset_ids"] = []
        return load_config_dict(base)

    def test_longmemeval_plan_loads_override_path(self, tmp_path: Path):
        path = tmp_path / "longmemeval_s_cleaned.json"
        path.write_text(json.dumps([question("0a0a0a0a")]), encoding="utf-8")
        config = self._config("longmemeval-s-cleaned@1")
        ds = load_dataset_for_config(config, path=path)
        assert isinstance(ds, LongMemEvalDataset)
        assert ds.sample_handles == ("0a0a0a0a",)

    def test_manual_plan_still_loads_fixture(self, tmp_path: Path):
        config = self._config("manual-fixtures@1")
        ds = load_dataset_for_config(
            config, path="eval/datasets/manual/smoke_samples.json"
        )
        from eval.datasets.manual import ManualDataset

        assert isinstance(ds, ManualDataset)

    def test_unknown_plan_fails_structurally(self):
        config = self._config("mystery-dataset@1")
        with pytest.raises(ContractError) as excinfo:
            load_dataset_for_config(config)
        assert excinfo.value.code == "unknown_dataset_plan"

    def test_default_real_path_is_repo_data_dir(self):
        assert DEFAULT_LONGMEMEVAL_S_PATH.name == "longmemeval_s_cleaned.json"
        assert DEFAULT_LONGMEMEVAL_S_PATH.parent.name == "longmemeval"
        assert DEFAULT_LONGMEMEVAL_S_PATH.parent.parent.name == "data"
