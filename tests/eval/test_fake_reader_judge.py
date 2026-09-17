"""Fake reader/judge components: modes, usage and input isolation."""

from __future__ import annotations

import pytest

from eval.contracts.adapter import Evidence, SourceSpan
from eval.datasets.manual import ManualDataset
from eval.judges.base import JudgeProtocolError
from eval.judges.fake import (
    FakeJudge,
    FakeJudgeSpec,
    build_fake_judge,
    parse_verdict,
    render_prompt,
)
from eval.prepare.evidence import build_history_index, prepare_evidence
from eval.readers.base import ReaderError
from eval.readers.fake import (
    ABSTENTION_HYPOTHESIS,
    FakeReader,
    FakeReaderSpec,
    build_fake_reader,
)


def prepared_for(handle: str, budget: int = 4096):
    dataset = ManualDataset.load_default()
    sessions = dataset.iter_sessions(handle)
    history = build_history_index(sessions)
    evidence = []
    for session in sessions:
        for message in session.messages:
            evidence.append(
                Evidence(
                    kind="extractive",
                    text=message.content,
                    extractive_span=SourceSpan(
                        session_id=session.session_id,
                        msg_id=message.msg_id,
                        start=0,
                        end=len(message.content),
                    ),
                    derivation_sources=[],
                    source_times=[session.occurred_at],
                    retrieval_score=None,
                )
            )
    return dataset, prepare_evidence(evidence, history, budget=budget)


class TestFakeReader:
    def test_evidence_echo_uses_first_retained_unit(self):
        dataset, prepared = prepared_for("smoke_single_session_user_0001")
        reader = FakeReader(FakeReaderSpec())
        question = dataset.get_question("smoke_single_session_user_0001")
        result = reader.answer(question, prepared)
        assert result.hypothesis == prepared.items[0].evidence.text
        assert result.raw_output == result.hypothesis
        assert result.model == "fake-reader"
        assert result.usage.llm_call_count == 1
        assert result.usage.input_tokens == len(question.query) + len(
            prepared.rendered_text
        )
        assert result.usage.output_tokens == len(result.hypothesis)

    def test_echo_without_evidence_abstains(self):
        dataset = ManualDataset.load_default()
        handle = "smoke_abstention_0001"
        sessions = dataset.iter_sessions(handle)
        prepared = prepare_evidence(
            [], build_history_index(sessions), budget=64
        )
        reader = FakeReader(FakeReaderSpec())
        result = reader.answer(dataset.get_question(handle), prepared)
        assert result.hypothesis == ABSTENTION_HYPOTHESIS

    def test_fixed_mode_and_fail_mode(self):
        dataset, prepared = prepared_for("smoke_single_session_user_0001")
        question = dataset.get_question("smoke_single_session_user_0001")
        fixed = FakeReader(
            FakeReaderSpec(mode="fixed", fixed_hypothesis="pnpm")
        )
        assert fixed.answer(question, prepared).hypothesis == "pnpm"
        failing = FakeReader(FakeReaderSpec(mode="fail"))
        with pytest.raises(ReaderError) as excinfo:
            failing.answer(question, prepared)
        assert excinfo.value.code == "reader_backend_failed"
        assert excinfo.value.transient is True

    def test_fixed_mode_requires_hypothesis(self):
        with pytest.raises(ValueError):
            FakeReaderSpec(mode="fixed", fixed_hypothesis="")

    def test_build_from_reader_plan_uses_extra(self):
        class Plan:
            model = "fake-reader-x"
            extra = {"mode": "fixed", "fixed_hypothesis": "ok"}

        reader = build_fake_reader(Plan())
        assert reader.spec.mode == "fixed"
        assert reader.spec.model == "fake-reader-x"

    def test_reader_journal_never_sees_private_scoring_data(self):
        dataset, prepared = prepared_for("smoke_single_session_user_0001")
        reader = FakeReader(FakeReaderSpec())
        reader.answer(dataset.get_question("smoke_single_session_user_0001"), prepared)
        import json

        blob = json.dumps(reader.journal, ensure_ascii=False)
        for marker in (
            "session_12_answer_7ab9",  # official (gold) session id
            "session_11_answer_4d02",
            "has_answer",
            "answer_session_ids",
            "_abs",
            "I don't know the answer.",  # abstention gold answers
        ):
            assert marker not in blob, marker
        # The dataset question_date is the shared query context.
        assert reader.journal[0]["question_date"] == "2026-09-06"


class TestFakeJudge:
    def test_substring_rule_and_usage(self):
        from eval.contracts.internal import JudgeRequest

        judge = FakeJudge(FakeJudgeSpec())
        request = JudgeRequest(
            question="现在用什么包管理器？",
            expected_answer="pnpm",
            hypothesis="迁移到 pnpm，以后都用 pnpm。",
            question_type="single-session-user",
            protocol_id="longmemeval-yes-no@1",
            protocol_fields={},
        )
        result = judge.evaluate(request)
        assert result.correct is True
        assert result.raw_output == "yes"
        assert result.model == "fake-judge"
        assert result.usage.llm_call_count == 1
        assert result.usage.input_tokens == len(render_prompt(request))
        assert result.usage.output_tokens == len("yes")

    def test_substring_rule_rejects_missing_answer(self):
        from eval.contracts.internal import JudgeRequest

        judge = FakeJudge(FakeJudgeSpec())
        result = judge.evaluate(
            JudgeRequest(
                question="q",
                expected_answer="pnpm",
                hypothesis="项目使用 npm。",
                question_type="single-session-user",
                protocol_id="p@1",
                protocol_fields={},
            )
        )
        assert result.correct is False
        assert result.raw_output == "no"

    @pytest.mark.parametrize(
        "rule,expected", [("always_correct", True), ("always_wrong", False)]
    )
    def test_injected_rules(self, rule, expected):
        from eval.contracts.internal import JudgeRequest

        judge = FakeJudge(FakeJudgeSpec(verdict_rule=rule))
        result = judge.evaluate(
            JudgeRequest(
                question="q",
                expected_answer="x",
                hypothesis="完全无关",
                question_type="multi-session",
                protocol_id="p@1",
                protocol_fields={},
            )
        )
        assert result.correct is expected

    def test_per_question_overrides_win(self):
        from eval.contracts.internal import JudgeRequest

        judge = FakeJudge(
            FakeJudgeSpec(
                verdict_overrides={"这个项目现在使用什么包管理器？": True}
            )
        )
        result = judge.evaluate(
            JudgeRequest(
                question="这个项目现在使用什么包管理器？",
                expected_answer="pnpm",
                hypothesis="项目使用 npm。",
                question_type="single-session-user",
                protocol_id="p@1",
                protocol_fields={},
            )
        )
        assert result.correct is True

    def test_unparseable_output_raises_protocol_error(self):
        from eval.contracts.internal import JudgeRequest

        judge = FakeJudge(FakeJudgeSpec(verdict_rule="unparseable"))
        with pytest.raises(JudgeProtocolError) as excinfo:
            judge.evaluate(
                JudgeRequest(
                    question="q",
                    expected_answer="x",
                    hypothesis="h",
                    question_type="multi-session",
                    protocol_id="p@1",
                    protocol_fields={},
                )
            )
        assert excinfo.value.code == "judge_output_unparseable"

    def test_parse_verdict_strict(self):
        assert parse_verdict("yes") is True
        assert parse_verdict(" no ") is False
        with pytest.raises(JudgeProtocolError):
            parse_verdict("可能是对的")
        with pytest.raises(JudgeProtocolError):
            parse_verdict("YES")  # case-sensitive by protocol

    def test_prompt_carries_protocol_and_private_fields_only(self):
        from eval.contracts.internal import JudgeRequest

        request = JudgeRequest(
            question="q",
            expected_answer="a",
            hypothesis="h",
            question_type="temporal-reasoning",
            protocol_id="longmemeval-yes-no@1",
            protocol_fields={},
        )
        prompt = render_prompt(request)
        assert "longmemeval-yes-no@1" in prompt
        assert "Question: q" in prompt
        assert "Answer: a" in prompt
        assert "Hypothesis: h" in prompt

    def test_build_from_judge_plan_uses_extra(self):
        class Plan:
            model = "fake-judge-y"
            base_url = "offline://fake"
            temperature = 0.0
            protocol_id = "p@1"
            protocol_source_commit = "0" * 40
            extra = {"verdict_rule": "always_correct"}

        judge = build_fake_judge(Plan())
        assert judge.spec.verdict_rule == "always_correct"
        assert judge.spec.model == "fake-judge-y"
