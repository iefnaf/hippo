"""Reader input preparation: validation, rendering and hard budget."""

from __future__ import annotations

import pytest

from eval.contracts.adapter import (
    Evidence,
    Message,
    RetrievalRequest,
    Session,
    SourceRef,
    SourceSpan,
)
from eval.prepare.evidence import (
    PrepareError,
    build_history_index,
    prepare_evidence,
)
from eval.prepare.tokens import TestCharTokenizer

SESSION_TIME = "2026-09-03"
CONTENT = "迁移到 pnpm，以后都用 pnpm。"


def make_history() -> object:
    session = Session(
        session_id="s_6f2c",
        occurred_at=SESSION_TIME,
        messages=[Message(msg_id="m_91ab", role="user", content=CONTENT)],
    )
    return build_history_index([session])


def extractive(text: str = CONTENT, start: int = 0, end: int | None = None) -> Evidence:
    return Evidence(
        kind="extractive",
        text=text,
        extractive_span=SourceSpan(
            session_id="s_6f2c", msg_id="m_91ab", start=start,
            end=end if end is not None else len(text),
        ),
        derivation_sources=[],
        source_times=[SESSION_TIME],
        retrieval_score=2.73,
    )


def generated(text: str = "项目目前统一使用 pnpm。") -> Evidence:
    return Evidence(
        kind="generated",
        text=text,
        extractive_span=None,
        derivation_sources=[
            SourceRef(session_id="s_6f2c", msg_id="m_91ab")
        ],
        source_times=[SESSION_TIME],
        retrieval_score=None,
    )


class TestValidation:
    def test_valid_extractive_passes_with_verified_span(self):
        prepared = prepare_evidence(
            [extractive()], make_history(), budget=4096
        )
        assert len(prepared.items) == 1
        item = prepared.items[0]
        assert item.verified_span == item.evidence.extractive_span
        assert item.truncated is False
        assert item.retained_chars == len(CONTENT)

    def test_unknown_source_fails(self):
        ev = Evidence(
            kind="extractive",
            text=CONTENT,
            extractive_span=SourceSpan(
                session_id="s_missing", msg_id="m_91ab", start=0, end=len(CONTENT)
            ),
            derivation_sources=[],
            source_times=[SESSION_TIME],
            retrieval_score=None,
        )
        with pytest.raises(PrepareError) as excinfo:
            prepare_evidence([ev], make_history(), budget=4096)
        assert excinfo.value.code == "unknown_source"
        assert "/evidence/0/extractive_span" in excinfo.value.location

    def test_range_beyond_message_fails(self):
        with pytest.raises(PrepareError) as excinfo:
            prepare_evidence(
                [extractive(start=0, end=len(CONTENT) + 10)],
                make_history(),
                budget=4096,
            )
        assert excinfo.value.code == "invalid_range"

    def test_text_mismatch_fails_without_patching(self):
        # Valid range but different text: never patched, never downgraded.
        ev = Evidence(
            kind="extractive",
            text=CONTENT + "多出来的字",
            extractive_span=SourceSpan(
                session_id="s_6f2c", msg_id="m_91ab", start=0, end=len(CONTENT)
            ),
            derivation_sources=[],
            source_times=[SESSION_TIME],
            retrieval_score=None,
        )
        with pytest.raises(PrepareError) as excinfo:
            prepare_evidence([ev], make_history(), budget=4096)
        assert excinfo.value.code == "text_mismatch"

    def test_source_time_inconsistent_fails(self):
        ev = Evidence(
            kind="extractive",
            text=CONTENT,
            extractive_span=SourceSpan(
                session_id="s_6f2c", msg_id="m_91ab", start=0, end=len(CONTENT)
            ),
            derivation_sources=[],
            source_times=["2026-09-01"],
            retrieval_score=None,
        )
        with pytest.raises(PrepareError) as excinfo:
            prepare_evidence([ev], make_history(), budget=4096)
        assert excinfo.value.code == "source_time_inconsistent"

    def test_failure_never_downgrades_to_generated(self):
        # A failing extractive unit raises; no PreparedEvidence is produced
        # and nothing is converted to generated content.
        with pytest.raises(PrepareError):
            prepare_evidence(
                [extractive(text=CONTENT[:3] + "X"), generated()],
                make_history(),
                budget=4096,
            )


class TestRendering:
    def test_fixed_format_renders_source_metadata(self):
        prepared = prepare_evidence(
            [extractive(), generated()], make_history(), budget=4096
        )
        first, second = prepared.items
        assert first.rendered_text.startswith("[1] 原文证据（来源会话 s_6f2c，消息 m_91ab，时间 2026-09-03）：")
        assert second.rendered_text.startswith("[2] 生成内容（来源时间 2026-09-03）：")
        assert prepared.rendered_text == first.rendered_text + second.rendered_text

    def test_derivation_sources_never_rendered(self):
        prepared = prepare_evidence([generated()], make_history(), budget=4096)
        assert "derivation" not in prepared.rendered_text
        assert "m_91ab" not in prepared.rendered_text  # sources stay diagnostic

    def test_generated_without_times_renders_unknown(self):
        ev = Evidence(
            kind="generated",
            text="不知道来源的摘要。",
            extractive_span=None,
            derivation_sources=[],
            source_times=[],
            retrieval_score=None,
        )
        prepared = prepare_evidence([ev], make_history(), budget=4096)
        assert "来源时间 未知" in prepared.rendered_text

    def test_empty_input_yields_empty_prepared(self):
        prepared = prepare_evidence([], make_history(), budget=4096)
        assert prepared.items == []
        assert prepared.rendered_text == ""
        assert prepared.token_count == 0
        assert prepared.dropped_raw_indices == []


class TestHardBudget:
    def test_truncation_shrinks_span_together_with_text(self):
        budget = 60  # metadata (~47 chars) leaves room for a prefix only
        prepared = prepare_evidence([extractive()], make_history(), budget=budget)
        assert prepared.token_count <= budget
        item = prepared.items[0]
        assert item.truncated is True
        assert item.evidence.text == CONTENT[: item.retained_chars]
        span = item.evidence.extractive_span
        assert span.end == span.start + item.retained_chars
        assert item.verified_span == span
        assert CONTENT[span.start : span.end] == item.evidence.text

    def test_units_dropped_when_metadata_does_not_fit(self):
        # Budget so small that even one character plus metadata overflows.
        budget = 20
        prepared = prepare_evidence(
            [extractive(), generated(), extractive()],
            make_history(),
            budget=budget,
        )
        assert prepared.dropped_raw_indices, "later units must be dropped"
        assert prepared.token_count <= budget

    def test_return_order_is_preserved_within_budget(self):
        history = make_history()
        units = [generated("摘要甲。" * 5), generated("摘要乙。" * 5)]
        prepared = prepare_evidence(units, history, budget=4096)
        assert [i.raw_index for i in prepared.items] == [0, 1]

    def test_every_raw_unit_retained_or_dropped_exactly_once(self):
        history = make_history()
        units = [generated("长摘要。" * 10), extractive(), generated("短摘要。")]
        prepared = prepare_evidence(units, history, budget=48)
        seen = sorted(
            [i.raw_index for i in prepared.items] + prepared.dropped_raw_indices
        )
        assert seen == [0, 1, 2]
        assert len(set(seen)) == len(seen)

    def test_mixed_flood_truncates_first_drops_rest(self):
        history = make_history()
        # A long generated unit followed by everything else.
        units = [generated("很长的生成内容。" * 30), extractive(), extractive()]
        prepared = prepare_evidence(units, history, budget=200)
        assert prepared.token_count <= 200
        first = prepared.items[0]
        assert first.truncated is True
        assert first.evidence.kind == "generated"  # truncation keeps the kind
        assert 1 in prepared.dropped_raw_indices or all(
            i.raw_index == 0 for i in prepared.items
        )

    def test_text_tokens_are_counted_separately(self):
        prepared = prepare_evidence(
            [extractive(), generated()], make_history(), budget=4096
        )
        tokenizer = TestCharTokenizer()
        assert prepared.text_token_count == sum(
            tokenizer.count(i.evidence.text) for i in prepared.items
        )
        assert prepared.text_token_count < prepared.token_count  # metadata overhead

    def test_zero_budget_rejected(self):
        with pytest.raises(ValueError):
            prepare_evidence([extractive()], make_history(), budget=0)
