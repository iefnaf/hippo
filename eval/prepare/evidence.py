"""Reader input preparation: validation, fixed rendering, hard budget.

Implements the "Reader 输入准备" step of the harness. It keeps the
adapter's return order, verifies extractive evidence against the cleaned
history only, renders every unit with a harness-fixed metadata format
(adapter time strings never enter reader context as free text) and
enforces the token budget as a hard cap in return order:

- a unit that fits is kept whole;
- a unit whose text can be shortened to fit is truncated (extractive
  spans are shrunk together with the text so retained evidence and
  source ranges stay consistent);
- a unit whose metadata plus any non-empty text cannot fit is dropped
  entirely and only its raw_index is recorded;
- the final rendered text is re-counted and must not exceed the budget.

Illegal adapter output fails preparation with a structured PrepareError:
unknown source, invalid range, text mismatch or inconsistent extractive
source times. The harness never patches missing content from history and
never downgrades an extractive unit to generated. Raw (pre-truncation)
returns are persisted by the runner as a separate diagnostic artifact.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from eval.contracts.adapter import Evidence, Session, SourceSpan
from eval.contracts.common import ContractError
from eval.contracts.internal import PreparedEvidence, PreparedItem
from eval.prepare.tokens import TEST_TOKENIZER_ID, TestCharTokenizer, TokenCounter

__all__ = [
    "PrepareError",
    "HistoryIndex",
    "build_history_index",
    "prepare_evidence",
]


class PrepareError(ContractError):
    """Preparation failed on illegal adapter output (stage-level failure)."""


class HistoryIndex:
    """Lookup of cleaned messages: (session_id, msg_id) -> (content, time)."""

    def __init__(self, sessions: Iterable[Session]) -> None:
        self._contents: dict[tuple[str, str], str] = {}
        self._times: dict[tuple[str, str], str] = {}
        for session in sessions:
            for message in session.messages:
                key = (session.session_id, message.msg_id)
                if key in self._contents:
                    raise PrepareError(
                        code="duplicate_source",
                        message=(
                            f"cleaned history contains duplicate message key "
                            f"{key}; cannot verify spans against it"
                        ),
                        location=f"/history/{session.session_id}/{message.msg_id}",
                    )
                self._contents[key] = message.content
                self._times[key] = session.occurred_at

    def lookup(self, session_id: str, msg_id: str) -> tuple[str, str] | None:
        key = (session_id, msg_id)
        if key not in self._contents:
            return None
        return self._contents[key], self._times[key]

    def message_count(self) -> int:
        return len(self._contents)


def build_history_index(sessions: Iterable[Session]) -> HistoryIndex:
    return HistoryIndex(sessions)


def _extractive_prefix(position: int, span: SourceSpan, session_time: str) -> str:
    return (
        f"[{position}] 原文证据（来源会话 {span.session_id}，"
        f"消息 {span.msg_id}，时间 {session_time}）："
    )


def _generated_prefix(position: int, source_times: Sequence[str]) -> str:
    rendered = "、".join(source_times) if source_times else "未知"
    return f"[{position}] 生成内容（来源时间 {rendered}）："

_UNIT_SUFFIX = "\n"


def _largest_fit(
    tokenizer: TokenCounter,
    prefix: str,
    suffix: str,
    text: str,
    max_chars: int,
    available: int,
) -> int:
    """Largest c in [0, max_chars] with count(prefix + text[:c] + suffix) <= available.

    Monotone counters (the M1 test tokenizer is exact) make the binary
    search sound; for real tokenizers M2 re-verifies on the full render.
    """
    if available <= 0:
        return 0

    def fits(c: int) -> bool:
        return tokenizer.count(prefix + text[:c] + suffix) <= available

    if not fits(0):
        # Even metadata + empty text overflows: the unit cannot keep any
        # non-empty text and is dropped entirely.
        return 0
    lo, hi = 0, max_chars
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fits(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def prepare_evidence(
    raw: Sequence[Evidence],
    history: HistoryIndex,
    *,
    budget: int,
    tokenizer: TokenCounter | None = None,
    tokenizer_id: str = TEST_TOKENIZER_ID,
    counting_mode: str = "test",
) -> PreparedEvidence:
    """Validate, render and budget one retrieve() return, in return order."""
    if budget <= 0:
        raise ValueError("budget must be a positive token amount")
    tokenizer = tokenizer or TestCharTokenizer()

    items: list[PreparedItem] = []
    dropped: list[int] = []
    fragments: list[str] = []
    consumed = 0
    text_tokens = 0

    for raw_index, ev in enumerate(raw):
        position = raw_index + 1
        available = budget - consumed

        if ev.kind == "extractive":
            span = ev.extractive_span
            if span is None:  # pragma: no cover - prevented by the model
                raise PrepareError(
                    code="invalid_range",
                    message="extractive evidence without a span",
                    location=f"/evidence/{raw_index}",
                )
            looked_up = history.lookup(span.session_id, span.msg_id)
            if looked_up is None:
                raise PrepareError(
                    code="unknown_source",
                    message=(
                        f"extractive span references unknown cleaned source "
                        f"({span.session_id!r}, {span.msg_id!r}); the harness "
                        "does not patch or invent content"
                    ),
                    location=f"/evidence/{raw_index}/extractive_span",
                )
            content, session_time = looked_up
            if not (0 <= span.start < span.end <= len(content)):
                raise PrepareError(
                    code="invalid_range",
                    message=(
                        f"span [{span.start}, {span.end}) is not a valid "
                        f"range within the cleaned message of length "
                        f"{len(content)}"
                    ),
                    location=f"/evidence/{raw_index}/extractive_span",
                )
            if ev.text != content[span.start : span.end]:
                raise PrepareError(
                    code="text_mismatch",
                    message=(
                        f"evidence text does not equal cleaned content"
                        f"[{span.start}:{span.end}]; the harness does not "
                        "patch text and does not downgrade to generated"
                    ),
                    location=f"/evidence/{raw_index}/text",
                )
            if ev.source_times != [session_time]:
                raise PrepareError(
                    code="source_time_inconsistent",
                    message=(
                        f"extractive source_times {ev.source_times!r} must be "
                        f"exactly the session time [{session_time!r}] of the "
                        "cleaned record"
                    ),
                    location=f"/evidence/{raw_index}/source_times",
                )

            prefix = _extractive_prefix(position, span, session_time)
            max_chars = span.end - span.start
            keep = _largest_fit(
                tokenizer, prefix, _UNIT_SUFFIX, ev.text, max_chars, available
            )
            if keep <= 0:
                dropped.append(raw_index)
                continue
            truncated = keep < max_chars
            new_text = ev.text[:keep]
            new_span = SourceSpan(
                session_id=span.session_id,
                msg_id=span.msg_id,
                start=span.start,
                end=span.start + keep,
            )
            new_ev = Evidence(
                kind="extractive",
                text=new_text,
                extractive_span=new_span,
                derivation_sources=[],
                source_times=[session_time],
                retrieval_score=ev.retrieval_score,
            )
            fragment = prefix + new_text + _UNIT_SUFFIX
            verified = new_span
        else:
            prefix = _generated_prefix(position, ev.source_times)
            max_chars = len(ev.text)
            keep = _largest_fit(
                tokenizer, prefix, _UNIT_SUFFIX, ev.text, max_chars, available
            )
            if keep <= 0:
                dropped.append(raw_index)
                continue
            truncated = keep < max_chars
            new_text = ev.text[:keep]
            new_ev = Evidence(
                kind="generated",
                text=new_text,
                extractive_span=None,
                derivation_sources=list(ev.derivation_sources),
                source_times=list(ev.source_times),
                retrieval_score=ev.retrieval_score,
            )
            fragment = prefix + new_text + _UNIT_SUFFIX
            verified = None

        fragment_tokens = tokenizer.count(fragment)
        consumed += fragment_tokens
        fragments.append(fragment)
        text_tokens += tokenizer.count(new_ev.text)
        items.append(
            PreparedItem(
                raw_index=raw_index,
                evidence=new_ev,
                verified_span=verified,
                rendered_text=fragment,
                token_count=fragment_tokens,
                retained_chars=keep,
                truncated=truncated,
            )
        )

    rendered_text = "".join(fragments)
    total = tokenizer.count(rendered_text)
    if total != consumed or total > budget:
        raise RuntimeError(
            f"budget accounting broke: counted {total}, accumulated "
            f"{consumed}, budget {budget}"
        )

    return PreparedEvidence(
        rendered_text=rendered_text,
        items=items,
        token_count=total,
        text_token_count=text_tokens,
        budget=budget,
        counting_mode=counting_mode,  # type: ignore[arg-type]
        tokenizer_id=tokenizer_id,
        dropped_raw_indices=dropped,
    )
