"""Context precheck: the fixed per-question context budget (M2).

Before any model call, the harness checks that the FULL reader input
fits the configured context window (docs/design/eval-harness.md, Token
预算与模型配置):

    total = public prompt + question
          + evidence tokens (worst case: the retrieval budget for
            budget-bound baselines; the actual full-history render for
            the full-history control)
          + message format overhead (flat configured estimate)
          + fixed output reserve (configured, NOT the model's max output)

A question that does not fit is marked ``context_exceeded``: it never
reaches the reader (no silent truncation for full history, no wasted
cost), contributes zero to planned-question scores and is counted
separately — this is what gives runnable_coverage a real denominator.

``context_window_tokens is None`` (offline M1 configs) means no limit
is declared; the precheck is skipped and runnable_coverage stays 1.0.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContextCheck:
    """Outcome of one precheck; components carry the breakdown."""

    fits: bool
    limit: int
    components: dict[str, int]
    total: int
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {
            "fits": self.fits,
            "limit": self.limit,
            "components": dict(self.components),
            "total": self.total,
            "detail": self.detail,
        }


def check_context(
    *,
    prompt_question_tokens: int,
    evidence_tokens: int,
    format_overhead_tokens: int,
    output_reserve_tokens: int,
    context_window_tokens: int,
) -> ContextCheck:
    """Compute the fixed context budget and whether the input fits."""
    components = {
        "public_prompt_and_question": prompt_question_tokens,
        "evidence": evidence_tokens,
        "format_overhead": format_overhead_tokens,
        "output_reserve": output_reserve_tokens,
    }
    total = sum(components.values())
    fits = total <= context_window_tokens
    detail = (
        " + ".join(f"{name}={value}" for name, value in components.items())
        + f" = {total} tokens vs context window {context_window_tokens}"
        + (" (fits)" if fits else " (EXCEEDED)")
    )
    return ContextCheck(
        fits=fits,
        limit=context_window_tokens,
        components=components,
        total=total,
        detail=detail,
    )


def full_history_allowance(
    *,
    prompt_question_tokens: int,
    format_overhead_tokens: int,
    output_reserve_tokens: int,
    context_window_tokens: int,
) -> int:
    """Evidence budget for the full-history control: everything the
    context window can still hold after prompt, overhead and reserve.

    This is NOT the 4K retrieval budget: the full-history control is
    reported separately and is never an equal-budget comparison. A
    non-positive allowance cannot fit any evidence.
    """
    return (
        context_window_tokens
        - prompt_question_tokens
        - format_overhead_tokens
        - output_reserve_tokens
    )
