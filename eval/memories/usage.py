"""Usage reporting for receipt-less adapter calls (retrieve/open/close).

Per the data contract, retrieve and lifecycle calls have no receipt to
carry usage, so the runner installs a contextvar-based UsageRecorder
around each call attempt; adapters report what they know through
report_usage(). No report within an attempt means the usage is unknown
(null), never zero-filled. await_ready must not re-report build usage
already counted on the submitting attempt.

merge_resource_usage sums only known values; a single unknown quantity
keeps that field unknown. Costs sum only within one currency.
"""

from __future__ import annotations

import contextlib
import decimal
from contextvars import ContextVar
from typing import Iterator

from eval.contracts.adapter import ResourceUsage

_ACTIVE_RECORDER: ContextVar["UsageRecorder | None"] = ContextVar(
    "hippo_eval_usage_recorder", default=None
)


def merge_resource_usage(
    usages: list[ResourceUsage],
) -> ResourceUsage | None:
    """Merge reported usages of one attempt; None if nothing was reported."""

    def _sum(values: list[int | None]) -> int | None:
        if any(v is None for v in values):
            return None
        return sum(v for v in values if v is not None)  # type: ignore[arg-type]

    if not usages:
        return None
    input_tokens = _sum([u.input_tokens for u in usages])
    output_tokens = _sum([u.output_tokens for u in usages])
    llm_calls = _sum([u.llm_call_count for u in usages])
    cost_amount: str | None = None
    currency: str | None = None
    if all(u.cost_amount is not None and u.currency is not None for u in usages):
        currencies = {u.currency for u in usages}
        if len(currencies) == 1:
            total = sum(
                (decimal.Decimal(u.cost_amount) for u in usages),  # type: ignore[arg-type]
                decimal.Decimal(0),
            )
            cost_amount = format(total, "f")
            currency = usages[0].currency
    return ResourceUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        llm_call_count=llm_calls,
        cost_amount=cost_amount,
        currency=currency,
    )


class UsageRecorder:
    """Collects usage reports made within one call attempt."""

    def __init__(self) -> None:
        self._reports: list[ResourceUsage] = []

    def report(self, usage: ResourceUsage) -> None:
        self._reports.append(usage)

    def merged(self) -> ResourceUsage | None:
        return merge_resource_usage(self._reports)


@contextlib.contextmanager
def recorded_usage() -> Iterator[UsageRecorder]:
    """Install a recorder for the duration of one adapter call attempt."""
    recorder = UsageRecorder()
    token = _ACTIVE_RECORDER.set(recorder)
    try:
        yield recorder
    finally:
        _ACTIVE_RECORDER.reset(token)


def report_usage(usage: ResourceUsage) -> None:
    """Report usage from inside an adapter call; no-op outside a attempt."""
    recorder = _ACTIVE_RECORDER.get()
    if recorder is not None:
        recorder.report(usage)
