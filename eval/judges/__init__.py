"""judges package: verdict components and protocol adapters."""

from eval.judges.base import Judge, JudgeProtocolError
from eval.judges.fake import (
    FAKE_JUDGE_NAME,
    FakeJudge,
    FakeJudgeSpec,
    build_fake_judge,
    parse_verdict,
    render_prompt,
)

__all__ = [
    "FAKE_JUDGE_NAME",
    "FakeJudge",
    "FakeJudgeSpec",
    "Judge",
    "JudgeProtocolError",
    "build_fake_judge",
    "parse_verdict",
    "render_prompt",
]
