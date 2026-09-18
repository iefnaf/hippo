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
from eval.judges.longmemeval import (
    OFFICIAL_PROTOCOL_ID,
    UPSTREAM_PROTOCOL_COMMIT,
    parse_official_verdict,
    render_official_prompt,
)


def build_judge_for_plan(plan, **kwargs):
    """Build the judge component a judge plan's api field declares.

    'offline_fake' -> the deterministic fake (M1 configs);
    'openai_chat' -> the real client bound to the official protocol.
    """
    api = getattr(plan, "api", "offline_fake")
    if api == "openai_chat":
        from eval.judges.openai_judge import build_openai_judge

        return build_openai_judge(plan, **kwargs)
    if api == "offline_fake":
        return build_fake_judge(plan)
    raise ValueError(f"unknown judge api {api!r}")


__all__ = [
    "FAKE_JUDGE_NAME",
    "FakeJudge",
    "FakeJudgeSpec",
    "Judge",
    "JudgeProtocolError",
    "OFFICIAL_PROTOCOL_ID",
    "UPSTREAM_PROTOCOL_COMMIT",
    "build_fake_judge",
    "build_judge_for_plan",
    "parse_official_verdict",
    "parse_verdict",
    "render_official_prompt",
    "render_prompt",
]
