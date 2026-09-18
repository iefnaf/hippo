"""readers package: fixed answering components."""

from eval.readers.base import Reader, ReaderError
from eval.readers.fake import (
    ABSTENTION_HYPOTHESIS,
    FAKE_READER_NAME,
    FakeReader,
    FakeReaderSpec,
    build_fake_reader,
)


def build_reader_for_plan(plan, **kwargs):
    """Build the reader component a reader plan's api field declares.

    'offline_fake' -> the deterministic fake (M1 configs);
    'openai_chat' -> the real OpenAI-compatible client. Extra kwargs
    (e.g. a test transport) forward to the real client only.
    """
    api = getattr(plan, "api", "offline_fake")
    if api == "openai_chat":
        from eval.readers.openai_reader import build_openai_reader

        return build_openai_reader(plan, **kwargs)
    if api == "offline_fake":
        return build_fake_reader(plan)
    raise ValueError(f"unknown reader api {api!r}")


__all__ = [
    "ABSTENTION_HYPOTHESIS",
    "FAKE_READER_NAME",
    "FakeReader",
    "FakeReaderSpec",
    "Reader",
    "ReaderError",
    "build_fake_reader",
    "build_reader_for_plan",
]
