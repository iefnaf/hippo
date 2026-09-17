"""readers package: fixed answering components."""

from eval.readers.base import Reader, ReaderError
from eval.readers.fake import (
    ABSTENTION_HYPOTHESIS,
    FAKE_READER_NAME,
    FakeReader,
    FakeReaderSpec,
    build_fake_reader,
)

__all__ = [
    "ABSTENTION_HYPOTHESIS",
    "FAKE_READER_NAME",
    "FakeReader",
    "FakeReaderSpec",
    "Reader",
    "ReaderError",
    "build_fake_reader",
]
