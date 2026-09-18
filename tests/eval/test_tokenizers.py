"""Offline tokenizer counting modes (issue #8: 精确计数与估算分离).

The DeepSeek offline tokenizer tests verify the pinned file when it is
present and SKIP with an explicit message when it is not (a missing
optional data file is a skip, never a silent pass); the heuristic and
test counters are always exercised offline.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from eval.prepare.tokens import (
    DEEPSEEK_TOKENIZER_ID,
    DEEPSEEK_TOKENIZER_SHA256,
    DEFAULT_DEEPSEEK_TOKENIZER_PATH,
    ESTIMATED_TOKENIZER_ID,
    TEST_TOKENIZER_ID,
    DeepSeekOfflineTokenizer,
    HeuristicEstimatingTokenizer,
    TestCharTokenizer,
    TokenizerUnavailable,
    build_token_counter,
)


class TestCountingModes:
    def test_test_counter_is_chars(self):
        tok = TestCharTokenizer()
        assert tok.tokenizer_id == TEST_TOKENIZER_ID
        assert tok.count("abc") == 3
        assert tok.count("") == 0

    def test_estimated_counter_is_documented_heuristic(self):
        tok = HeuristicEstimatingTokenizer()
        assert tok.tokenizer_id == ESTIMATED_TOKENIZER_ID
        assert tok.count("") == 0
        assert tok.count("abcd") == 1
        assert tok.count("abcde") == 2
        assert tok.count("项目使用 pnpm") == math.ceil(len("项目使用 pnpm") / 4)

    def test_estimated_counter_is_monotone(self):
        # prepare_evidence binary-searches on counter monotonicity.
        tok = HeuristicEstimatingTokenizer()
        text = "x" * 500
        assert [tok.count(text[:n]) for n in range(0, 40)] == sorted(
            tok.count(text[:n]) for n in range(0, 40)
        )

    def test_builder_selects_by_mode_and_rejects_id_mismatch(self):
        assert isinstance(build_token_counter("test"), TestCharTokenizer)
        assert (
            build_token_counter("estimated").tokenizer_id == ESTIMATED_TOKENIZER_ID
        )
        with pytest.raises(ValueError, match="tokenizer id agree"):
            build_token_counter("estimated", tokenizer_id=DEEPSEEK_TOKENIZER_ID)
        with pytest.raises(ValueError, match="unknown counting mode"):
            build_token_counter("precise-ish")

    def test_builder_exact_requires_pinned_file(self, tmp_path: Path):
        with pytest.raises(TokenizerUnavailable, match="fetch_deepseek_tokenizer"):
            build_token_counter("exact", path=tmp_path / "missing.json")

    def test_exact_rejects_tampered_file(self, tmp_path: Path):
        if not DEFAULT_DEEPSEEK_TOKENIZER_PATH.exists():
            pytest.skip(
                "pinned DeepSeek tokenizer file not fetched; run "
                "scripts/fetch_deepseek_tokenizer.py to exercise exact mode"
            )
        tampered = tmp_path / "tokenizer.json"
        tampered.write_text("{}", encoding="utf-8")
        with pytest.raises(TokenizerUnavailable, match="sha256"):
            DeepSeekOfflineTokenizer(tampered)


class TestPinnedDeepSeekTokenizer:
    """Exact counting over the pinned file; skipped until fetched."""

    @pytest.fixture()
    def tokenizer(self) -> DeepSeekOfflineTokenizer:
        if not DEFAULT_DEEPSEEK_TOKENIZER_PATH.exists():
            pytest.skip(
                "pinned DeepSeek tokenizer file not fetched; run "
                "`uv run python scripts/fetch_deepseek_tokenizer.py` "
                "(explicit skip, not a silent pass)"
            )
        return DeepSeekOfflineTokenizer()

    def test_pin_checksum_matches_on_load(self, tokenizer):
        import hashlib

        digest = hashlib.sha256(
            Path(tokenizer.path).read_bytes()
        ).hexdigest()
        assert digest == DEEPSEEK_TOKENIZER_SHA256

    def test_counts_are_stable_and_sane(self, tokenizer):
        assert tokenizer.tokenizer_id == DEEPSEEK_TOKENIZER_ID
        assert tokenizer.count("") == 0
        en = tokenizer.count("Hello world, this is a test.")
        zh = tokenizer.count("项目使用 pnpm，已迁移完成。")
        assert 1 <= en <= 40
        assert 1 <= zh <= 40
        # Monotone in prefixes (binary-search soundness for prepare)
        text = "The project moved to pnpm for CI and local development. 2026-09-03."
        counts = [tokenizer.count(text[:n]) for n in range(0, len(text) + 1, 7)]
        assert counts == sorted(counts)

    def test_builder_exact_returns_pinned_counter(self, tokenizer):
        counter = build_token_counter(
            "exact", tokenizer_id=DEEPSEEK_TOKENIZER_ID
        )
        assert counter.tokenizer_id == DEEPSEEK_TOKENIZER_ID
