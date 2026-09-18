"""Stratified 50/450 split and smoke selection (issue #7).

The synthetic item multiset mirrors the real pinned distribution
(470 answers across six question types, 30 abstention in four of
them), so the allocation/guarantee paths run offline exactly as they
will on the real data.
"""

from __future__ import annotations

import pytest

from eval.contracts.common import ContractError
from eval.datasets.longmemeval import SplitItem
from eval.datasets.split import (
    SMOKE_SEED,
    SPLIT_ALGORITHM,
    SPLIT_SEED,
    SmokePlan,
    SplitPlan,
    StratumCount,
    allocate_dev,
    hash_rank,
    select_smoke,
    stratified_split,
    stratum_key,
)

#: Real pinned distribution (see scripts/fetch_longmemeval.py output).
REAL_DISTRIBUTION = {
    ("single-session-user", False): 64,
    ("single-session-assistant", False): 56,
    ("single-session-preference", False): 30,
    ("temporal-reasoning", False): 127,
    ("knowledge-update", False): 72,
    ("multi-session", False): 121,
    ("single-session-user", True): 6,
    ("temporal-reasoning", True): 6,
    ("knowledge-update", True): 6,
    ("multi-session", True): 12,
}


def real_items() -> list[SplitItem]:
    items: list[SplitItem] = []
    for i, ((qtype, is_abs), count) in enumerate(sorted(REAL_DISTRIBUTION.items())):
        for j in range(count):
            handle = f"{qtype[:4]}{i:02d}{j:04d}" + ("_abs" if is_abs else "")
            items.append(SplitItem(handle=handle, question_type=qtype, is_abstention=is_abs))
    return items


def build_split(items=None, *, dev_size=50, seed=SPLIT_SEED) -> SplitPlan:
    return stratified_split(
        items if items is not None else real_items(),
        dev_size=dev_size,
        seed=seed,
        dataset_plan="longmemeval-s-cleaned@1",
        source_sha256="a" * 64,
    )


class TestHashRank:
    def test_deterministic_and_handle_sensitive(self):
        assert hash_rank("s", "scope", "h1") == hash_rank("s", "scope", "h1")
        assert hash_rank("s", "scope", "h1") != hash_rank("s", "scope", "h2")
        assert hash_rank("s", "scope", "h1") != hash_rank("s", "other", "h1")


class TestStratumKey:
    def test_encodes_type_and_abstention(self):
        a = SplitItem(handle="x", question_type="multi-session", is_abstention=False)
        b = SplitItem(handle="y", question_type="multi-session", is_abstention=True)
        assert stratum_key(a) == "multi-session|ans"
        assert stratum_key(b) == "multi-session|abs"


class TestAllocateDev:
    def test_sums_to_dev_size_and_keeps_bounds(self):
        totals = {k: len([i for i in real_items() if stratum_key(i) == k]) for k in
                  {stratum_key(i) for i in real_items()}}
        alloc = allocate_dev(totals, 50)
        assert sum(alloc.values()) == 50
        for k, d in alloc.items():
            assert 0 <= d <= totals[k]
            if totals[k] >= 2:
                assert 1 <= d <= totals[k] - 1

    def test_both_sides_guarantee_for_small_strata(self):
        totals = {"a|ans": 9, "b|ans": 9, "c|abs": 2}
        alloc = allocate_dev(totals, 3)
        assert sum(alloc.values()) == 3
        assert alloc["c|abs"] == 1  # both sides get the abstention stratum

    def test_truly_infeasible_guarantee_fails(self):
        # 2 dev slots cannot put three >=2-member strata on both sides.
        with pytest.raises(ContractError) as excinfo:
            allocate_dev({"a|ans": 9, "b|ans": 9, "c|abs": 2}, 2)
        assert excinfo.value.code == "split_infeasible"

    def test_singleton_stratum_lands_on_one_side(self):
        totals = {"a|ans": 10, "b|abs": 1}
        alloc = allocate_dev(totals, 1)
        assert sum(alloc.values()) == 1
        # the singleton cannot be on both sides; no crash either way
        assert alloc["b|abs"] in (0, 1)

    @pytest.mark.parametrize("bad", [0, 11])
    def test_infeasible_sizes_fail(self, bad):
        with pytest.raises(ContractError) as excinfo:
            allocate_dev({"a|ans": 10, "b|ans": 1}, bad)
        assert excinfo.value.code == "split_infeasible"


class TestSplitPlan:
    def test_invariants_on_real_shaped_items(self):
        split = build_split()
        items = real_items()
        dev, hold = set(split.dev_ids), set(split.holdout_ids)
        assert len(split.dev_ids) == 50
        assert len(split.holdout_ids) == 450
        assert not (dev & hold)  # no overlap
        assert dev | hold == {i.handle for i in items}  # full coverage
        # stratified by question_type x abstention
        per = {i.handle: i for i in items}
        dev_types = {per[h].question_type for h in dev}
        hold_types = {per[h].question_type for h in hold}
        assert dev_types == hold_types == {t for t, _ in REAL_DISTRIBUTION}
        # abstention on both sides
        assert any(per[h].is_abstention for h in dev)
        assert any(per[h].is_abstention for h in hold)
        assert split.algorithm == SPLIT_ALGORITHM
        assert split.seed == SPLIT_SEED

    def test_deterministic_and_order_independent(self):
        baseline = build_split()
        assert build_split() == baseline
        shuffled = list(reversed(real_items()))
        assert build_split(shuffled) == baseline

    def test_different_seed_changes_selection(self):
        baseline = build_split()
        other = build_split(seed="hippo-longmemeval-s-split@other")
        assert other.dev_ids != baseline.dev_ids
        # same invariants regardless
        assert len(other.dev_ids) == 50

    def test_proportional_allocation_is_sane(self):
        split = build_split()
        items = real_items()
        per = {i.handle: i for i in items}
        for key, count in split.strata.items():
            dev_share = count.dev / count.total
            # within one stratum, dev share tracks the global 10% (±1 member)
            assert abs(dev_share - 0.1) <= 1 / count.total + 1e-9, key
        assert sum(s.total for s in split.strata.values()) == 500
        del per

    def test_duplicate_handles_rejected(self):
        items = real_items()[:10] + real_items()[:10]
        with pytest.raises(ContractError) as excinfo:
            build_split(items, dev_size=5)
        assert excinfo.value.code == "duplicate_sample_handle"

    def test_plan_validators_catch_overlap_and_shortfall(self):
        good = build_split()
        base = good.model_dump()
        overlap = dict(base)
        hold = list(good.holdout_ids)
        hold[0] = good.dev_ids[0]  # overlaps dev while sizes stay valid
        overlap["holdout_ids"] = hold
        with pytest.raises(Exception, match="overlap"):
            SplitPlan.model_validate(overlap)
        shortfall = dict(base, dev_ids=list(good.dev_ids[:-1]))
        # removing a dev id breaks union coverage first (49 + 450 != 500)
        with pytest.raises(Exception, match="union must cover"):
            SplitPlan.model_validate(shortfall)
        wrong_size = dict(base, dev_size=49)
        with pytest.raises(Exception, match="dev side"):
            SplitPlan.model_validate(wrong_size)

    def test_stratum_count_requires_both_sides_when_possible(self):
        with pytest.raises(Exception, match="BOTH sides"):
            StratumCount(total=5, dev=0)
        with pytest.raises(Exception):
            StratumCount(total=5, dev=5)
        StratumCount(total=1, dev=0)  # singleton: one side only is fine
        with pytest.raises(Exception):
            StratumCount(total=4, dev=5)

    def test_abstention_must_not_be_single_sided(self):
        # Two singleton abstention strata, both in dev: abs_dev == abs_total
        # trips the both-sides rule (singletons pass StratumCount itself).
        doc = {
            "schema_version": 1,
            "dataset_plan": "p",
            "source_sha256": "a" * 64,
            "algorithm": SPLIT_ALGORITHM,
            "seed": "s",
            "dev_size": 12,
            "total_size": 84,
            "strata": {
                "a|ans": {"total": 40, "dev": 5},
                "b|ans": {"total": 42, "dev": 5},
                "x|abs": {"total": 1, "dev": 1},
                "y|abs": {"total": 1, "dev": 1},
            },
            "dev_ids": [f"d{i}" for i in range(12)],
            "holdout_ids": [f"h{i}" for i in range(72)],
        }
        with pytest.raises(Exception, match="abstention"):
            SplitPlan.model_validate(doc)

    def test_single_abstention_total_is_allowed_on_one_side(self):
        doc = {
            "schema_version": 1,
            "dataset_plan": "p",
            "source_sha256": "a" * 64,
            "algorithm": SPLIT_ALGORITHM,
            "seed": "s",
            "dev_size": 12,
            "total_size": 83,
            "strata": {
                "a|ans": {"total": 40, "dev": 6},
                "b|ans": {"total": 42, "dev": 6},
                "x|abs": {"total": 1, "dev": 0},
            },
            "dev_ids": [f"d{i}" for i in range(12)],
            "holdout_ids": [f"h{i}" for i in range(71)],
        }
        SplitPlan.model_validate(doc)  # one abstention total: one side only

    def test_json_round_trip_preserves_plan(self):
        split = build_split()
        loaded = SplitPlan.load_json(split.model_dump_json())
        assert loaded == split


class TestSmokePlan:
    def test_smoke_semantics_on_dev_side_only(self):
        items = real_items()
        split = build_split(items)
        smoke = select_smoke(
            items,
            seed=SMOKE_SEED,
            dataset_plan="longmemeval-s-cleaned@1",
            source_sha256="a" * 64,
            split=split,
        )
        assert len(smoke.smoke_ids) == 8
        assert set(smoke.smoke_ids) <= set(split.dev_ids)  # holdout untouched
        per = {i.handle: i for i in items}
        non_abs = [h for h in smoke.smoke_ids if not per[h].is_abstention]
        abs_ids = [h for h in smoke.smoke_ids if per[h].is_abstention]
        assert len(non_abs) == 6
        assert len(abs_ids) == 2
        assert {per[h].question_type for h in non_abs} == {
            t for t, a in REAL_DISTRIBUTION if not a
        }
        assert smoke.split_sha256

    def test_smoke_deterministic_and_order_independent(self):
        items = real_items()
        split = build_split(items)
        kwargs = dict(
            seed=SMOKE_SEED,
            dataset_plan="longmemeval-s-cleaned@1",
            source_sha256="a" * 64,
        )
        baseline = select_smoke(items, split=split, **kwargs)
        assert select_smoke(list(reversed(items)), split=split, **kwargs) == baseline

    def test_smoke_needs_two_dev_abstentions(self):
        # Only one abstention question overall -> the dev side cannot hold 2.
        items = [
            SplitItem(handle="abs1", question_type="multi-session", is_abstention=True),
            *(
                SplitItem(handle=f"a{j}", question_type="single-session-user", is_abstention=False)
                for j in range(9)
            ),
        ]
        split = stratified_split(
            items,
            dev_size=5,
            seed=SPLIT_SEED,
            dataset_plan="p",
            source_sha256="a" * 64,
        )
        with pytest.raises(ContractError) as excinfo:
            select_smoke(
                items,
                seed=SMOKE_SEED,
                dataset_plan="p",
                source_sha256="a" * 64,
                split=split,
            )
        assert excinfo.value.code == "split_infeasible"

    def test_smoke_needs_every_question_type_on_dev(self):
        # A dataset missing one type cannot provide the six-type subset.
        items = [
            SplitItem(handle="abs1", question_type="multi-session", is_abstention=True),
            SplitItem(handle="abs2", question_type="multi-session", is_abstention=True),
            *(
                SplitItem(handle=f"a{j}", question_type="single-session-user", is_abstention=False)
                for j in range(18)
            ),
        ]
        split = stratified_split(
            items,
            dev_size=5,
            seed=SPLIT_SEED,
            dataset_plan="p",
            source_sha256="a" * 64,
        )
        with pytest.raises(ContractError) as excinfo:
            select_smoke(
                items,
                seed=SMOKE_SEED,
                dataset_plan="p",
                source_sha256="a" * 64,
                split=split,
            )
        assert excinfo.value.code == "split_infeasible"

    def test_smoke_plan_validators(self):
        items = real_items()
        split = build_split(items)
        smoke = select_smoke(
            items,
            seed=SMOKE_SEED,
            dataset_plan="p",
            source_sha256="a" * 64,
            split=split,
        )
        base = smoke.model_dump()
        with pytest.raises(Exception, match="exactly 8"):
            SmokePlan.model_validate(dict(base, smoke_ids=list(smoke.smoke_ids[:7])))
        with pytest.raises(Exception, match="two abstention"):
            SmokePlan.model_validate(
                dict(base, abstention_ids=list(smoke.abstention_ids[:1]))
            )
