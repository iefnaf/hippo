#!/usr/bin/env python3
"""Judge calibration one-key workflow (issue #9, M2).

Three steps around the HUMAN blind-annotation phase:

  plan    collect the dev-side outputs of TWO run conditions and write
          the calibration plan, the rubric and the de-identified
          worksheets (offline; deterministic, fixed seed)
  judge   batch-call the REAL judge for every planned item (LIVE;
          needs the judge credential env var, e.g. ZAI_API_KEY — fails
          fast with an explicit message when unset)
  report  import the FILLED worksheets, pair them with the judge calls,
          compute statistics + threshold decision and write the
          calibration record (offline)

Typical session (docs/design/judge-calibration-workflow.md has the full
workflow for the annotator):

    uv run python scripts/judge_calibration.py plan \
        --run-a runs/run-...-bm25 --run-b runs/run-...-none \
        --out calib
    # -> fill calib/worksheet_random.csv (blind, rubric in calib/rubric.md)
    uv run python scripts/judge_calibration.py judge --dir calib \
        --config eval/configs/examples/real_dev50_bm25.toml
    # -> after >= 1 day, fill calib/worksheet_selfconsistency.csv
    uv run python scripts/judge_calibration.py report --dir calib \
        --config eval/configs/examples/real_dev50_bm25.toml
    # -> calib/calibration.json (attach with: hippo-eval run
        --judge-calibration calib/calibration.json)

The produced record keeps the judge's four drift identifiers plus the
calibration run date; a run whose calibration did not pass gets its QA
conclusions degraded to diagnostics (see eval.calibration.integration).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _die(message: str, code: str = "calibration_error") -> None:
    print(f"error: {code}: {message}", file=sys.stderr)
    raise SystemExit(2)


def cmd_plan(args: argparse.Namespace) -> int:
    from eval.calibration.sampling import (
        build_calibration_plan,
        collect_run_outputs,
        render_worksheet_csv,
        worksheet_rows,
    )
    from eval.calibration.criteria import RUBRIC_MD
    from eval.contracts.common import now_utc

    out = Path(args.dir)
    if (out / "plan.json").exists() and not args.force:
        _die(
            f"{out / 'plan.json'} already exists (use --force to redo; the "
            "plan seed and worksheets should stay fixed once annotation "
            "started)",
            "plan_exists",
        )
    out.mkdir(parents=True, exist_ok=True)
    candidates = []
    for label, run_dir in (("A", args.run_a), ("B", args.run_b)):
        try:
            found, stats = collect_run_outputs(run_dir, label)
        except Exception as exc:  # noqa: BLE001 - surfaced as CLI error
            _die(f"collecting {run_dir} failed: {exc}", "collect_failed")
        missing = stats["without_judge_record"]
        if missing and args.strict:
            _die(
                f"run {run_dir} has {len(missing)} samples without a judge "
                f"record ({missing[:5]}…); calibration needs the full "
                "dev-side population (resume the run first)",
                "incomplete_run",
            )
        print(
            f"[{label}] {run_dir}: {len(found)} outputs from run "
            f"{stats['run_id']}"
        )
        candidates.extend(found)
    try:
        plan = build_calibration_plan(
            candidates,
            created_at=now_utc(),
            seed=args.seed,
            random_size=args.random_size,
            boundary_size=args.boundary_size,
            strict=args.strict,
        )
    except Exception as exc:  # noqa: BLE001 - surfaced as CLI error
        _die(str(exc), "plan_infeasible")
    (out / "plan.json").write_text(
        plan.model_dump_json(indent=2), encoding="utf-8"
    )
    (out / "rubric.md").write_text(RUBRIC_MD, encoding="utf-8")
    (out / "worksheet_random.csv").write_text(
        render_worksheet_csv(worksheet_rows(plan, "random")), encoding="utf-8"
    )
    (out / "worksheet_boundary.csv").write_text(
        render_worksheet_csv(worksheet_rows(plan, "boundary")), encoding="utf-8"
    )
    (out / "worksheet_selfconsistency.csv").write_text(
        render_worksheet_csv(worksheet_rows(plan, "self_consistency")),
        encoding="utf-8",
    )
    overlap = sum(1 for i in plan.items if i.overlaps_random)
    print(
        json.dumps(
            {
                "plan_id": plan.plan_id,
                "random": plan.random_size,
                "boundary": plan.boundary_size,
                "boundary_overlapping_random": overlap,
                "self_consistency_items": len(plan.self_consistency_item_ids),
                "out": str(out),
                "next": (
                    "盲标 worksheet_random.csv（与 worksheet_boundary.csv），"
                    "rubric 见 rubric.md；间隔至少一天后再标 "
                    "worksheet_selfconsistency.csv；然后运行 judge 子命令"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def cmd_judge(args: argparse.Namespace) -> int:
    import os

    from eval.calibration.judge_batch import run_judge_batch
    from eval.calibration.sampling import CalibrationPlanArtifact
    from eval.config import load_config_toml
    from eval.contracts.common import ContractError
    from eval.judges import build_judge_for_plan

    calib_dir = Path(args.dir)
    plan = CalibrationPlanArtifact.load_json(
        (calib_dir / "plan.json").read_text(encoding="utf-8")
    )
    config = load_config_toml(args.config)
    if config.judge.api != "openai_chat":
        _die(
            "the judge step needs a real judge config (api = 'openai_chat')",
            "config_invalid",
        )
    if not os.environ.get(config.judge.api_key_env):
        _die(
            f"environment variable {config.judge.api_key_env!r} is not set — "
            "the calibration judge step is LIVE and refuses to run (or "
            "pretend to run) without the credential",
            "missing_api_key",
        )
    judge = build_judge_for_plan(config.judge)
    try:
        artifact = run_judge_batch(
            plan,
            judge,
            api_key_env=config.judge.api_key_env,
            max_retries=args.max_retries,
        )
    except ContractError as exc:
        _die(exc.message, exc.code)
    (calib_dir / "judge_calls.json").write_text(
        artifact.model_dump_json(indent=2), encoding="utf-8"
    )
    parsed = sum(1 for c in artifact.calls if c.verdict is not None)
    failed = sum(1 for c in artifact.calls if c.error is not None)
    print(
        json.dumps(
            {
                "calls": len(artifact.calls),
                "parsed": parsed,
                "failed_or_parse_failed": failed,
                "observed_response_models": list(
                    artifact.observed_response_models
                ),
                "out": str(calib_dir / "judge_calls.json"),
                "next": (
                    "盲标完成后运行 report 子命令（filled worksheet 作为 "
                    "--annotations / --boundary-annotations / "
                    "--self-consistency 传入）"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if failed == 0 else 1


def _load_annotations(path: str, ids, round_name: str):
    from eval.calibration.sampling import import_annotations
    from eval.contracts.common import ContractError

    try:
        return import_annotations(
            Path(path).read_text(encoding="utf-8"),
            expected_item_ids=ids,
            round=round_name,
        )
    except ContractError as exc:
        _die(f"{path}: {exc.message}", exc.code)


def cmd_report(args: argparse.Namespace) -> int:
    from eval.calibration.judge_batch import CalibrationJudgeCallsArtifact
    from eval.calibration.record import build_calibration_record
    from eval.calibration.sampling import CalibrationPlanArtifact
    from eval.calibration.stats import compute_statistics, decide_calibration
    from eval.config import load_config_toml
    from eval.contracts.common import now_utc

    calib_dir = Path(args.dir)
    plan = CalibrationPlanArtifact.load_json(
        (calib_dir / "plan.json").read_text(encoding="utf-8")
    )
    calls = CalibrationJudgeCallsArtifact.load_json(
        (calib_dir / "judge_calls.json").read_text(encoding="utf-8")
    )
    config = load_config_toml(args.config)
    random_ids = [i.item_id for i in plan.items if i.cohort == "random"]
    boundary_ids = [i.item_id for i in plan.items if i.cohort == "boundary"]

    annotations = _load_annotations(args.annotations, random_ids, "first")
    if args.boundary_annotations:
        annotations = annotations + _load_annotations(
            args.boundary_annotations, boundary_ids, "first"
        )
    self_annotations = []
    if args.self_consistency:
        self_annotations = _load_annotations(
            args.self_consistency,
            list(plan.self_consistency_item_ids),
            "self_consistency",
        )
    statistics = compute_statistics(
        items=list(plan.items),
        annotations=annotations,
        judge_calls=calls.calls,
        self_consistency_annotations=self_annotations,
    )
    decision = decide_calibration(statistics)
    record = build_calibration_record(
        plan=plan,
        annotations=annotations,
        self_consistency_annotations=self_annotations,
        judge_calls=calls,
        statistics=statistics,
        decision=decision,
        judge_vendor_documented_version=config.judge.vendor_documented_version,
        judge_vendor_documented_on=config.judge.vendor_documented_on,
        created_at=now_utc(),
    )
    (calib_dir / "calibration.json").write_text(
        record.model_dump_json(indent=2), encoding="utf-8"
    )
    from eval.calibration.render import render_calibration_markdown

    (calib_dir / "calibration.md").write_text(
        render_calibration_markdown(record), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "record_id": record.record_id,
                "verdict": decision.verdict,
                "overall_agreement": statistics.overall.agreement,
                "overall_wilson_low": statistics.overall.wilson_low,
                "undecided_ratio": statistics.undecided_ratio,
                "self_consistency": (
                    statistics.self_consistency.agreement
                    if statistics.self_consistency
                    else None
                ),
                "cross_condition_diff": statistics.cross_condition_diff,
                "parse_failures": statistics.judge_parse_failures,
                "out": str(calib_dir / "calibration.json"),
                "next": (
                    "hippo-eval run --judge-calibration "
                    f"{calib_dir / 'calibration.json'} 附上记录；未通过"
                    "（failed/inconclusive）的运行其问答结论会降级为诊断项"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="judge_calibration.py",
        description="M2 judge calibration: plan -> (human blind annotation) "
        "-> judge batch -> report",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_plan = sub.add_parser("plan", help="build the sampling plan + worksheets")
    p_plan.add_argument("--run-a", required=True, metavar="DIR")
    p_plan.add_argument("--run-b", required=True, metavar="DIR")
    p_plan.add_argument("--dir", required=True, metavar="DIR")
    p_plan.add_argument("--seed", default=None, help="sampling seed override")
    p_plan.add_argument("--random-size", type=int, default=None)
    p_plan.add_argument("--boundary-size", type=int, default=None)
    p_plan.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    p_plan.add_argument("--force", action="store_true")
    p_plan.set_defaults(func=cmd_plan)

    p_judge = sub.add_parser(
        "judge", help="batch-call the real judge for every planned item (LIVE)"
    )
    p_judge.add_argument("--dir", required=True, metavar="DIR")
    p_judge.add_argument("--config", required=True, metavar="TOML")
    p_judge.add_argument("--max-retries", type=int, default=3)
    p_judge.set_defaults(func=cmd_judge)

    p_report = sub.add_parser(
        "report", help="pair annotations with judge calls and decide"
    )
    p_report.add_argument("--dir", required=True, metavar="DIR")
    p_report.add_argument("--config", required=True, metavar="TOML")
    p_report.add_argument(
        "--annotations",
        required=True,
        metavar="CSV",
        help="filled RANDOM-cohort worksheet",
    )
    p_report.add_argument(
        "--boundary-annotations",
        default=None,
        metavar="CSV",
        help="filled BOUNDARY worksheet (optional; diagnostics only)",
    )
    p_report.add_argument(
        "--self-consistency",
        default=None,
        metavar="CSV",
        help="filled re-annotation worksheet (>= 1 day after the first round)",
    )
    p_report.set_defaults(func=cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "plan":
        if args.seed is None:
            from eval.calibration.sampling import CALIBRATION_SEED

            args.seed = CALIBRATION_SEED
        if args.random_size is None:
            from eval.calibration.criteria import RANDOM_SAMPLE_SIZE

            args.random_size = RANDOM_SAMPLE_SIZE
        if args.boundary_size is None:
            from eval.calibration.criteria import BOUNDARY_SAMPLE_SIZE

            args.boundary_size = BOUNDARY_SAMPLE_SIZE
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())