"""Hippo eval CLI: run / resume / compare / validate.

validate is fully offline and structural. run executes the M1 offline
loop (per-session ingest -> reopen -> retrieve -> evidence preparation)
with the fake memory components declared by the config when --out is
given; without --out it prints the plan only (offline summary, nothing
executed). resume continues an interrupted or partially failed run from
its checkpoint directory: it refuses to reuse checkpoints whose config
fingerprint, space identity or artifact schema version does not match
and tells the user to start a new run. compare takes two run
directories, checks comparability on every key field except the memory
plan (the allowed experimental factor), refuses the same-condition
label and metric auto-alignment on registry mismatch, reports both
sides' full planned-set results and persists the common runnable
intersection with its ID list.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eval.config import ExperimentConfig, load_config_toml
from eval.contracts.common import ContractError
from eval.validate import ValidationReport, validate_all


def _print_contract_error(exc: ContractError) -> None:
    print(
        f"error: {exc.code} at {exc.location or '(root)'}: {exc.message}",
        file=sys.stderr,
    )
    for key, detail in exc.details.items():
        print(f"  ({key}) {detail}", file=sys.stderr)


def cmd_validate(args: argparse.Namespace) -> int:
    fixtures: list[Path] = []
    for pattern in args.fixtures:
        p = Path(pattern)
        if p.is_dir():
            fixtures.extend(sorted(p.rglob("*.json")))
        else:
            fixtures.append(p)
    report: ValidationReport = validate_all(list(args.configs), fixtures)
    print(report.render())
    return report.exit_code()


def _load_config_or_exit(path: str) -> ExperimentConfig:
    try:
        return load_config_toml(path)
    except ContractError as exc:
        _print_contract_error(exc)
        raise SystemExit(2) from exc


def _build_components(config: ExperimentConfig):
    """Build the (adapter, reader, judge) components a config declares.

    Baseline kinds (none/bm25/full_history) build their harness
    baseline adapters; api='openai_chat' plans build the real clients
    (credentials resolved from the named environment variables at call
    time); everything else stays on the offline fakes.
    """
    from eval.judges import build_judge_for_plan
    from eval.memories import build_memory_for_plan
    from eval.readers import build_reader_for_plan

    return (
        build_memory_for_plan(config.memory),
        build_reader_for_plan(config.reader),
        build_judge_for_plan(config.judge),
    )


def _execute_offline_run(config: ExperimentConfig, args: argparse.Namespace) -> int:
    from eval.contracts.common import now_utc
    from eval.memories import build_memory_for_plan
    from eval.runs import RunStore, new_run_id

    try:
        adapter = build_memory_for_plan(config.memory)
    except ContractError as exc:
        _print_contract_error(exc)
        return 2
    run_id = new_run_id(config.fingerprint(), clock=now_utc)
    store = RunStore(Path(args.out), run_id)
    if config.suite == "operations":
        from eval.operations import OperationsRunner

        runner = OperationsRunner(
            config=config, adapter=adapter, store=store, run_id=run_id
        )
        try:
            outcome = runner.run()
        except ContractError as exc:
            _print_contract_error(exc)
            return 2
        payload = {
            "command": "run",
            "config": config.name,
            "config_fingerprint": config.fingerprint(),
            "metrics_registry_version": config.canonical_payload()[
                "metrics_registry_version"
            ],
            **outcome.payload(),
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0 if outcome.failed == 0 else 1

    from eval.datasets import load_dataset_for_config
    from eval.judges import build_judge_for_plan
    from eval.readers import build_reader_for_plan
    from eval.runner import OfflineRunner

    try:
        dataset = load_dataset_for_config(config, path=args.dataset)
    except ContractError as exc:
        _print_contract_error(exc)
        return 2
    try:
        runner = OfflineRunner(
            config=config,
            dataset=dataset,
            adapter=adapter,
            reader=build_reader_for_plan(config.reader),
            judge=build_judge_for_plan(config.judge),
            store=store,
            run_id=run_id,
        )
    except ContractError as exc:
        _print_contract_error(exc)
        return 2
    try:
        outcome = runner.run()
    except ContractError as exc:
        _print_contract_error(exc)
        return 2
    from eval.report import Reporter

    report = Reporter(outcome.run_dir).build()
    headline = {
        m["metric_id"]: m["value"] if m["status"] == "computed" else None
        for m in report["metrics"]
        if m["metric_id"]
        in (
            "planned_question_score",
            "scored_accuracy",
            "verifiable_session_recall_macro",
            "verifiable_session_recall_micro",
            "recall_at_1",
        )
    }
    payload = {
        "command": "run",
        "run_id": outcome.run_id,
        "run_dir": str(outcome.run_dir),
        "config": config.name,
        "config_fingerprint": config.fingerprint(),
        "metrics_registry_version": config.canonical_payload()[
            "metrics_registry_version"
        ],
        "sample_count": len(outcome.results),
        "headline_metrics": headline,
        **outcome.summary(),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if outcome.failed == 0 and outcome.invalid_input == 0 else 1


def cmd_run(args: argparse.Namespace) -> int:
    config = _load_config_or_exit(args.config)
    if args.out:
        return _execute_offline_run(config, args)
    plan = {
        "command": "run",
        "config": config.name,
        "config_fingerprint": config.fingerprint(),
        "metrics_registry_version": config.canonical_payload()[
            "metrics_registry_version"
        ],
        "sample_count": len(config.sample_ids),
        "status": "not-executed (pass --out DIR to execute the offline M1 "
        "loop: per-session ingest -> reopen -> retrieve -> prepare -> read "
        "-> score/judge -> report; fake components only, no external model)",
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from eval.report import Reporter, render_markdown

    try:
        report = Reporter(args.run).build()
    except (ContractError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(render_markdown(report))
    return 0


def _execute_resume(config: ExperimentConfig, args: argparse.Namespace) -> int:
    from pathlib import Path

    from eval.contracts.common import now_utc
    from eval.memories import build_memory_for_plan
    from eval.runs import RunStore

    try:
        adapter = build_memory_for_plan(config.memory)
    except ContractError as exc:
        _print_contract_error(exc)
        return 2
    run_dir = Path(args.run)
    store = RunStore(run_dir.parent, run_dir.name)
    if config.suite == "operations":
        from eval.operations import OperationsRunner

        runner = OperationsRunner(
            config=config, adapter=adapter, store=store, run_id=run_dir.name
        )
        try:
            outcome = runner.resume()
        except ContractError as exc:
            _print_contract_error(exc)
            return 2
        payload = {
            "command": "resume",
            "config": config.name,
            "config_fingerprint": config.fingerprint(),
            "metrics_registry_version": config.canonical_payload()[
                "metrics_registry_version"
            ],
            **outcome.payload(),
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0 if outcome.failed == 0 else 1

    from eval.datasets import load_dataset_for_config
    from eval.judges import build_judge_for_plan
    from eval.readers import build_reader_for_plan
    from eval.runner import OfflineRunner

    try:
        dataset = load_dataset_for_config(config, path=args.dataset)
    except ContractError as exc:
        _print_contract_error(exc)
        return 2
    try:
        runner = OfflineRunner(
            config=config,
            dataset=dataset,
            adapter=adapter,
            reader=build_reader_for_plan(config.reader),
            judge=build_judge_for_plan(config.judge),
            store=store,
            run_id=run_dir.name,
        )
    except ContractError as exc:
        _print_contract_error(exc)
        return 2
    try:
        outcome = runner.resume()
    except ContractError as exc:
        _print_contract_error(exc)
        return 2
    from eval.report import Reporter

    report = Reporter(outcome.run_dir).build()
    headline = {
        m["metric_id"]: m["value"] if m["status"] == "computed" else None
        for m in report["metrics"]
        if m["metric_id"]
        in (
            "planned_question_score",
            "scored_accuracy",
            "verifiable_session_recall_macro",
            "verifiable_session_recall_micro",
            "recall_at_1",
        )
    }
    payload = {
        "command": "resume",
        "run_id": outcome.run_id,
        "run_dir": str(outcome.run_dir),
        "config": config.name,
        "config_fingerprint": config.fingerprint(),
        "metrics_registry_version": config.canonical_payload()[
            "metrics_registry_version"
        ],
        "sample_count": len(outcome.results),
        "headline_metrics": headline,
        **outcome.summary(),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if outcome.failed == 0 and outcome.invalid_input == 0 else 1


def cmd_resume(args: argparse.Namespace) -> int:
    config = _load_config_or_exit(args.config)
    return _execute_resume(config, args)


def cmd_compare(args: argparse.Namespace) -> int:
    from eval.compare import CompareError, compare_runs, save_comparison

    try:
        payload = compare_runs(args.left, args.right)
    except (CompareError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    out_dir = args.out or str(Path(args.left).parent)
    refs = save_comparison(payload, out_dir)
    payload["artifacts"] = refs
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hippo-eval",
        description="Hippo memory eval harness (M1 offline scaffold).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser(
        "validate",
        help="validate experiment configs and manual fixtures structurally",
    )
    p_validate.add_argument(
        "--config",
        dest="configs",
        action="append",
        default=[],
        required=True,
        help="TOML config path (repeatable)",
    )
    p_validate.add_argument(
        "--fixture",
        dest="fixtures",
        action="append",
        default=[],
        help="JSON fixture path or directory (repeatable; kind from suffix)",
    )
    p_validate.set_defaults(func=cmd_validate)

    p_run = sub.add_parser("run", help="run an experiment (M1 offline loop)")
    p_run.add_argument("--config", required=True)
    p_run.add_argument(
        "--out",
        default=None,
        metavar="DIR",
        help="execute the offline loop and write the run directory here",
    )
    p_run.add_argument(
        "--dataset",
        default=None,
        metavar="JSON",
        help="dataset file override (default: the file the config's "
        "dataset_plan pins: bundled manual smoke samples, or "
        "data/longmemeval/longmemeval_s_cleaned.json after "
        "scripts/fetch_longmemeval.py)",
    )
    p_run.set_defaults(func=cmd_run)

    p_report = sub.add_parser(
        "report",
        help="rebuild and print the summary report of an existing run",
    )
    p_report.add_argument("--run", required=True, metavar="DIR")
    p_report.set_defaults(func=cmd_report)

    p_resume = sub.add_parser(
        "resume", help="resume a run from its checkpoint directory"
    )
    p_resume.add_argument("--config", required=True)
    p_resume.add_argument(
        "--run",
        required=True,
        metavar="DIR",
        help="existing run directory to resume (must match the config)",
    )
    p_resume.add_argument(
        "--dataset",
        default=None,
        metavar="JSON",
        help="dataset file override (default: the file the config's "
        "dataset_plan pins)",
    )
    p_resume.set_defaults(func=cmd_resume)

    p_compare = sub.add_parser(
        "compare",
        help="compare two run directories (key-field comparability, both "
        "full planned-set results and the common runnable intersection)",
    )
    p_compare.add_argument("left", metavar="RUN_DIR")
    p_compare.add_argument("right", metavar="RUN_DIR")
    p_compare.add_argument(
        "--out",
        default=None,
        metavar="DIR",
        help="directory for the compare artifact (default: the LEFT run's "
        "parent directory; run directories stay immutable)",
    )
    p_compare.set_defaults(func=cmd_compare)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
