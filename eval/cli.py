"""Hippo eval CLI: run / resume / compare / validate.

M1 scope: validate is fully offline and structural. run/resume/compare
print their planned behavior with the loaded config fingerprint and exit
non-zero on invalid input; full orchestration lands in later milestones.
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


def cmd_run(args: argparse.Namespace) -> int:
    config = _load_config_or_exit(args.config)
    plan = {
        "command": "run",
        "config": config.name,
        "config_fingerprint": config.fingerprint(),
        "metrics_registry_version": config.canonical_payload()[
            "metrics_registry_version"
        ],
        "sample_count": len(config.sample_ids),
        "status": "not-implemented (M2 wires the real runner; M1 validates "
        "configs, contracts and fixtures offline)",
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    config = _load_config_or_exit(args.config)
    plan = {
        "command": "resume",
        "config": config.name,
        "config_fingerprint": config.fingerprint(),
        "status": "not-implemented (checkpoint resume requires a run "
        "directory; M1 fixes the artifact schema versions it will check)",
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    left = _load_config_or_exit(args.left)
    right = _load_config_or_exit(args.right)
    same = left.fingerprint() == right.fingerprint()
    plan = {
        "command": "compare",
        "left": {"config": left.name, "fingerprint": left.fingerprint()},
        "right": {"config": right.name, "fingerprint": right.fingerprint()},
        "same_config_fingerprint": same,
        "same_sample_plan": left.sample_plan_id == right.sample_plan_id
        and set(left.sample_ids) == set(right.sample_ids),
        "status": "not-implemented (M1 performs config-level comparability "
        "checks only; metric-level comparison lands with the runner)",
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False))
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

    p_run = sub.add_parser("run", help="run an experiment (M2)")
    p_run.add_argument("--config", required=True)
    p_run.set_defaults(func=cmd_run)

    p_resume = sub.add_parser("resume", help="resume a run from checkpoints (M2)")
    p_resume.add_argument("--config", required=True)
    p_resume.set_defaults(func=cmd_resume)

    p_compare = sub.add_parser("compare", help="compare two runs (M2)")
    p_compare.add_argument("left")
    p_compare.add_argument("right")
    p_compare.set_defaults(func=cmd_compare)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
