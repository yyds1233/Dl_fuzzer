# screen/cli/main.py
from __future__ import annotations

import argparse
from pathlib import Path

from screen.bandit_audit_driver_hier import orchestrate
from screen.config.schema import AuditParams, BanditParams, DriverConfig, RuntimeParams


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()

    # Inputs
    ap.add_argument("--harnesses_json", default=None)
    ap.add_argument("--harness", default=None)
    ap.add_argument("--harness_id", default=None)
    ap.add_argument("--top_json", default=None)

    # Common
    ap.add_argument("--root", default="fuzz_output")
    ap.add_argument("--python", default="python3")
    ap.add_argument("--epoch", type=int, default=60)
    ap.add_argument("--steps", type=int, default=200, help=">0: run N steps; 0: run forever")
    ap.add_argument("--fuzz_flags", default="-ignore_timeouts=1 -rss_limit_mb=4096 -use_value_profile=1 -entropic=1")
    ap.add_argument("--mix", type=float, default=0.7)

    # bandit params
    ap.add_argument("--c_fast", type=float, default=2.0)
    ap.add_argument("--c_slow", type=float, default=2.0)
    ap.add_argument("--epsilon_harness", type=float, default=0.02)
    ap.add_argument("--epsilon_profile", type=float, default=0.05)
    ap.add_argument("--alpha_min", type=float, default=0.2)
    ap.add_argument("--elim_margin", type=float, default=0.0)
    ap.add_argument("--elim_patience", type=int, default=3)
    ap.add_argument("--elim_min_pulls", type=int, default=8)
    ap.add_argument("--cooldown_steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)

    # manifests
    ap.add_argument("--manifest_dir", default="manifests")

    # audit
    ap.add_argument("--audit_every", type=int, default=10)
    ap.add_argument("--full_corpus_audit", action="store_true")
    ap.add_argument("--audit_max_inputs", type=int, default=2000)
    ap.add_argument("--audit_profile_topk", type=int, default=5)

    ap.add_argument("--cov_venv_activate", default="/root/pytorch_cov/bin/activate")
    ap.add_argument("--cov_audit_script", required=True)
    ap.add_argument("--global_dir", default="global_union")
    ap.add_argument("--primary_object", required=True)
    ap.add_argument("--extra_object", action="append", default=[])
    ap.add_argument("--ignore_filename_regex", default=None)
    ap.add_argument("--cov_replay_extra", default="")

    ap.add_argument("--slow_metric", choices=["BRH", "LH", "FNH"], default="BRH")
    ap.add_argument("--min_credit_inputs", type=int, default=20)
    ap.add_argument("--zero_slow_penalty", type=float, default=0.0)

    return ap


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()

    # basic validation
    if not args.harnesses_json:
        if not args.harness or not args.top_json:
            raise SystemExit("Need --harnesses_json OR legacy (--harness and --top_json)")

    runtime = RuntimeParams(
        root=Path(args.root),
        python=args.python,
        epoch=args.epoch,
        steps=args.steps,
        fuzz_flags=args.fuzz_flags,
        mix=args.mix,
        manifest_dir=Path(args.manifest_dir),
    )

    bandit = BanditParams(
        c_fast=args.c_fast,
        c_slow=args.c_slow,
        epsilon_harness=args.epsilon_harness,
        epsilon_profile=args.epsilon_profile,
        alpha_min=args.alpha_min,
        elim_margin=args.elim_margin,
        elim_patience=args.elim_patience,
        elim_min_pulls=args.elim_min_pulls,
        cooldown_steps=args.cooldown_steps,
        seed=args.seed,
    )

    audit = AuditParams(
        audit_every=args.audit_every,
        full_corpus_audit=args.full_corpus_audit,
        audit_max_inputs=args.audit_max_inputs,
        audit_profile_topk=args.audit_profile_topk,
        slow_metric=args.slow_metric,
        min_credit_inputs=args.min_credit_inputs,
        zero_slow_penalty=args.zero_slow_penalty,

        cov_venv_activate=Path(args.cov_venv_activate),
        cov_audit_script=Path(args.cov_audit_script),
        global_dir=Path(args.global_dir),
        primary_object=args.primary_object,
        extra_object=list(args.extra_object or []),
        ignore_filename_regex=args.ignore_filename_regex,
        cov_replay_extra=args.cov_replay_extra,
    )

    cfg = DriverConfig(
        runtime=runtime,
        bandit=bandit,
        audit=audit,
        harnesses_json=Path(args.harnesses_json) if args.harnesses_json else None,
        harness=Path(args.harness) if args.harness else None,
        harness_id=args.harness_id,
        top_json=Path(args.top_json) if args.top_json else None,
    )

    orchestrate(cfg)


if __name__ == "__main__":
    main()
