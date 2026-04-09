#!/usr/bin/env python3
# generate_from_yaml.py
#
# FIXED: Multi-rank support
#   - Default: generate ONE harness per YAML (not one per rank)
#   - Rank selection happens at RUNTIME in param_sampler.pick_rank()
#   - constraint_func checks both global constraints AND per-rank constraints
#   - Still supports --per_rank flag for legacy one-harness-per-rank mode
#
import sys
from pathlib import Path
from typing import Any, Dict, Optional, List

import yaml
import pprint
import argparse
import re


TEMPLATE = """\
import os
import sys
import importlib
import hashlib
import random
import atheris
import torch
import math

from utils.param_sampler import gen_config_for_api, mutate_cfg

# ============================================================
# Generated harness from YAML spec
#
# Multi-rank: rank is selected at RUNTIME by param_sampler.pick_rank()
# from rank_hints.rank_candidates. Each fuzz input may test a different rank.
#
# NOTE: param_sampler.py reads extra diversity knobs via env:
#
#   # non-contiguous / stride diversity (default OFF)
#   export P_NONCONTIG=0.05
#   export P_RECONTIG=0.10
#
#   # empty-dim (0-length) diversity (default OFF)
#   export ALLOW_EMPTY=1
#   export P_EMPTY_DIM=0.01
#   export P_EMPTY_NC=0.001   # DANGEROUS: usually keep 0
# ============================================================

# Full spec from YAML (includes rank_hints, shape_spec_by_rank, etc.)
SPEC = __SPEC_LITERAL__

# Global constraints from YAML
CONSTRAINTS = __CONSTRAINTS_LITERAL__


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except Exception:
        return default


def _seed_from_bytes(data: bytes) -> int:
    h = hashlib.sha1(data).digest()
    return int.from_bytes(h[:8], "little") & 0x7FFFFFFF


SEED_TRIES = _env_int("SEED_TRIES", 8)
MUT_STEPS_MAX = _env_int("MUT_STEPS_MAX", 10)
MUT_ATTEMPTS = _env_int("MUT_ATTEMPTS", 6)
P_TYPE_MUT = _env_float("P_TYPE_MUT", 0.8)
P_SHAPE_MUT = _env_float("P_SHAPE_MUT", 0.30)


def constraint_func(cfg):
    \"\"\"
    Check constraints for a given cfg.

    Checks TWO sources:
      1) CONSTRAINTS: global constraints from YAML top-level
      2) cfg['_rank_constraints']: per-rank constraints injected by param_sampler
         (from params[x].constraints_by_rank[active_rank])
    \"\"\"
    shape_vars = cfg.get("_shape_vars", {})

    locs = dict(shape_vars)
    for k, v in cfg.items():
        if k.startswith("_"):
            continue
        locs[k] = v

    def _as_tuple(x):
        if isinstance(x, (list, tuple)):
            return tuple(x)
        return (x,)

    padding = cfg.get("padding", 0)
    stride = cfg.get("stride", 1)
    dilation = cfg.get("dilation", 1)
    kernel_size = cfg.get("kernel_size", 1)

    locs["padding_tuple"] = _as_tuple(padding)
    locs["stride_tuple"] = _as_tuple(stride)
    locs["dilation_tuple"] = _as_tuple(dilation)
    locs["kernel_size_tuple"] = _as_tuple(kernel_size)
    locs["torch"] = torch
    locs["math"] = math

    # 1) Check global constraints
    for expr in CONSTRAINTS:
        try:
            if not eval(expr, {}, locs):
                return False
        except Exception:
            return False

    # 2) Check per-rank constraints (injected by param_sampler)
    rank_constraints = cfg.get("_rank_constraints", [])
    for expr in rank_constraints:
        try:
            if not eval(expr, {}, locs):
                return False
        except Exception:
            return False

    return True


def gen_valid_config(spec, fdp, max_tries: int = None):
    if max_tries is None:
        max_tries = SEED_TRIES
    for _ in range(max_tries):
        cfg = gen_config_for_api(spec, fdp)
        if constraint_func(cfg):
            return cfg
    return None


def _call_target_api(cfg):
    api_name = SPEC.get("api_name")
    if not api_name:
        raise RuntimeError("SPEC missing 'api_name'")

    try:
        mod_name, func_name = api_name.rsplit(".", 1)
    except ValueError:
        raise RuntimeError(f"Invalid api_name: {api_name!r}")

    mod = importlib.import_module(mod_name)
    target = getattr(mod, func_name)

    call_kwargs = {}
    for pname in SPEC.get("params", {}).keys():
        if pname in cfg:
            call_kwargs[pname] = cfg[pname]

    return target(**call_kwargs)


@atheris.instrument_func
def TestOneInput(data: bytes):
    seed = _seed_from_bytes(data)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    fdp = atheris.FuzzedDataProvider(data)

    cfg = gen_valid_config(SPEC, fdp)
    if cfg is None:
        return

    n_params = len(SPEC.get("params", {}))
    upper = max(1, min(MUT_STEPS_MAX, n_params))
    steps = fdp.ConsumeIntInRange(1, upper)

    cfg = mutate_cfg(
        SPEC, cfg, fdp,
        constraint_func=constraint_func,
        steps=steps,
        max_attempts_per_step=max(1, MUT_ATTEMPTS),
        p_type_mut=max(0.0, min(1.0, P_TYPE_MUT)),
        p_shape_mut=max(0.0, min(1.0, P_SHAPE_MUT)),
    )

    try:
        _ = _call_target_api(cfg)
    except (RuntimeError, ValueError, TypeError, AssertionError):
        return
    except Exception:
        return


def main():
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
"""


def safe_name(s: Any, max_len: int = 120) -> str:
    if s is None:
        return "null"
    s = str(s).strip()
    if not s:
        return "empty"
    s = s.replace("::", "_").replace("/", "_").replace("\\", "_")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("._-")
    if not s:
        s = "empty"
    return s[:max_len]


def load_yaml_spec(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"YAML spec must be a mapping/dict: {path}")
    return data


def make_spec_literal(spec: Dict[str, Any]) -> str:
    """
    Serialize spec dict as Python literal for embedding in harness code.
    Removes 'constraints' (handled separately) and 'generator' (metadata only).
    """
    spec_copy = dict(spec)
    spec_copy.pop("constraints", None)
    spec_copy.pop("generator", None)
    return pprint.pformat(spec_copy, width=80, sort_dicts=False)


def make_constraints_literal(spec: Dict[str, Any]) -> str:
    constraints = spec.get("constraints", []) or []
    return pprint.pformat(constraints, width=80, sort_dicts=False)


def get_rank_candidates(spec: Dict[str, Any]) -> List[int]:
    rh = spec.get("rank_hints")
    if not isinstance(rh, dict):
        return []
    cands = rh.get("rank_candidates")
    if not isinstance(cands, list):
        return []
    out: List[int] = []
    seen = set()
    for x in cands:
        if isinstance(x, int) and x not in seen:
            out.append(x)
            seen.add(x)
    return out


def build_default_out_path(
    yaml_file: Path,
    spec: Dict[str, Any],
    rank: Optional[int],
    out_dir: Optional[Path],
) -> Path:
    api_name = spec.get("api_name", yaml_file.stem)
    aten = spec.get("aten") or {}
    overload = "default"
    if isinstance(aten, dict):
        overload = aten.get("overload") or "default"

    base = f"auto_{safe_name(api_name)}__ov_{safe_name(overload)}"
    if isinstance(rank, int):
        base += f"__rank{rank}"
    filename = base + ".py"

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / filename
    return yaml_file.with_name(filename)


def generate_one(yaml_file: Path, spec: Dict[str, Any], out_file: Path, active_rank: Optional[int]):
    spec_literal = make_spec_literal(spec)
    constraints_literal = make_constraints_literal(spec)

    code = TEMPLATE.replace("__SPEC_LITERAL__", spec_literal)
    code = code.replace("__CONSTRAINTS_LITERAL__", constraints_literal)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(code, encoding="utf-8")

    ranks = get_rank_candidates(spec)
    rank_info = f"rank={active_rank}" if active_rank is not None else f"multi-rank={ranks}" if ranks else "no-rank"
    print(f"[+] Generated {out_file} ({rank_info})")


def generate_from_yaml(
    yaml_path: str,
    out_path: str | None = None,
    out_dir: str | None = None,
    per_rank: bool = False,
):
    yaml_file = Path(yaml_path)
    spec = load_yaml_spec(yaml_file)
    ranks = get_rank_candidates(spec)
    out_dir_path = Path(out_dir).resolve() if out_dir else None

    # If user specified --out, generate exactly one file
    if out_path is not None:
        out_file = Path(out_path)
        generate_one(yaml_file, spec, out_file, active_rank=None)
        return

    # --per_rank: legacy mode, one harness per rank
    if per_rank and ranks:
        for r in ranks:
            out_file = build_default_out_path(yaml_file, spec, r, out_dir_path)
            generate_one(yaml_file, spec, out_file, active_rank=r)
        return

    # DEFAULT: single harness, multi-rank handled at runtime
    out_file = build_default_out_path(yaml_file, spec, None, out_dir_path)
    generate_one(yaml_file, spec, out_file, active_rank=None)

def generate_from_yaml_dir(
    yaml_dir: str,
    out_dir: str | None = None,
    per_rank: bool = False,
):
    yaml_dir_path = Path(yaml_dir)
    if not yaml_dir_path.is_dir():
        raise ValueError(f"--yaml_dir is not a valid directory: {yaml_dir}")

    yaml_files = sorted(list(yaml_dir_path.glob("*.yaml")) + list(yaml_dir_path.glob("*.yml")))
    if not yaml_files:
        print(f"No YAML files found in directory: {yaml_dir}")
        return

    for yaml_file in yaml_files:
        generate_from_yaml(
            yaml_path=str(yaml_file),
            out_path=None,   # 批量模式下不支持单独 --out
            out_dir=out_dir,
            per_rank=per_rank,
        )



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", default=None, help="YAML spec path")
    ap.add_argument("--yaml_dir", default=None, help="where all YAML files save")
    ap.add_argument("--out", default=None, help="output .py path (generates one harness)")
    ap.add_argument("--out_dir", default=None, help="output directory for auto naming")
    ap.add_argument(
        "--per_rank",
        action="store_true",
        help="legacy mode: generate one harness per rank "
             "(default: single harness, rank selected at runtime)"
    )
    args = ap.parse_args()

    if args.yaml_dir is not None:
        if args.out is not None:
            raise ValueError("--out cannot be used together with --yaml_dir")
        generate_from_yaml_dir(args.yaml_dir, args.out_dir, args.per_rank)
    elif args.yaml is not None:
        generate_from_yaml(args.yaml, args.out, args.out_dir, args.per_rank)
    else:
        raise ValueError("Either --yaml or --yaml_dir must be provided")
