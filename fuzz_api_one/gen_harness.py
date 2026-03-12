#!/usr/bin/env python3
# generate_from_yaml.py
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
#
# These knobs are NOT stored in YAML. Put them into your profile/runner env.
# ============================================================

# 由 YAML 自动生成的 API 规格
SPEC = __SPEC_LITERAL__

# 从 YAML 中读取的约束表达式（字符串）
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
    \"\"\"Derive a stable seed from fuzz input bytes.\"\"\"
    h = hashlib.sha1(data).digest()
    return int.from_bytes(h[:8], "little") & 0x7FFFFFFF


# ---- Profile knobs (defaults keep your current behavior) ----
# 生成 seed cfg 时最多尝试多少次（对应 gen_valid_config 的 max_tries）
SEED_TRIES = _env_int("SEED_TRIES", 8)

# mutation: 每个 input 最多做多少步变异（steps 的上限）
MUT_STEPS_MAX = _env_int("MUT_STEPS_MAX", 10)

# mutation: 每步变异失败时最多重试多少次（max_attempts_per_step）
MUT_ATTEMPTS = _env_int("MUT_ATTEMPTS", 6)

# mutation: 类型变异 / shape 变异概率
P_TYPE_MUT = _env_float("P_TYPE_MUT", 0.8)
P_SHAPE_MUT = _env_float("P_SHAPE_MUT", 0.30)


def constraint_func(cfg):
    \"\"\"通用约束检查函数（预条件）。

    - 把 cfg['_shape_vars'] 和 cfg 本身都摊平成局部变量
    - 自动构造 padding_tuple / stride_tuple / dilation_tuple
    - 每条 CONSTRAINTS 表达式用 eval(expr, {}, locs) 检查

    NOTE:
      - locs 注入 torch/math，方便你写 torch.isfinite / math.gcd 等约束
    \"\"\"
    shape_vars = cfg.get("_shape_vars", {})

    locs = dict(shape_vars)
    for k, v in cfg.items():
        if k == "_shape_vars":
            continue
        locs[k] = v

    # helper：把 int / list / tuple 统一成 tuple（不强制 2 维）
    def _as_tuple(x):
        if isinstance(x, (list, tuple)):
            return tuple(x)
        return (x,)

    # conv-like helper（其他 API 不用也没关系）
    padding = cfg.get("padding", 0)
    stride = cfg.get("stride", 1)
    dilation = cfg.get("dilation", 1)

    locs["padding_tuple"] = _as_tuple(padding)
    locs["stride_tuple"] = _as_tuple(stride)
    locs["dilation_tuple"] = _as_tuple(dilation)

    # allow torch/math in constraint expressions
    locs["torch"] = torch
    locs["math"] = math

    for expr in CONSTRAINTS:
        try:
            if not eval(expr, {}, locs):
                return False
        except Exception:
            return False
    return True


def gen_valid_config(spec, fdp, max_tries: int = None):
    \"\"\"多次尝试生成满足约束的 cfg。\"\"\"
    if max_tries is None:
        max_tries = SEED_TRIES
    for _ in range(max_tries):
        cfg = gen_config_for_api(spec, fdp)
        if constraint_func(cfg):
            return cfg
    return None


def _call_target_api(cfg):
    \"\"\"根据 SPEC['api_name'] 和 SPEC['params'] 自动调用目标 API。\"\"\"
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
        # 兜底：不要把“普通失败”当 crash；Atheris 会自己抓真正崩溃
        return


def main():
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
"""


def safe_name(s: Any, max_len: int = 120) -> str:
    """
    Convert arbitrary string to a filesystem-safe name.
    """
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
    spec_copy = dict(spec)
    spec_copy.pop("constraints", None)
    return pprint.pformat(spec_copy, width=80, sort_dicts=False)


def make_constraints_literal(spec: Dict[str, Any]) -> str:
    constraints = spec.get("constraints", []) or []
    return pprint.pformat(constraints, width=80, sort_dicts=False)


def get_rank_candidates(spec: Dict[str, Any]) -> List[int]:
    """
    从 YAML 里读取 rank_hints.rank_candidates，返回一个去重后的 int 列表（保持顺序）。
    """
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
    """
    生成单个 harness 文件。
    当前 harness 模板本身并不直接使用 active_rank（rank 选择逻辑在 param_sampler 或约束里体现）。
    这里主要用于命名/区分输出。
    """
    spec_literal = make_spec_literal(spec)
    constraints_literal = make_constraints_literal(spec)

    code = TEMPLATE.replace("__SPEC_LITERAL__", spec_literal)
    code = code.replace("__CONSTRAINTS_LITERAL__", constraints_literal)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(code, encoding="utf-8")
    print(f"[+] Generated {out_file} (rank={active_rank})")


def generate_from_yaml(
    yaml_path: str,
    out_path: str | None = None,
    out_dir: str | None = None,
    single: bool = False,
):
    yaml_file = Path(yaml_path)
    spec = load_yaml_spec(yaml_file)
    ranks = get_rank_candidates(spec)

    out_dir_path = Path(out_dir).resolve() if out_dir else None

    # 如果用户指定 --out，则只生成一个文件（完全按 out 路径）
    if out_path is not None:
        out_file = Path(out_path)
        active_rank = ranks[0] if ranks else None
        generate_one(yaml_file, spec, out_file, active_rank)
        return

    # single 模式：即使有 ranks 也只生成一个（取第一个 rank）
    if single:
        active_rank = ranks[0] if ranks else None
        out_file = build_default_out_path(yaml_file, spec, active_rank, out_dir_path)
        generate_one(yaml_file, spec, out_file, active_rank)
        return

    # 默认：有 rank_candidates => 每个 rank 一个 harness；没有 => 一个
    if ranks:
        for r in ranks:
            out_file = build_default_out_path(yaml_file, spec, r, out_dir_path)
            generate_one(yaml_file, spec, out_file, r)
    else:
        out_file = build_default_out_path(yaml_file, spec, None, out_dir_path)
        generate_one(yaml_file, spec, out_file, None)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", default="matmul.yaml", help="YAML spec path, e.g. ./matmul.yaml")
    ap.add_argument("--out", default=None, help="output .py path (optional, generates only one harness)")
    ap.add_argument("--out_dir", default=None, help="output directory (optional, for auto naming; ignored if --out is set)")
    ap.add_argument("--single", action="store_true", help="generate only one harness even if multiple ranks exist")
    args = ap.parse_args()
    generate_from_yaml(args.yaml, args.out, args.out_dir, args.single)