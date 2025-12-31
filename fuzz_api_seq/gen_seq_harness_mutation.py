#!/usr/bin/env python3
import argparse
import pprint
from pathlib import Path
from typing import Any, Dict, List, Set

import yaml

TEMPLATE = """\
import sys
import os
import importlib
import atheris
import torch

from utils.param_sampler import gen_config_for_api, mutate_cfg
from utils.seq_env import SequenceEnv
from utils.oracle_runtime import eval_sequence_oracles


# ===== 自动嵌入的 API 规格、约束和序列定义 =====

API_SPECS = __API_SPECS_LITERAL__
API_CONSTRAINTS = __API_CONSTRAINTS_LITERAL__
SEQUENCE_SPEC = __SEQ_SPEC_LITERAL__
SEQUENCE_ORACLES = __SEQ_ORACLES_LITERAL__


# ===== helpers: env knobs (compatible with screening profiles) =====

def _env_int(names, default: int) -> int:
    for n in names:
        v = os.getenv(n)
        if v is None:
            continue
        try:
            return int(v)
        except Exception:
            continue
    return int(default)

def _env_float(names, default: float) -> float:
    for n in names:
        v = os.getenv(n)
        if v is None:
            continue
        try:
            return float(v)
        except Exception:
            continue
    return float(default)

# 兼容两套命名：优先 SEQ_*，否则回退到通用 MUT_* / P_*
MUT_STEPS_MAX = _env_int(["SEQ_MUT_STEPS_MAX", "MUT_STEPS_MAX"], 6)
P_TYPE_MUT    = _env_float(["SEQ_MUT_TYPE_P", "P_TYPE_MUT"], 0.35)
P_SHAPE_MUT   = _env_float(["SEQ_MUT_SHAPE_P", "P_SHAPE_MUT"], 0.10)
MUT_ATTEMPTS  = _env_int(["SEQ_MUT_ATTEMPTS", "MUT_ATTEMPTS"], 6)
SEED_TRIES    = _env_int(["SEQ_SEED_TRIES", "SEED_TRIES"], 8)
# 可选：允许你筛选时彻底关掉变异
ENABLE_MUTATION = _env_int(["SEQ_ENABLE_MUT", "ENABLE_MUT"], 1) != 0


def constraint_func(cfg, constraints):
    \"\"\"通用约束检查函数（预条件）。\"\"\"
    shape_vars = cfg.get("_shape_vars", {})
    locs = dict(shape_vars)
    for k, v in cfg.items():
        if k == "_shape_vars":
            continue
        locs[k] = v

    def _as_2tuple(x):
        if isinstance(x, (list, tuple)):
            return tuple(x)
        return (x, x)

    padding = cfg.get("padding", 0)
    stride = cfg.get("stride", 1)
    dilation = cfg.get("dilation", 1)

    locs["padding_tuple"] = _as_2tuple(padding)
    locs["stride_tuple"] = _as_2tuple(stride)
    locs["dilation_tuple"] = _as_2tuple(dilation)
    locs["torch"] = torch

    for expr in constraints:
        try:
            if not eval(expr, {}, locs):
                return False
        except Exception:
            return False
    return True


def apply_input_bindings(cfg, step_spec, env: SequenceEnv, fdp: atheris.FuzzedDataProvider) -> bool:
    \"\"\"根据序列 YAML 里的 inputs 规则，覆盖 cfg 中的部分参数。\"\"\"
    inputs_spec = step_spec.get("inputs", {}) or {}
    for pname, src in inputs_spec.items():
        if isinstance(src, str) and src.startswith("@tensor:"):
            t_name = src.split(":", 1)[1]
            t = env.get_tensor_by_name(t_name)
            if t is None:
                return False
            cfg[pname] = t
        elif src == "@env:any":
            t = env.pick_any_tensor(fdp)
            if t is None:
                return False
            cfg[pname] = t
        else:
            cfg[pname] = src
    return True


def call_api_by_name(api_name: str, call_kwargs: dict):
    mod_name, func_name = api_name.rsplit(".", 1)
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, func_name)
    return fn(**call_kwargs)


def _gen_one_step_cfg(api_name: str, step_spec: dict, env: SequenceEnv, fdp: atheris.FuzzedDataProvider):
    \"\"\"为某个 step 生成一次可用 cfg（带重试 + 可控变异）。\"\"\"
    spec = API_SPECS[api_name]
    constraints = API_CONSTRAINTS.get(api_name, []) or []
    fixed_params = set((step_spec.get("inputs", {}) or {}).keys())

    # per-step 重试，避免整条序列因为一次 bad cfg 就直接失败
    for _ in range(max(1, SEED_TRIES)):
        cfg = gen_config_for_api(spec, fdp)

        if not apply_input_bindings(cfg, step_spec, env, fdp):
            continue

        # 变异：只允许变异非固定参数
        if ENABLE_MUTATION:
            n_params = len(spec.get("params", {}))
            upper = max(1, min(MUT_STEPS_MAX, n_params))
            msteps = fdp.ConsumeIntInRange(1, upper)

            # 若有固定参数（尤其固定 tensor），shape mutation 很容易导致 shape_vars 与固定 tensor 不一致
            if fixed_params:
                mutable_spec = dict(spec)
                mutable_spec["params"] = {
                    k: v for k, v in spec.get("params", {}).items() if k not in fixed_params
                }
                shape_p = 0.0
            else:
                mutable_spec = spec
                shape_p = P_SHAPE_MUT

            # 让 mutate 内部就用约束回调“自我筛掉坏候选”，提升有效 cfg 产出率
            def _cfunc(trial_cfg):
                if constraints and not constraint_func(trial_cfg, constraints):
                    return False
                return True

            cfg2 = mutate_cfg(
                mutable_spec,
                cfg,
                fdp,
                constraint_func=_cfunc if constraints else None,
                steps=msteps,
                max_attempts_per_step=max(1, MUT_ATTEMPTS),
                p_type_mut=P_TYPE_MUT,
                p_shape_mut=shape_p,
            )
            if cfg2 is None:
                continue
            cfg = cfg2

            # 兜底：再绑定一次固定输入（防未来改动把 fixed 混进 mutable）
            if fixed_params:
                if not apply_input_bindings(cfg, step_spec, env, fdp):
                    continue

        # 最终约束检查（即使 mutate 已检查，也保留一次）
        if constraints and not constraint_func(cfg, constraints):
            continue

        return cfg

    return None


def execute_sequence_once(sequence_spec: dict,
                          fdp: atheris.FuzzedDataProvider):
    \"\"\"执行一条 API 序列。返回 (env, ok, step_cfgs)。\"\"\"
    env = SequenceEnv()
    steps = sequence_spec["steps"]
    step_cfgs = {}

    for i, step in enumerate(steps):
        api_name = step["api"]

        cfg = _gen_one_step_cfg(api_name, step, env, fdp)
        if cfg is None:
            return env, False, step_cfgs

        # 调用 API
        try:
            spec = API_SPECS[api_name]
            call_kwargs = {
                pname: cfg[pname]
                for pname in spec.get("params", {}).keys()
                if pname in cfg
            }
            out = call_api_by_name(api_name, call_kwargs)
        except (RuntimeError, ValueError, TypeError):
            return env, False, step_cfgs

        # 记录 cfg：加 step index 防止同名 API 重复覆盖
        func_short = api_name.rsplit(".", 1)[1].replace(".", "_")
        step_cfgs[f"step{i}_{func_short}"] = cfg

        # 输出 tensor 入 env
        out_names = [o["name"] for o in step.get("outputs", [])]

        def _add_one_tensor(t: torch.Tensor, idx: int):
            if idx < len(out_names):
                name = out_names[idx]
            else:
                name = f"{api_name}_out{idx}"
            env.add_tensor(t, name=name, origin_api=api_name)

        if isinstance(out, torch.Tensor):
            _add_one_tensor(out, 0)
        elif isinstance(out, (list, tuple)):
            for j, t in enumerate(out):
                if isinstance(t, torch.Tensor):
                    _add_one_tensor(t, j)

    return env, True, step_cfgs


# ========== Atheris Harness ==========

@atheris.instrument_func
def TestOneInput(data: bytes) -> None:
    fdp = atheris.FuzzedDataProvider(data)

    env, ok, step_cfgs = execute_sequence_once(SEQUENCE_SPEC, fdp)
    if not ok:
        return

    # 序列级 oracle reward
    oracle_reward, oracle_violation = eval_sequence_oracles(env, step_cfgs, SEQUENCE_ORACLES)

    # NaN/Inf 检查（保留）
    has_nan_or_inf = False
    for ti in env.tensors:
        t = ti.tensor
        if torch.isnan(t).any() or torch.isinf(t).any():
            has_nan_or_inf = True
            break

    total_reward = oracle_reward + (1.0 if has_nan_or_inf else 0.0)

    # reward 分桶给 coverage “造分支”
    if total_reward <= 0.0:
        bucket = 0
    elif total_reward < 1.0:
        bucket = 1
    elif total_reward < 5.0:
        bucket = 2
    else:
        bucket = 3

    _ = bucket


def main():
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
"""


# ===== 下面是生成器部分 =====

def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_api_mapping(api_yaml_dir: Path, required_api_names: Set[str]):
    api_specs: Dict[str, Dict[str, Any]] = {}
    api_constraints: Dict[str, List[str]] = {}

    for p in Path(api_yaml_dir).glob("*.yaml"):
        spec = load_yaml(p)
        api_name = spec.get("api_name")
        if not api_name or api_name not in required_api_names:
            continue

        spec_copy = dict(spec)
        constraints = spec_copy.pop("constraints", []) or []
        api_specs[api_name] = spec_copy
        api_constraints[api_name] = list(constraints)

    missing = required_api_names - set(api_specs.keys())
    if missing:
        raise RuntimeError(f"Missing YAML spec(s) for: {missing}")

    return api_specs, api_constraints

def make_literal(obj: Any) -> str:
    return pprint.pformat(obj, width=80, sort_dicts=False)

def generate_sequence_harness(seq_yaml: str, api_yaml_dir: str, out_path: str | None = None):
    seq_path = Path(seq_yaml)
    api_dir = Path(api_yaml_dir)

    if not seq_path.is_file():
        raise FileNotFoundError(f"Sequence YAML not found: {seq_path}")
    if not api_dir.is_dir():
        raise NotADirectoryError(f"API YAML directory not found: {api_dir}")

    seq_spec = load_yaml(seq_path)
    steps = seq_spec.get("steps", [])
    required_api_names: Set[str] = {s["api"] for s in steps if s.get("api")}

    if not required_api_names:
        raise RuntimeError(f"No 'api' found in sequence steps of {seq_path}")

    api_specs, api_constraints = build_api_mapping(api_dir, required_api_names)
    seq_oracles = seq_spec.get("oracles", []) or []

    code = TEMPLATE.replace("__API_SPECS_LITERAL__", make_literal(api_specs))
    code = code.replace("__API_CONSTRAINTS_LITERAL__", make_literal(api_constraints))
    code = code.replace("__SEQ_SPEC_LITERAL__", make_literal(seq_spec))
    code = code.replace("__SEQ_ORACLES_LITERAL__", make_literal(seq_oracles))

    if out_path is None:
        seq_name = seq_spec.get("sequence_name", seq_path.stem)
        sanitized = seq_name.replace(" ", "_")
        out_file = seq_path.with_name(f"auto_seq_{sanitized}.py")
    else:
        out_file = Path(out_path)

    out_file.write_text(code, encoding="utf-8")
    print(f"[+] Generated sequence harness: {out_file}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_yaml", required=True, help="序列 YAML 路径")
    ap.add_argument("--api_yaml_dir", required=True, help="单 API YAML 目录")
    ap.add_argument("--out", default=None, help="输出的 .py 文件路径（可选）")
    args = ap.parse_args()
    generate_sequence_harness(args.seq_yaml, args.api_yaml_dir, args.out)

if __name__ == "__main__":
    main()
