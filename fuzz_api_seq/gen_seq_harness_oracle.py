#!/usr/bin/env python3
import argparse
import pprint
from pathlib import Path
from typing import Any, Dict, List, Set

import yaml


TEMPLATE = """\
import sys
import importlib
import atheris
import torch

from param_sampler import gen_config_for_api
from seq_env import SequenceEnv


# ===== 自动嵌入的 API 规格、约束和序列定义 =====

API_SPECS = __API_SPECS_LITERAL__

API_CONSTRAINTS = __API_CONSTRAINTS_LITERAL__

SEQUENCE_SPEC = __SEQ_SPEC_LITERAL__

SEQUENCE_ORACLES = __SEQ_ORACLES_LITERAL__


def constraint_func(cfg, constraints):
    \"\"\"通用约束检查函数（预条件）。

    - 把 cfg['_shape_vars'] 和 cfg 本身都摊平成局部变量，约束里可以直接写 N / C_in / input 等名字。
    - 自动构造 padding_tuple / stride_tuple / dilation_tuple。
    - constraints 是一组字符串表达式，用 eval(expr, {}, locs) 检查。
    \"\"\"
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

    # 如有需要，也可以在约束里用 torch
    locs["torch"] = torch

    for expr in constraints:
        try:
            if not eval(expr, {}, locs):
                return False
        except Exception:
            return False
    return True


def apply_input_bindings(cfg, step_spec, env: SequenceEnv, fdp: atheris.FuzzedDataProvider) -> bool:
    \"\"\"根据序列 YAML 里的 inputs 规则，覆盖 cfg 中的部分参数。

    支持几种简单 DSL：
      - "@tensor:name" -> 从 env 里拿名字为 name 的张量
      - "@env:any"     -> 从 env 里随机挑一个张量
      - 其他值         -> 按字面值写入（预留将来扩展）
    \"\"\"
    from seq_env import SequenceEnv  # 为了类型提示安静一点

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
            # 常量 / 未来可以扩展更多 DSL
            cfg[pname] = src
    return True


def call_api_by_name(api_name: str, call_kwargs: dict):
    \"\"\"根据字符串 api_name 加载并调用对应的函数。\"\"\"
    mod_name, func_name = api_name.rsplit(".", 1)
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, func_name)
    return fn(**call_kwargs)


def execute_sequence_once(sequence_spec: dict,
                          api_specs: dict,
                          api_constraints: dict,
                          fdp: atheris.FuzzedDataProvider):
    \"\"\"执行一条 API 序列。

    返回 (env, ok, step_cfgs):
      - env: SequenceEnv，里面存了所有中间 Tensor
      - ok: False 表示中途失败（约束不满足 / API 抛错等）
      - step_cfgs: {short_name: cfg}，用于 oracle 中访问 cfg_xxx
    \"\"\"
    env = SequenceEnv()
    steps = sequence_spec["steps"]
    step_cfgs = {}

    for step in steps:
        api_name = step["api"]
        spec = api_specs[api_name]
        constraints = api_constraints.get(api_name, []) or []

        # 1) 用单 API 的逻辑生成 cfg
        cfg = gen_config_for_api(spec, fdp)

        # 2) 应用序列里 inputs 的绑定（从 env 复用张量）
        if not apply_input_bindings(cfg, step, env, fdp):
            return env, False, step_cfgs

        # 3) 单 API 级别约束检查
        if constraints and not constraint_func(cfg, constraints):
            return env, False, step_cfgs

        # 4) 调用 API
        try:
            call_kwargs = {
                pname: cfg[pname]
                for pname in spec.get("params", {}).keys()
                if pname in cfg
            }
            out = call_api_by_name(api_name, call_kwargs)
        except (RuntimeError, ValueError, TypeError):
            # 认为是“正常报错”，对于 fuzz 来说只是一个普通分支
            return env, False, step_cfgs

        # 5) 记录该 step 的 cfg，按 api 短名映射为 cfg_xxx
        short_name = api_name.rsplit(".", 1)[1].replace(".", "_")
        step_cfgs[short_name] = cfg

        # 6) 把输出里的 Tensor 扔回 env
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
            for i, t in enumerate(out):
                if isinstance(t, torch.Tensor):
                    _add_one_tensor(t, i)

    return env, True, step_cfgs


def eval_sequence_oracles(env: SequenceEnv, step_cfgs: dict):
    \"\"\"在整条序列执行完后，执行 SEQUENCE_ORACLES 里的表达式。

    约定：
      - 所有 env 里的张量按 name 暴露为同名变量，例如 ln_out, drop_out, residual_out。
      - 每个 API 的 cfg 暴露为 cfg_<短名>，例如:
          * torch.nn.functional.layer_norm -> cfg_layer_norm
          * torch.nn.functional.dropout   -> cfg_dropout
          * torch.add                     -> cfg_add
    返回:
      (oracle_failed: bool)
    \"\"\"
    locs = {}

    # 1) 把所有命名 tensor 放进作用域
    for ti in env.tensors:
        locs[ti.name] = ti.tensor

    # 2) 把每个 step 的 cfg 放进作用域，名为 cfg_<short_name>
    for short_name, cfg in step_cfgs.items():
        locs[f"cfg_{short_name}"] = cfg

    # 3) 提供 torch
    locs["torch"] = torch

    oracle_failed = False

    for oracle in SEQUENCE_ORACLES:
        expr = oracle.get("expr")
        if not expr:
            continue
        try:
            ok = bool(eval(expr, {}, locs))
        except Exception:
            ok = False
        if not ok:
            oracle_failed = True

    return oracle_failed


# ========== Atheris Harness ==========

@atheris.instrument_func
def TestOneInput(data: bytes) -> None:
    \"\"\"Atheris 入口：从 bytes 生成并执行一次 API 序列。\"\"\"
    fdp = atheris.FuzzedDataProvider(data)

    env, ok, step_cfgs = execute_sequence_once(SEQUENCE_SPEC, API_SPECS, API_CONSTRAINTS, fdp)
    if not ok:
        return

    # 1) 基于序列级 oracle 做检查
    oracle_failed = eval_sequence_oracles(env, step_cfgs)

    # 2) 额外：简单 NaN/Inf 检查，仍然保留
    has_nan_or_inf = False
    for ti in env.tensors:
        t = ti.tensor
        if torch.isnan(t).any() or torch.isinf(t).any():
            has_nan_or_inf = True
            break

    # 这里只是把 oracle / NaN 结果折叠成一个分支信号给 fuzzer
    if oracle_failed or has_nan_or_inf:
        dummy = 1
    else:
        dummy = 0

    _ = dummy  # 避免未使用变量的警告


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
    """
    从 api_yaml_dir 下加载所有 *.yaml，抽出我们需要的那些 API 的 spec。
    返回:
      api_specs: { api_name: spec_without_constraints }
      api_constraints: { api_name: [expr_str, ...] }
    """
    api_specs: Dict[str, Dict[str, Any]] = {}
    api_constraints: Dict[str, List[str]] = {}

    # 一次性扫目录，建立 api_name -> spec
    for p in api_yaml_dir.glob("*.yaml"):
        spec = load_yaml(p)
        api_name = spec.get("api_name")
        if not api_name:
            continue
        if api_name not in required_api_names:
            continue

        spec_copy = dict(spec)
        constraints = spec_copy.pop("constraints", []) or []
        api_specs[api_name] = spec_copy
        api_constraints[api_name] = list(constraints)

    missing = required_api_names - set(api_specs.keys())
    if missing:
        raise RuntimeError(
            f"These api_name(s) are used in the sequence but no YAML spec found: {missing}"
        )

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

    # 收集该序列用到的 api_name 集合
    steps = seq_spec.get("steps", [])
    required_api_names: Set[str] = set()
    for s in steps:
        api_name = s.get("api")
        if api_name:
            required_api_names.add(api_name)

    if not required_api_names:
        raise RuntimeError(f"No 'api' found in sequence steps of {seq_path}")

    # 从 api_yaml_dir 里加载我们要用到的这些 API specs + constraints
    api_specs, api_constraints = build_api_mapping(api_dir, required_api_names)

    # 序列级 oracle
    seq_oracles = seq_spec.get("oracles", []) or []

    # 转成 Python 字面量字符串
    api_specs_literal = make_literal(api_specs)
    api_constraints_literal = make_literal(api_constraints)
    seq_spec_literal = make_literal(seq_spec)
    seq_oracles_literal = make_literal(seq_oracles)

    code = TEMPLATE.replace("__API_SPECS_LITERAL__", api_specs_literal)
    code = code.replace("__API_CONSTRAINTS_LITERAL__", api_constraints_literal)
    code = code.replace("__SEQ_SPEC_LITERAL__", seq_spec_literal)
    code = code.replace("__SEQ_ORACLES_LITERAL__", seq_oracles_literal)

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
    ap.add_argument(
        "--seq_yaml",
        required=True,
        help="序列 YAML 路径，例如 ./sequences/conv_block.yaml",
    )
    ap.add_argument(
        "--api_yaml_dir",
        required=True,
        help="单 API YAML 目录，例如 ./api_specs",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="输出的 .py 文件路径（可选，默认 auto_seq_<sequence_name>.py）",
    )
    args = ap.parse_args()

    generate_sequence_harness(args.seq_yaml, args.api_yaml_dir, args.out)


if __name__ == "__main__":
    main()
