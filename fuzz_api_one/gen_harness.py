# generate_from_yaml.py
import sys
from pathlib import Path
from typing import Any, Dict

import yaml
import pprint
import argparse


TEMPLATE = """\
import os
import sys
import importlib
import atheris
import torch

from param_sampler import gen_config_for_api

# 由 YAML 自动生成的 API 规格
SPEC = __SPEC_LITERAL__

# 从 YAML 中读取的约束表达式（字符串）
CONSTRAINTS = __CONSTRAINTS_LITERAL__


def constraint_func(cfg):
    \"\"\"通用约束检查函数（预条件）。

    - 把 cfg['_shape_vars'] 和 cfg 本身都摊平成局部变量，
      这样约束里写 C_in / groups / input 等名字可以直接用。
    - 自动构造 padding_tuple / stride_tuple / dilation_tuple，方便 conv2d 这类 API 使用。
    - 对于 CONSTRAINTS 里每一个表达式 expr，用 eval(expr, {}, locs) 检查。
    \"\"\"
    # 形状变量（来自采样器）
    shape_vars = cfg.get("_shape_vars", {})
    # 局部命名空间：先放 shape_vars，再放 cfg 本身（避免覆盖 _shape_vars 键）
    locs = dict(shape_vars)
    for k, v in cfg.items():
        if k == "_shape_vars":
            continue
        locs[k] = v

    # helper：把 int / list / tuple 统一成 2-tuple
    def _as_2tuple(x):
        if isinstance(x, (list, tuple)):
            return tuple(x)
        return (x, x)

    # conv2d 这类会用到的辅助变量（其他 API 不用也没关系）
    padding = cfg.get("padding", 0)
    stride = cfg.get("stride", 1)
    dilation = cfg.get("dilation", 1)

    locs["padding_tuple"] = _as_2tuple(padding)
    locs["stride_tuple"] = _as_2tuple(stride)
    locs["dilation_tuple"] = _as_2tuple(dilation)

    # 如有需要，也可以把 torch 放进作用域：
    # locs["torch"] = torch

    # 逐条检查约束表达式
    for expr in CONSTRAINTS:
        try:
            if not eval(expr, {}, locs):
                return False
        except Exception:
            # 表达式本身出错（比如写错名字），直接认为不满足
            return False
    return True


def gen_valid_config(spec, fdp, max_tries: int = 8):
    \"\"\"多次尝试生成满足约束的 cfg。\"\"\"
    from param_sampler import gen_config_for_api
    for _ in range(max_tries):
        cfg = gen_config_for_api(spec, fdp)
        if constraint_func(cfg):
            return cfg
    return None


def _call_target_api(cfg):
    \"\"\"根据 SPEC['api_name'] 和 SPEC['params'] 自动调用目标 API。

    约定：YAML 里 params 的 key 必须和真实 API 的参数名一致，
    我们统一用关键字调用：target(**call_kwargs)。
    \"\"\"
    api_name = SPEC.get("api_name")
    if not api_name:
        raise RuntimeError("SPEC missing 'api_name'")

    # 拆成模块名和函数名，例如 "torch.fft.fft" -> ("torch.fft", "fft")
    try:
        mod_name, func_name = api_name.rsplit(".", 1)
    except ValueError:
        # 没有点的情况，比如 "torch" 之类（一般不会）
        raise RuntimeError(f"Invalid api_name: {api_name!r}")

    mod = importlib.import_module(mod_name)
    target = getattr(mod, func_name)

    # 只从 cfg 里取出 SPEC['params'] 列出的参数
    call_kwargs = {}
    for pname in SPEC.get("params", {}).keys():
        if pname in cfg:
            call_kwargs[pname] = cfg[pname]

    # 直接关键字调用
    return target(**call_kwargs)


# ========== Atheris Harness ==========

@atheris.instrument_func
def TestOneInput(data: bytes):
    \"\"\"Atheris 入口：从 bytes 生成一次 API 调用。\"\"\"
    fdp = atheris.FuzzedDataProvider(data)
    cfg = gen_valid_config(SPEC, fdp)
    if cfg is None:
        # 这一串 bytes 很难凑出满足约束的参数，直接丢弃
        return

    try:
        # 自动调用 SPEC 对应的 API，例如 torch.fft.fft / torch.matmul / torch.where 等
        result = _call_target_api(cfg)
    except (RuntimeError, ValueError, TypeError):
        # 输入不合法 / 内部检查抛错：正常情况，不当作 crash
        return

    # 如果你后续在 YAML 里增加 result_constraints，
    # 可以在这里调用一个 result_oracle(cfg, result) 来做结果检查。
    # 目前先不做额外断言。
    _ = result


def main():
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
"""


def load_yaml_spec(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data


def make_spec_literal(spec: Dict[str, Any]) -> str:
    # constraints 单独拿出去，不放进 SPEC（避免重复）
    spec_copy = dict(spec)
    spec_copy.pop("constraints", None)
    # 不排序，保留原字段顺序
    return pprint.pformat(spec_copy, width=80, sort_dicts=False)


def make_constraints_literal(spec: Dict[str, Any]) -> str:
    constraints = spec.get("constraints", []) or []
    return pprint.pformat(constraints, width=80, sort_dicts=False)


def generate_from_yaml(yaml_path: str, out_path: str | None = None):
    yaml_file = Path(yaml_path)
    spec = load_yaml_spec(yaml_file)

    spec_literal = make_spec_literal(spec)
    constraints_literal = make_constraints_literal(spec)

    code = TEMPLATE.replace("__SPEC_LITERAL__", spec_literal)
    code = code.replace("__CONSTRAINTS_LITERAL__", constraints_literal)

    if out_path is None:
        # 默认输出文件名：auto_<api_name_sanitized>.py
        api_name = spec.get("api_name", yaml_file.stem)
        # 把 torch.nn.functional.conv2d 这种名字转成 torch_nn_functional_conv2d
        sanitized = api_name.replace(".", "_")
        out_file = yaml_file.with_name(f"auto_{sanitized}.py")
    else:
        out_file = Path(out_path)

    out_file.write_text(code, encoding="utf-8")
    print(f"[+] Generated {out_file}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--yaml",
        default="matmul.yaml",
        help="YAML spec path, e.g. ./matmul.yaml",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="输出的 .py 文件路径（可选，默认自动生成 auto_xxx.py）",
    )
    args = ap.parse_args()

    yaml_path = args.yaml
    out_path = args.out
    generate_from_yaml(yaml_path, out_path)

