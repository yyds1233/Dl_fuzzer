# param_sampler.py
import math
import random
from typing import Any, Dict, List, Tuple

import atheris
import torch


# ---- 1. shape vars 采样 ----

def gen_shape_vars(spec: Dict[str, Any], fdp: atheris.FuzzedDataProvider) -> Dict[str, int]:
    """
    根据 spec['shape_vars'] 生成隐含的形状变量。
    shape_vars: { "N": [min, max], "C": [min, max], ... }
    """
    shape_vars = {}
    for name, (lo, hi) in spec.get("shape_vars", {}).items():
        # Atheris: ConsumeIntInRange(lo, hi)
        val = fdp.ConsumeIntInRange(int(lo), int(hi))
        shape_vars[name] = val
    return shape_vars


# ---- 2. 工具：解析 dtype / shape ----

def choose_dtype(p_spec: Dict[str, Any], fdp: atheris.FuzzedDataProvider):
    dtypes = p_spec.get("dtype_choices", ["float32"])
    dtype_name = fdp.PickValueInList(dtypes)
    # 简单映射；按需要扩展
    mapping = {
        "float32": torch.float32,
        "float": torch.float32,
        "float64": torch.float64,
        "double": torch.float64,
        "complex64": torch.complex64,
        "complex128": torch.complex128,
        "int64": torch.int64,
        "long": torch.int64,
        "int32": torch.int32,
        "int": torch.int32,
        "bool": torch.bool,
    }
    return mapping.get(dtype_name, torch.float32)


def resolve_shape(shape_spec: List[Any], shape_vars: Dict[str, int]) -> Tuple[int, ...]:
    """
    shape_spec 可以是形如 ["N", "C_in", 3, "H"] 的 list。
    如果是字符串，就从 shape_vars 取值；如果是整数，就直接用。
    """
    dims = []
    for dim in shape_spec:
        if isinstance(dim, str):
            if dim not in shape_vars:
                raise KeyError(f"shape var {dim} not in shape_vars")
            dims.append(int(shape_vars[dim]))
        else:
            dims.append(int(dim))
    return tuple(dims)


# ---- 3. 各种 kind 对应的采样逻辑 ----

def sample_int(p_spec: Dict[str, Any], fdp: atheris.FuzzedDataProvider) -> int:
    if "values" in p_spec:
        return fdp.PickValueInList(list(p_spec["values"]))
    lo, hi = p_spec.get("range", [0, 10])
    return fdp.ConsumeIntInRange(int(lo), int(hi))

def sample_float(p_spec: Dict[str, Any], fdp: atheris.FuzzedDataProvider) -> float:
    """
    支持两种方式：
    - values: [0.1, 0.5, 1.0]
    - range: [-2.0, 2.0]
    """
    if "values" in p_spec:
        return float(fdp.PickValueInList(list(p_spec["values"])))
    lo, hi = p_spec.get("range", [-1.0, 1.0])
    lo = float(lo)
    hi = float(hi)
    # 用离散网格 + ConsumeIntInRange 生成一个 [lo, hi] 里的浮点数
    steps = int(p_spec.get("steps", 1000))  # 可配置精度
    idx = fdp.ConsumeIntInRange(0, steps)
    return lo + (hi - lo) * (idx / steps)

def sample_float_optional(p_spec: Dict[str, Any], fdp: atheris.FuzzedDataProvider):
    """
    可选 float：50% 概率返回 None。
    """
    if fdp.ConsumeBool():
        return None
    return sample_float(p_spec, fdp)

def sample_int_optional(p_spec: Dict[str, Any], fdp: atheris.FuzzedDataProvider):
    if fdp.ConsumeBool():
        return None
    return sample_int(p_spec, fdp)


def sample_bool(p_spec: Dict[str, Any], fdp: atheris.FuzzedDataProvider) -> bool:
    return fdp.ConsumeBool()


def sample_enum(p_spec: Dict[str, Any], fdp: atheris.FuzzedDataProvider):
    values = p_spec["values"]
    return fdp.PickValueInList(list(values))


def sample_enum_optional(p_spec: Dict[str, Any], fdp: atheris.FuzzedDataProvider):
    # 50% 机会返回 None
    if fdp.ConsumeBool():
        return None
    return sample_enum(p_spec, fdp)


def sample_int_or_tuple(p_spec: Dict[str, Any], fdp: atheris.FuzzedDataProvider):
    """
    可以返回 int 或 (int, int)。
    如果 p_spec 有 'values'，从中选；否则用 range。
    """
    as_tuple = fdp.ConsumeBool()
    if "values" in p_spec:
        v = fdp.PickValueInList(list(p_spec["values"]))
        if isinstance(v, list) or isinstance(v, tuple):
            return tuple(v)
        if as_tuple:
            return (int(v), int(v))
        return int(v)
    else:
        lo, hi = p_spec.get("range", [1, 4])
        if as_tuple:
            v1 = fdp.ConsumeIntInRange(int(lo), int(hi))
            v2 = fdp.ConsumeIntInRange(int(lo), int(hi))
            return (v1, v2)
        else:
            return fdp.ConsumeIntInRange(int(lo), int(hi))


def sample_int_list(p_spec: Dict[str, Any], fdp: atheris.FuzzedDataProvider):
    """
    生成一个长度可变的 int list。长度范围可以由 p_spec['len_range'] 控制。
    """
    len_lo, len_hi = p_spec.get("len_range", [1, 4])
    length = fdp.ConsumeIntInRange(int(len_lo), int(len_hi))
    lo, hi = p_spec.get("range", [0, 10])
    return [fdp.ConsumeIntInRange(int(lo), int(hi)) for _ in range(length)]


def sample_tensor(p_spec: Dict[str, Any], fdp: atheris.FuzzedDataProvider, shape_vars: Dict[str, int]):
    shape_spec = p_spec["shape_spec"]
    shape = resolve_shape(shape_spec, shape_vars)
    dtype = choose_dtype(p_spec, fdp)
    # 可以在这里根据 dtype 选择不同的分布（整数范围、浮点等）
    if dtype.is_floating_point or dtype.is_complex:
        return torch.randn(shape, dtype=dtype)
    elif dtype == torch.int64 or dtype == torch.int32:
        return torch.randint(low=-10, high=10, size=shape, dtype=dtype)
    elif dtype == torch.bool:
        return torch.rand(shape) > 0.5
    else:
        return torch.zeros(shape, dtype=dtype)


def sample_tensor_optional(p_spec: Dict[str, Any], fdp: atheris.FuzzedDataProvider, shape_vars: Dict[str, int]):
    if fdp.ConsumeBool():
        return None
    return sample_tensor(p_spec, fdp, shape_vars)


def sample_tensor_or_scalar(p_spec: Dict[str, Any], fdp: atheris.FuzzedDataProvider, shape_vars: Dict[str, int]):
    """
    有时 where / add 等 API 允许标量或张量。
    """
    as_scalar = fdp.ConsumeBool()
    if as_scalar:
        # 标量用 float/int 等
        if "dtype_choices" in p_spec and "int" in p_spec["dtype_choices"]:
            lo, hi = p_spec.get("scalar_range", [-10, 10])
            return fdp.ConsumeIntInRange(int(lo), int(hi))
        else:
            return fdp.ConsumeFloat()
    else:
        return sample_tensor(p_spec, fdp, shape_vars)


def sample_tensor_list(p_spec: Dict[str, Any], fdp: atheris.FuzzedDataProvider, shape_vars: Dict[str, int]):
    """
    Tensor[] 参数，常见于 cat/stack。
    简单起见：生成 1~len_hi 个形状一样的 tensor。
    """
    len_lo, len_hi = p_spec.get("len_range", [1, 4])
    length = fdp.ConsumeIntInRange(int(len_lo), int(len_hi))
    elems = []
    for _ in range(length):
        elems.append(sample_tensor(p_spec["elem"], fdp, shape_vars))
    return elems


# ---- 4. kind -> handler 映射 ----

KIND_HANDLERS = {
    "int": lambda p, f, s: sample_int(p, f),
    "int_optional": lambda p, f, s: sample_int_optional(p, f),
    "bool": lambda p, f, s: sample_bool(p, f),
    "enum": lambda p, f, s: sample_enum(p, f),
    "enum_optional": lambda p, f, s: sample_enum_optional(p, f),
    "int_or_tuple": lambda p, f, s: sample_int_or_tuple(p, f),
    "int_list": lambda p, f, s: sample_int_list(p, f),
    "tensor": lambda p, f, s: sample_tensor(p, f, s),
    "tensor_optional": lambda p, f, s: sample_tensor_optional(p, f, s),
    "tensor_or_scalar": lambda p, f, s: sample_tensor_or_scalar(p, f, s),
    "tensor_list": lambda p, f, s: sample_tensor_list(p, f, s),
    "float": lambda p, f, s: sample_float(p, f),
    "float_optional": lambda p, f, s: sample_float_optional(p, f),
}


def sample_param(kind: str, p_spec: Dict[str, Any], fdp: atheris.FuzzedDataProvider, shape_vars: Dict[str, int]):
    if kind not in KIND_HANDLERS:
        raise ValueError(f"Unknown param kind: {kind}")
    return KIND_HANDLERS[kind](p_spec, fdp, shape_vars)


# ---- 5. 顶层：对一个 API 生成 cfg ----

def gen_config_for_api(spec: Dict[str, Any], fdp: atheris.FuzzedDataProvider) -> Dict[str, Any]:
    """
    根据 API 的 spec + fuzzed data provider，生成一次完整参数配置 cfg。
    cfg 形如：
      {
        'input': Tensor(...),
        'weight': Tensor(...),
        'bias': Tensor or None,
        'stride': int or tuple,
        'padding': str or int,
        'groups': int,
        ...
        '_shape_vars': {...}   # 额外带出去
      }
    """
    cfg: Dict[str, Any] = {}

    # 1. 生成形状变量
    shape_vars = gen_shape_vars(spec, fdp)
    cfg["_shape_vars"] = shape_vars

    # 2. 遍历参数
    for name, p_spec in spec.get("params", {}).items():
        kind = p_spec["kind"]
        value = sample_param(kind, p_spec, fdp, shape_vars)
        cfg[name] = value

    return cfg

