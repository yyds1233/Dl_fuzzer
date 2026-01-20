# param_sampler.py
import math
import random
from typing import Any, Dict, List, Tuple

import atheris
import torch
from copy import deepcopy
from typing import Callable, Optional

# =========================
# Helpers: controlled diversity buckets
# =========================

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name, "")
    if v == "":
        return default
    return v.lower() in ("1", "true", "yes", "y", "on")

def maybe_make_noncontig(t: torch.Tensor, fdp: atheris.FuzzedDataProvider) -> torch.Tensor:
    """
    Small probability to change layout/stride (non-contiguous) to hit slow-paths.
    Env:
      P_NONCONTIG=0.05  (default 0 -> disabled)
      P_RECONTIG=0.10   (within noncontig branch, chance to call contiguous() as control)
    """
    p = _env_float("P_NONCONTIG", 0.0)
    if p <= 0.0:
        return t

    # Use fdp to keep determinism per input
    if fdp.ConsumeIntInRange(0, 999) >= int(p * 1000):
        return t

    # Only meaningful for ndim>=2
    if t.dim() >= 2:
        # 0: transpose last 2 dims, 1: permute, 2: slice/narrow
        choice = fdp.ConsumeIntInRange(0, 2)
        try:
            if choice == 0:
                t = t.transpose(-1, -2)
            elif choice == 1 and t.dim() >= 3:
                # simple permute: rotate dims
                perm = list(range(t.dim()))
                perm = perm[1:] + perm[:1]
                t = t.permute(*perm)
            else:
                # narrow along last dim if possible
                last = t.size(-1)
                if last > 1:
                    start = fdp.ConsumeIntInRange(0, last - 1)
                    length = fdp.ConsumeIntInRange(1, last - start)
                    t = t.narrow(-1, start, length)
        except Exception:
            # if anything goes wrong, fall back to original
            return t

    # optional: small chance to return contiguous() as control group
    p_re = _env_float("P_RECONTIG", 0.0)
    if p_re > 0.0 and fdp.ConsumeIntInRange(0, 999) < int(p_re * 1000):
        try:
            t = t.contiguous()
        except Exception:
            pass

    return t

def maybe_zero_shape_var(name: str, val: int, fdp: atheris.FuzzedDataProvider) -> int:
    """
    Controlled empty-dim bucket (0-length dimension).
    Env:
      ALLOW_EMPTY=1         enable
      P_EMPTY_DIM=0.01      probability to zero eligible dims
      P_EMPTY_NC=0.001      probability to zero N/C (default 0 -> disabled)

    Strategy:
      - allow zero mostly for H/W/D/L/len/length/seq-style vars
      - keep N/C very rare (or disabled)
    """
    if not _env_bool("ALLOW_EMPTY", False):
        return val

    p = _env_float("P_EMPTY_DIM", 0.0)
    if p <= 0.0:
        return val

    n = name.lower()
    is_nc = (n in ("n", "c")) or n.startswith("n_") or n.startswith("c_")
    if is_nc:
        p_nc = _env_float("P_EMPTY_NC", 0.0)
        if p_nc <= 0.0:
            return val
        if fdp.ConsumeIntInRange(0, 999) < int(p_nc * 1000):
            return 0
        return val

    # whitelist: spatial/temporal/length-like dims
    if any(k in n for k in ("h", "w", "d", "l", "len", "length", "seq", "time")):
        if fdp.ConsumeIntInRange(0, 999) < int(p * 1000):
            return 0

    return val


# ---- 1. shape vars 采样 ----

def gen_shape_vars(spec: Dict[str, Any], fdp: atheris.FuzzedDataProvider) -> Dict[str, int]:
    """
    根据 spec['shape_vars'] 生成隐含的形状变量。
    shape_vars: { "N": [min, max], "C": [min, max], ... }
    """
    shape_vars = {}
    for name, (lo, hi) in spec.get("shape_vars", {}).items():
        val = fdp.ConsumeIntInRange(int(lo), int(hi))
        # D) empty-dim bucket (controlled)
        val = maybe_zero_shape_var(name, int(val), fdp)
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

    if dtype.is_floating_point or dtype.is_complex:
        t = torch.randn(shape, dtype=dtype)
    elif dtype == torch.int64 or dtype == torch.int32:
        t = torch.randint(low=-10, high=10, size=shape, dtype=dtype)
    elif dtype == torch.bool:
        t = (torch.rand(shape) > 0.5)
    else:
        t = torch.zeros(shape, dtype=dtype)

    # C) non-contig/stride bucket (controlled)
    t = maybe_make_noncontig(t, fdp)
    return t


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

# =========================
# Mutation (FreeFuzz-style)
# =========================

def _pick_other_in_list(fdp: atheris.FuzzedDataProvider, vals, cur):
    vals = list(vals)
    if len(vals) <= 1:
        return cur
    # 尽量选一个不同的
    for _ in range(4):
        v = fdp.PickValueInList(vals)
        if v != cur:
            return v
    return fdp.PickValueInList(vals)

def _as_2tuple(x):
    if isinstance(x, (list, tuple)):
        if len(x) == 2:
            return (int(x[0]), int(x[1]))
        if len(x) == 1:
            return (int(x[0]), int(x[0]))
        # 其他长度，截断/填充
        a = int(x[0]) if len(x) > 0 else 1
        b = int(x[1]) if len(x) > 1 else a
        return (a, b)
    return (int(x), int(x))

def _boundary_ints(lo: int, hi: int):
    # 常用边界值 + 一些“容易出错”的值
    cands = [lo, hi, 0, 1, -1, 2, 3, 4, 8, 16, 32, 64, 128]
    # 只保留在范围附近的（不严格限制，给异常路径留点空间）
    return cands

# def _mutate_int_value(p_spec, fdp, cur: int) -> int:
#     if "values" in p_spec:
#         return int(_pick_other_in_list(fdp, p_spec["values"], cur))
#     lo, hi = p_spec.get("range", [0, 10])
#     lo, hi = int(lo), int(hi)
#     # 50% 从边界候选里取，50% 随机
#     if fdp.ConsumeBool():
#         cand = fdp.PickValueInList(_boundary_ints(lo, hi))
#         # 尽量别老返回原值
#         if cand == cur:
#             cand = fdp.ConsumeIntInRange(lo, hi)
#         return int(cand)
#     else:
#         v = fdp.ConsumeIntInRange(lo, hi)
#         if v == cur and hi > lo:
#             v = lo if cur != lo else hi
#         return int(v)
def _mutate_int_value(p_spec, fdp, cur: int) -> int:
    # values 优先：pick 另一个值
    if "values" in p_spec and p_spec["values"]:
        cand = _pick_other_in_list(fdp, p_spec["values"], cur)
        # 关键修复：values 里可能有 list/tuple（例如 [1,1]）
        if isinstance(cand, (list, tuple)):
            # 方案1：直接返回一个 tuple/list（交给上层约束或调用报错处理）
            return int(cand[0])  # type: ignore
            # 方案2（可选）：return int(cand[0]) if cand else cur
        return int(cand)
    lo, hi = p_spec.get("range", [0, 10])
    # 小扰动
    delta = fdp.ConsumeIntInRange(-3, 3)
    v = int(cur) + int(delta)
    return max(int(lo), min(int(hi), v))


def _mutate_float_value(p_spec, fdp, cur: float) -> float:
    # 边界/特殊值
    specials = [0.0, 1.0, -1.0, 1e-12, 1e-6, 1e-3, 1e3]
    if fdp.ConsumeBool():
        v = float(fdp.PickValueInList(specials))
        return v if v != cur else -v
    # 轻微扰动
    delta = (fdp.ConsumeIntInRange(-1000, 1000) / 1000.0)
    return float(cur + delta)

def _mutate_enum_value(p_spec, fdp, cur):
    return _pick_other_in_list(fdp, p_spec["values"], cur)

def _mutate_tensor_value_like(t: torch.Tensor, fdp: atheris.FuzzedDataProvider) -> torch.Tensor:
    # 内容变异：保持 shape & dtype
    dtype = t.dtype
    shape = tuple(t.shape)

    if dtype.is_floating_point:
        out = torch.randn(shape, dtype=dtype)
        # 小概率注入 NaN/Inf
        if out.numel() > 0 and fdp.ConsumeIntInRange(0, 99) == 0:
            out.view(-1)[0] = float("nan")
        if out.numel() > 0 and fdp.ConsumeIntInRange(0, 99) == 0:
            out.view(-1)[-1] = float("inf")
        out = maybe_make_noncontig(out, fdp)
        return out

    if dtype.is_complex:
        out = torch.randn(shape, dtype=dtype)
        if out.numel() > 0 and fdp.ConsumeIntInRange(0, 199) == 0:
            out.view(-1)[0] = complex(float("nan"), 0.0)
        out = maybe_make_noncontig(out, fdp)
        return out

    if dtype in (torch.int32, torch.int64):
        out = torch.randint(low=-10, high=10, size=shape, dtype=dtype)
        out = maybe_make_noncontig(out, fdp)
        return out

    if dtype == torch.bool:
        out = (torch.rand(shape) > 0.5)
        out = maybe_make_noncontig(out, fdp)
        return out

    out = torch.zeros(shape, dtype=dtype)
    out = maybe_make_noncontig(out, fdp)
    return out


def _deps_for_shape_vars(spec):
    """
    建立 shape_var -> 依赖它的参数名列表 的映射，用于 shape var 变异后重采样 tensor。
    """
    deps = {k: [] for k in spec.get("shape_vars", {}).keys()}
    for pname, p_spec in spec.get("params", {}).items():
        kind = p_spec.get("kind", "")
        # tensor / tensor_optional
        if kind in ("tensor", "tensor_optional"):
            for dim in p_spec.get("shape_spec", []) or []:
                if isinstance(dim, str) and dim in deps:
                    deps[dim].append(pname)
        # tensor_list elem
        if kind == "tensor_list":
            elem = p_spec.get("elem", {})
            for dim in elem.get("shape_spec", []) or []:
                if isinstance(dim, str) and dim in deps:
                    deps[dim].append(pname)
    return deps

# def _repair_conv_like(spec, cfg, fdp):
#     """
#     启发式修复：如果发现 conv2d 常见变量/参数，则尽量修复 groups 与 C_in/C_out 的整除关系，
#     并设置 C_per_group = C_in // groups。
#     对其它 API 没有影响（识别不到就直接返回）。
#     """
#     sv = cfg.get("_shape_vars", {})
#     if not isinstance(sv, dict):
#         return

#     if "C_in" not in sv or "groups" not in cfg:
#         return

#     Cin = int(sv.get("C_in", 1))
#     Cout = int(sv.get("C_out", Cin))
#     g = int(cfg.get("groups", 1))

#     # 找到 Cin 与 Cout 的公因子集合（作为合法 groups）
#     import math
#     gg = math.gcd(Cin, Cout)
#     divisors = []
#     for d in range(1, gg + 1):
#         if gg % d == 0:
#             divisors.append(d)

#     if len(divisors) == 0:
#         cfg["groups"] = 1
#         if "C_per_group" in sv:
#             sv["C_per_group"] = Cin
#         return

#     # groups 还要满足 spec 中 range（如果有）
#     gspec = spec.get("params", {}).get("groups", {})
#     lo, hi = gspec.get("range", [1, max(divisors)])
#     lo, hi = int(lo), int(hi)
#     legal = [d for d in divisors if lo <= d <= hi]
#     if not legal:
#         legal = [1]

#     # 选一个（尽量不同于当前）
#     new_g = g
#     if new_g not in legal or fdp.ConsumeBool():
#         new_g = int(_pick_other_in_list(fdp, legal, g))
#     cfg["groups"] = new_g

#     # 修复 C_per_group
#     if "C_per_group" in sv:
#         sv["C_per_group"] = int(Cin // new_g)

def _resample_param_value(spec, pname, cfg, fdp):
    """
    按 spec 的 kind 重采样一个参数（会使用当前 cfg['_shape_vars']）
    """
    p_spec = spec["params"][pname]
    kind = p_spec["kind"]
    shape_vars = cfg.get("_shape_vars", {})
    return sample_param(kind, p_spec, fdp, shape_vars)

def mutate_cfg(
    spec: Dict[str, Any],
    cfg: Dict[str, Any],
    fdp: atheris.FuzzedDataProvider,
    constraint_func: Optional[Callable[[Dict[str, Any]], bool]] = None,
    steps: int = 2,
    max_attempts_per_step: int = 6,
    p_type_mut: float = 0.35,
    p_shape_mut: float = 0.10,
):
    """
    FreeFuzz 风格：对 seed cfg 做多步变异（type/value），变异后如果约束不满足就回滚/重试。
    - constraint_func: 由 harness 传入，用于判定 cfg 是否可执行
    """
    if cfg is None:
        return None

    params = list(spec.get("params", {}).keys())
    if not params:
        return cfg

    deps = _deps_for_shape_vars(spec)
    # 为了避免大量 deepcopy tensor 带来的开销，尽量按步复制并回滚
    cur = cfg

    # steps 至少 1，且不超过参数数的 2 倍（可自己调）
    steps = max(1, int(steps))

    for _ in range(steps):
        ok = False
        for _try in range(max_attempts_per_step):
            trial = deepcopy(cur)

            # 1) 低概率做 shape var 变异（会影响多个 tensor）
            if spec.get("shape_vars") and (fdp.ConsumeIntInRange(0, 999) < int(p_shape_mut * 1000)):
                sv = trial.get("_shape_vars", {})
                if isinstance(sv, dict) and len(sv) > 0:
                    var = fdp.PickValueInList(list(sv.keys()))
                    lo, hi = spec["shape_vars"][var]
                    lo, hi = int(lo), int(hi)
                    oldv = int(sv.get(var, lo))
                    newv = fdp.ConsumeIntInRange(lo, hi)
                    if newv == oldv and hi > lo:
                        newv = lo if oldv != lo else hi
                    sv[var] = int(newv)

                    # conv-like 修复（避免 groups/通道约束全失败）
                    # _repair_conv_like(spec, trial, fdp)

                    # 重采样依赖该 shape var 的 tensor 参数（保持可执行）
                    for pname in deps.get(var, []):
                        # 对 optional tensor：如果原来是 None，就维持 None（否则会改变语义太大）
                        pk = spec["params"][pname]["kind"]
                        if pk == "tensor_optional" and trial.get(pname) is None:
                            continue
                        trial[pname] = _resample_param_value(spec, pname, trial, fdp)

            # 2) 选一个参数做 type/value 变异
            pname = fdp.PickValueInList(params)
            p_spec = spec["params"][pname]
            kind = p_spec["kind"]
            val = trial.get(pname, None)

            do_type = (fdp.ConsumeIntInRange(0, 999) < int(p_type_mut * 1000))

            # ---- type mutation ----
            if do_type:
                if kind == "tensor_optional":
                    # None <-> Tensor
                    if val is None:
                        trial[pname] = _resample_param_value(spec, pname, trial, fdp)
                    else:
                        trial[pname] = None

                elif kind == "int_optional":
                    trial[pname] = None if val is not None else _resample_param_value(spec, pname, trial, fdp)

                elif kind == "enum_optional":
                    trial[pname] = None if val is not None else _resample_param_value(spec, pname, trial, fdp)

                elif kind == "int_or_tuple":
                    # int <-> (int,int)
                    if isinstance(val, (tuple, list)):
                        # 变成 int（取第一个）
                        trial[pname] = int(_as_2tuple(val)[0])
                    else:
                        v = int(val) if val is not None else 1
                        trial[pname] = (v, v)

                elif kind == "tensor":
                    # dtype mutation：重新采样 dtype（shape 不变）
                    if isinstance(val, torch.Tensor):
                        old_dtype = val.dtype
                        # 选一个不同 dtype
                        dts = p_spec.get("dtype_choices", ["float32"])
                        # 让 choose_dtype 用 fdp 选一个不同的
                        for _k in range(4):
                            nd = choose_dtype(p_spec, fdp)
                            if nd != old_dtype:
                                trial[pname] = _mutate_tensor_value_like(val.to(nd), fdp)
                                break
                        else:
                            # 选不到就做内容变异
                            trial[pname] = _mutate_tensor_value_like(val, fdp)

                elif kind == "tensor_or_scalar":
                    # scalar <-> tensor
                    if isinstance(val, torch.Tensor):
                        # 转 scalar
                        trial[pname] = float(fdp.ConsumeIntInRange(-10, 10))
                    else:
                        # 转 tensor
                        trial[pname] = _resample_param_value(spec, pname, trial, fdp)

                # 其他 kind：type mutation 先不强行做

            # ---- value mutation ----
            else:
                if kind == "int":
                    trial[pname] = _mutate_int_value(p_spec, fdp, int(val))

                elif kind == "float":
                    trial[pname] = _mutate_float_value(p_spec, fdp, float(val))

                elif kind == "bool":
                    trial[pname] = (not bool(val))

                elif kind == "enum":
                    trial[pname] = _mutate_enum_value(p_spec, fdp, val)

                elif kind == "int_or_tuple":
                    if isinstance(val, (tuple, list)):
                        a, b = _as_2tuple(val)
                        a2 = _mutate_int_value({"range": p_spec.get("range", [1, 4])}, fdp, a)
                        b2 = _mutate_int_value({"range": p_spec.get("range", [1, 4])}, fdp, b)
                        trial[pname] = (a2, b2)
                    else:
                        trial[pname] = _mutate_int_value(p_spec, fdp, int(val))

                elif kind == "int_list":
                    lst = list(val) if isinstance(val, list) else []
                    if not lst:
                        trial[pname] = _resample_param_value(spec, pname, trial, fdp)
                    else:
                        idx = fdp.ConsumeIntInRange(0, len(lst) - 1)
                        lst[idx] = _mutate_int_value(p_spec, fdp, int(lst[idx]))
                        trial[pname] = lst

                elif kind in ("tensor", "tensor_optional"):
                    if val is None:
                        # None 情况下做一次“生成 tensor”的值变异也行，但会改变语义；这里保守不做
                        pass
                    elif isinstance(val, torch.Tensor):
                        trial[pname] = _mutate_tensor_value_like(val, fdp)

                elif kind == "tensor_list":
                    lst = val if isinstance(val, list) else []
                    if not lst:
                        trial[pname] = _resample_param_value(spec, pname, trial, fdp)
                    else:
                        idx = fdp.ConsumeIntInRange(0, len(lst) - 1)
                        if isinstance(lst[idx], torch.Tensor):
                            lst[idx] = _mutate_tensor_value_like(lst[idx], fdp)
                        trial[pname] = lst

                elif kind == "tensor_or_scalar":
                    if isinstance(val, torch.Tensor):
                        trial[pname] = _mutate_tensor_value_like(val, fdp)
                    else:
                        # scalar value mutate
                        if isinstance(val, int):
                            trial[pname] = int(val) ^ 1  # 简单扰动
                        else:
                            trial[pname] = float(val) + (fdp.ConsumeIntInRange(-1000, 1000) / 1000.0)

            # 3) 再做一次 conv-like repair（尤其当 groups / C_in 等被动过）
            # _repair_conv_like(spec, trial, fdp)

            # 4) 检查约束；不满足则回滚重试
            if constraint_func is None or constraint_func(trial):
                cur = trial
                ok = True
                break

        # 这一 step 没做成就保持原 cfg，不强行失败
        if not ok:
            continue

    return cur


