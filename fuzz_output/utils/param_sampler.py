# param_sampler.py
import math
import random
from typing import Any, Dict, List, Tuple
import os

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
    p = _env_float("P_NONCONTIG", 0.0)
    if p <= 0.0:
        return t
    if fdp.ConsumeIntInRange(0, 999) >= int(p * 1000):
        return t
    if t.dim() >= 2:
        choice = fdp.ConsumeIntInRange(0, 2)
        try:
            if choice == 0:
                t = t.transpose(-1, -2)
            elif choice == 1 and t.dim() >= 3:
                perm = list(range(t.dim()))
                perm = perm[1:] + perm[:1]
                t = t.permute(*perm)
            else:
                last = t.size(-1)
                if last > 1:
                    start = fdp.ConsumeIntInRange(0, last - 1)
                    length = fdp.ConsumeIntInRange(1, last - start)
                    t = t.narrow(-1, start, length)
        except Exception:
            return t
    p_re = _env_float("P_RECONTIG", 0.0)
    if p_re > 0.0 and fdp.ConsumeIntInRange(0, 999) < int(p_re * 1000):
        try:
            t = t.contiguous()
        except Exception:
            pass
    return t

def maybe_zero_shape_var(name: str, val: int, fdp: atheris.FuzzedDataProvider) -> int:
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
    if any(k in n for k in ("h", "w", "d", "l", "len", "length", "seq", "time")):
        if fdp.ConsumeIntInRange(0, 999) < int(p * 1000):
            return 0
    return val


# ==========================================================
# NEW: Multi-rank support utilities
# ==========================================================

def get_rank_candidates(spec: Dict[str, Any]) -> List[int]:
    """Extract valid integer rank candidates from spec['rank_hints']."""
    rh = spec.get("rank_hints")
    if not isinstance(rh, dict):
        return []
    cands = rh.get("rank_candidates")
    if not isinstance(cands, list):
        return []
    return [int(x) for x in cands if isinstance(x, int)]


def pick_rank(spec: Dict[str, Any], fdp: atheris.FuzzedDataProvider) -> Optional[int]:
    """
    Pick a rank from rank_candidates using fdp.
    Returns None if no valid candidates.
    """
    candidates = get_rank_candidates(spec)
    if not candidates:
        return None
    return fdp.PickValueInList(candidates)


def get_shape_spec_for_param(
    p_spec: Dict[str, Any],
    active_rank: Optional[int],
) -> List[Any]:
    """
    Resolve the effective shape_spec for a param, considering shape_spec_by_rank.

    Priority:
      1) shape_spec_by_rank[str(active_rank)] if active_rank is set and key exists
      2) shape_spec (base fallback)
    """
    # Check shape_spec_by_rank first
    if active_rank is not None:
        sbr = p_spec.get("shape_spec_by_rank")
        if isinstance(sbr, dict):
            key = str(active_rank)
            if key in sbr:
                return list(sbr[key])

    # Fallback to base shape_spec
    base = p_spec.get("shape_spec")
    if isinstance(base, list):
        return list(base)

    return ["TODO_SHAPE"]


def collect_needed_shape_vars(spec: Dict[str, Any], active_rank: Optional[int]) -> set:
    """
    Scan all params' effective shape_specs to find which shape_var names are needed.
    This helps us know which vars to generate.
    """
    needed = set()
    for pname, p_spec in spec.get("params", {}).items():
        kind = p_spec.get("kind", "")
        if kind not in ("tensor", "tensor_optional"):
            continue
        shape_spec = get_shape_spec_for_param(p_spec, active_rank)
        for dim in shape_spec:
            if isinstance(dim, str) and dim != "TODO_SHAPE":
                needed.add(dim)
    return needed


# ---- 1. shape vars sampling ----

def gen_shape_vars(
    spec: Dict[str, Any],
    fdp: atheris.FuzzedDataProvider,
    active_rank: Optional[int] = None,
) -> Dict[str, int]:
    """
    Generate shape variables from spec['shape_vars'].
    Only generates vars that are actually needed for the active_rank's shape specs.
    """
    all_vars = spec.get("shape_vars", {})
    if not all_vars:
        return {}

    # Determine which vars are actually referenced
    needed = collect_needed_shape_vars(spec, active_rank)

    shape_vars = {}
    for name, rng in all_vars.items():
        if not isinstance(rng, (list, tuple)) or len(rng) != 2:
            continue
        lo, hi = int(rng[0]), int(rng[1])

        # If this var is not needed by any shape_spec, still generate it
        # (it might be used in constraints), but skip empty-dim logic
        val = fdp.ConsumeIntInRange(lo, hi)

        if name in needed:
            val = maybe_zero_shape_var(name, int(val), fdp)

        shape_vars[name] = int(val)

    return shape_vars


# ---- 2. dtype / shape utilities ----

def choose_dtype(p_spec: Dict[str, Any], fdp: atheris.FuzzedDataProvider):
    dtypes = p_spec.get("dtype_choices", ["float32"])
    dtype_name = fdp.PickValueInList(dtypes)
    mapping = {
        "float32": torch.float32, "float": torch.float32,
        "float64": torch.float64, "double": torch.float64,
        "float16": torch.float16, "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "complex64": torch.complex64, "complex128": torch.complex128,
        "int64": torch.int64, "long": torch.int64,
        "int32": torch.int32, "int": torch.int32,
        "int16": torch.int16, "short": torch.int16,
        "int8": torch.int8,
        "uint8": torch.uint8,
        "bool": torch.bool,
    }
    return mapping.get(dtype_name, torch.float32)


def resolve_shape(shape_spec: List[Any], shape_vars: Dict[str, int]) -> Tuple[int, ...]:
    """
    shape_spec items: strings -> lookup in shape_vars; ints -> use directly.
    Raises KeyError if a string var is not found AND is not TODO_SHAPE.
    """
    dims = []
    for dim in shape_spec:
        if isinstance(dim, str):
            if dim == "TODO_SHAPE":
                # Fallback: use a small random dim so we don't crash
                dims.append(random.randint(1, 8))
            elif dim not in shape_vars:
                raise KeyError(f"shape var {dim} not in shape_vars")
            else:
                dims.append(int(shape_vars[dim]))
        else:
            dims.append(int(dim))
    return tuple(dims)


# ---- 3. sampling by kind ----

def sample_int(p_spec: Dict[str, Any], fdp: atheris.FuzzedDataProvider) -> int:
    if "values" in p_spec:
        return fdp.PickValueInList(list(p_spec["values"]))
    lo, hi = p_spec.get("range", [0, 10])
    return fdp.ConsumeIntInRange(int(lo), int(hi))

def sample_float(p_spec: Dict[str, Any], fdp: atheris.FuzzedDataProvider) -> float:
    if "values" in p_spec:
        return float(fdp.PickValueInList(list(p_spec["values"])))
    lo, hi = p_spec.get("range", [-1.0, 1.0])
    lo, hi = float(lo), float(hi)
    steps = int(p_spec.get("steps", 1000))
    idx = fdp.ConsumeIntInRange(0, steps)
    return lo + (hi - lo) * (idx / steps)

def sample_float_optional(p_spec: Dict[str, Any], fdp: atheris.FuzzedDataProvider):
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
    values = p_spec.get("values", [""])
    # Filter out TODO markers
    valid = [v for v in values if not (isinstance(v, str) and v.startswith("__"))]
    if not valid:
        valid = values
    return fdp.PickValueInList(list(valid))

def sample_enum_optional(p_spec: Dict[str, Any], fdp: atheris.FuzzedDataProvider):
    if fdp.ConsumeBool():
        return None
    return sample_enum(p_spec, fdp)

def sample_int_or_tuple(p_spec: Dict[str, Any], fdp: atheris.FuzzedDataProvider):
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
    len_lo, len_hi = p_spec.get("len_range", [1, 4])
    length = fdp.ConsumeIntInRange(int(len_lo), int(len_hi))
    lo, hi = p_spec.get("range", [0, 10])
    return [fdp.ConsumeIntInRange(int(lo), int(hi)) for _ in range(length)]


def sample_tensor(
    p_spec: Dict[str, Any],
    fdp: atheris.FuzzedDataProvider,
    shape_vars: Dict[str, int],
    active_rank: Optional[int] = None,
):
    """
    Sample a tensor. Uses shape_spec_by_rank[active_rank] if available,
    otherwise falls back to shape_spec.
    """
    shape_spec = get_shape_spec_for_param(p_spec, active_rank)
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

    t = maybe_make_noncontig(t, fdp)
    return t


def sample_tensor_optional(
    p_spec: Dict[str, Any],
    fdp: atheris.FuzzedDataProvider,
    shape_vars: Dict[str, int],
    active_rank: Optional[int] = None,
):
    if fdp.ConsumeBool():
        return None
    return sample_tensor(p_spec, fdp, shape_vars, active_rank)


def sample_tensor_or_scalar(
    p_spec: Dict[str, Any],
    fdp: atheris.FuzzedDataProvider,
    shape_vars: Dict[str, int],
    active_rank: Optional[int] = None,
):
    as_scalar = fdp.ConsumeBool()
    if as_scalar:
        if "dtype_choices" in p_spec and "int" in p_spec["dtype_choices"]:
            lo, hi = p_spec.get("scalar_range", [-10, 10])
            return fdp.ConsumeIntInRange(int(lo), int(hi))
        else:
            return fdp.ConsumeFloat()
    else:
        return sample_tensor(p_spec, fdp, shape_vars, active_rank)


def sample_tensor_list(
    p_spec: Dict[str, Any],
    fdp: atheris.FuzzedDataProvider,
    shape_vars: Dict[str, int],
    active_rank: Optional[int] = None,
):
    len_lo, len_hi = p_spec.get("len_range", [1, 4])
    length = fdp.ConsumeIntInRange(int(len_lo), int(len_hi))
    elems = []
    for _ in range(length):
        elems.append(sample_tensor(p_spec["elem"], fdp, shape_vars, active_rank))
    return elems


# ---- 4. kind -> handler mapping ----
# NOTE: handlers now accept active_rank as 4th positional arg

KIND_HANDLERS = {
    "int":              lambda p, f, s, r: sample_int(p, f),
    "int_optional":     lambda p, f, s, r: sample_int_optional(p, f),
    "bool":             lambda p, f, s, r: sample_bool(p, f),
    "enum":             lambda p, f, s, r: sample_enum(p, f),
    "enum_optional":    lambda p, f, s, r: sample_enum_optional(p, f),
    "int_or_tuple":     lambda p, f, s, r: sample_int_or_tuple(p, f),
    "int_list":         lambda p, f, s, r: sample_int_list(p, f),
    "tensor":           lambda p, f, s, r: sample_tensor(p, f, s, r),
    "tensor_optional":  lambda p, f, s, r: sample_tensor_optional(p, f, s, r),
    "tensor_or_scalar": lambda p, f, s, r: sample_tensor_or_scalar(p, f, s, r),
    "tensor_list":      lambda p, f, s, r: sample_tensor_list(p, f, s, r),
    "float":            lambda p, f, s, r: sample_float(p, f),
    "float_optional":   lambda p, f, s, r: sample_float_optional(p, f),
}


def sample_param(
    kind: str,
    p_spec: Dict[str, Any],
    fdp: atheris.FuzzedDataProvider,
    shape_vars: Dict[str, int],
    active_rank: Optional[int] = None,
):
    if kind not in KIND_HANDLERS:
        raise ValueError(f"Unknown param kind: {kind}")
    return KIND_HANDLERS[kind](p_spec, fdp, shape_vars, active_rank)


# ---- 5. top-level: gen_config_for_api ----

def gen_config_for_api(
    spec: Dict[str, Any],
    fdp: atheris.FuzzedDataProvider,
) -> Dict[str, Any]:
    """
    Generate a complete parameter config for one API call.

    Multi-rank support:
      1) Pick a rank from rank_hints.rank_candidates
      2) Store it as cfg['_active_rank']
      3) Use shape_spec_by_rank[rank] for tensor params that have it
      4) Also merge per-rank constraints_by_rank into cfg['_rank_constraints']
         so the harness constraint_func can check them
    """
    cfg: Dict[str, Any] = {}

    # 1. Pick a rank
    active_rank = pick_rank(spec, fdp)
    cfg["_active_rank"] = active_rank

    # 2. Generate shape variables
    shape_vars = gen_shape_vars(spec, fdp, active_rank=active_rank)
    cfg["_shape_vars"] = shape_vars

    # 3. Collect per-rank constraints (if any)
    rank_constraints: List[str] = []
    if active_rank is not None:
        for pname, p_spec in spec.get("params", {}).items():
            cbr = p_spec.get("constraints_by_rank")
            if isinstance(cbr, dict):
                key = str(active_rank)
                if key in cbr and isinstance(cbr[key], list):
                    rank_constraints.extend(cbr[key])
    cfg["_rank_constraints"] = rank_constraints

    # 4. Sample each parameter
    for name, p_spec in spec.get("params", {}).items():
        kind = p_spec["kind"]
        value = sample_param(kind, p_spec, fdp, shape_vars, active_rank)
        cfg[name] = value

    return cfg

# =========================
# Mutation (FreeFuzz-style)
# =========================

def _pick_other_in_list(fdp: atheris.FuzzedDataProvider, vals, cur):
    vals = list(vals)
    if len(vals) <= 1:
        return cur
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
        a = int(x[0]) if len(x) > 0 else 1
        b = int(x[1]) if len(x) > 1 else a
        return (a, b)
    return (int(x), int(x))

def _boundary_ints(lo: int, hi: int):
    cands = [lo, hi, 0, 1, -1, 2, 3, 4, 8, 16, 32, 64, 128]
    return cands

def _mutate_int_value(p_spec, fdp, cur: int) -> int:
    if "values" in p_spec and p_spec["values"]:
        cand = _pick_other_in_list(fdp, p_spec["values"], cur)
        if isinstance(cand, (list, tuple)):
            return int(cand[0])
        return int(cand)
    lo, hi = p_spec.get("range", [0, 10])
    delta = fdp.ConsumeIntInRange(-3, 3)
    v = int(cur) + int(delta)
    return max(int(lo), min(int(hi), v))


def _mutate_float_value(p_spec, fdp, cur: float) -> float:
    specials = [0.0, 1.0, -1.0, 1e-12, 1e-6, 1e-3, 1e3]
    if fdp.ConsumeBool():
        v = float(fdp.PickValueInList(specials))
        return v if v != cur else -v
    delta = (fdp.ConsumeIntInRange(-1000, 1000) / 1000.0)
    return float(cur + delta)

def _mutate_enum_value(p_spec, fdp, cur):
    values = p_spec.get("values", [cur])
    valid = [v for v in values if not (isinstance(v, str) and v.startswith("__"))]
    if not valid:
        valid = values
    return _pick_other_in_list(fdp, valid, cur)

def _mutate_tensor_value_like(t: torch.Tensor, fdp: atheris.FuzzedDataProvider) -> torch.Tensor:
    dtype = t.dtype
    shape = tuple(t.shape)

    if dtype.is_floating_point:
        out = torch.randn(shape, dtype=dtype)
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
    """Build shape_var -> list of dependent tensor param names."""
    deps = {k: [] for k in spec.get("shape_vars", {}).keys()}
    for pname, p_spec in spec.get("params", {}).items():
        kind = p_spec.get("kind", "")
        if kind in ("tensor", "tensor_optional"):
            # Check both shape_spec and all shape_spec_by_rank entries
            all_specs = []
            base_ss = p_spec.get("shape_spec", [])
            if isinstance(base_ss, list):
                all_specs.append(base_ss)
            sbr = p_spec.get("shape_spec_by_rank")
            if isinstance(sbr, dict):
                for rk, ss in sbr.items():
                    if isinstance(ss, list):
                        all_specs.append(ss)

            for ss in all_specs:
                for dim in ss:
                    if isinstance(dim, str) and dim in deps:
                        if pname not in deps[dim]:
                            deps[dim].append(pname)

        if kind == "tensor_list":
            elem = p_spec.get("elem", {})
            for dim in elem.get("shape_spec", []) or []:
                if isinstance(dim, str) and dim in deps:
                    if pname not in deps[dim]:
                        deps[dim].append(pname)
    return deps


def _resample_param_value(spec, pname, cfg, fdp):
    p_spec = spec["params"][pname]
    kind = p_spec["kind"]
    shape_vars = cfg.get("_shape_vars", {})
    active_rank = cfg.get("_active_rank")
    return sample_param(kind, p_spec, fdp, shape_vars, active_rank)


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
    FreeFuzz-style multi-step mutation with constraint checking.
    """
    if cfg is None:
        return None

    params = list(spec.get("params", {}).keys())
    if not params:
        return cfg

    deps = _deps_for_shape_vars(spec)
    cur = cfg
    steps = max(1, int(steps))

    for _ in range(steps):
        ok = False
        for _try in range(max_attempts_per_step):
            trial = deepcopy(cur)

            # 1) Shape var mutation
            if spec.get("shape_vars") and (fdp.ConsumeIntInRange(0, 999) < int(p_shape_mut * 1000)):
                sv = trial.get("_shape_vars", {})
                if isinstance(sv, dict) and len(sv) > 0:
                    var = fdp.PickValueInList(list(sv.keys()))
                    rng = spec["shape_vars"].get(var)
                    if isinstance(rng, (list, tuple)) and len(rng) == 2:
                        lo, hi = int(rng[0]), int(rng[1])
                        oldv = int(sv.get(var, lo))
                        newv = fdp.ConsumeIntInRange(lo, hi)
                        if newv == oldv and hi > lo:
                            newv = lo if oldv != lo else hi
                        sv[var] = int(newv)

                        for pname in deps.get(var, []):
                            pk = spec["params"][pname]["kind"]
                            if pk == "tensor_optional" and trial.get(pname) is None:
                                continue
                            trial[pname] = _resample_param_value(spec, pname, trial, fdp)

            # 2) Pick a param for type/value mutation
            pname = fdp.PickValueInList(params)
            p_spec = spec["params"][pname]
            kind = p_spec["kind"]
            val = trial.get(pname, None)

            do_type = (fdp.ConsumeIntInRange(0, 999) < int(p_type_mut * 1000))

            if do_type:
                if kind == "tensor_optional":
                    if val is None:
                        trial[pname] = _resample_param_value(spec, pname, trial, fdp)
                    else:
                        trial[pname] = None
                elif kind == "int_optional":
                    trial[pname] = None if val is not None else _resample_param_value(spec, pname, trial, fdp)
                elif kind == "enum_optional":
                    trial[pname] = None if val is not None else _resample_param_value(spec, pname, trial, fdp)
                elif kind == "int_or_tuple":
                    if isinstance(val, (tuple, list)):
                        trial[pname] = int(_as_2tuple(val)[0])
                    else:
                        v = int(val) if val is not None else 1
                        trial[pname] = (v, v)
                elif kind == "tensor":
                    if isinstance(val, torch.Tensor):
                        old_dtype = val.dtype
                        dts = p_spec.get("dtype_choices", ["float32"])
                        for _k in range(4):
                            nd = choose_dtype(p_spec, fdp)
                            if nd != old_dtype:
                                trial[pname] = _mutate_tensor_value_like(val.to(nd), fdp)
                                break
                        else:
                            trial[pname] = _mutate_tensor_value_like(val, fdp)
                elif kind == "tensor_or_scalar":
                    if isinstance(val, torch.Tensor):
                        trial[pname] = float(fdp.ConsumeIntInRange(-10, 10))
                    else:
                        trial[pname] = _resample_param_value(spec, pname, trial, fdp)
            else:
                if kind == "int":
                    trial[pname] = _mutate_int_value(p_spec, fdp, int(val) if val is not None else 0)
                elif kind == "float":
                    trial[pname] = _mutate_float_value(p_spec, fdp, float(val) if val is not None else 0.0)
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
                        trial[pname] = _mutate_int_value(p_spec, fdp, int(val) if val is not None else 1)
                elif kind == "int_list":
                    lst = list(val) if isinstance(val, list) else []
                    if not lst:
                        trial[pname] = _resample_param_value(spec, pname, trial, fdp)
                    else:
                        idx = fdp.ConsumeIntInRange(0, len(lst) - 1)
                        lst[idx] = _mutate_int_value(p_spec, fdp, int(lst[idx]))
                        trial[pname] = lst
                elif kind in ("tensor", "tensor_optional"):
                    if val is not None and isinstance(val, torch.Tensor):
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
                        if isinstance(val, int):
                            trial[pname] = int(val) ^ 1
                        else:
                            trial[pname] = float(val or 0) + (fdp.ConsumeIntInRange(-1000, 1000) / 1000.0)

            if constraint_func is None or constraint_func(trial):
                cur = trial
                ok = True
                break

        if not ok:
            continue

    return cur
