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
SPEC = {'api_name': 'torch.nn.functional.group_norm',
 'category': 'group_norm',
 'rank_hints': {'marker': '__RANK_FROM_DOC__',
                'status': 'unassigned',
                'rank_candidates': ['__RANK_TODO__'],
                'rank_any': False,
                'rank_min': None,
                'rank_max': None},
 'aten': {'aten_name': 'group_norm',
          'overload': 'default',
          'schema_str': 'aten::group_norm(Tensor input, int num_groups, '
                        'Tensor? weight=None, Tensor? bias=None, float '
                        'eps=1.0000000000000001e-05, bool cudnn_enabled=True) '
                        '-> Tensor',
          'source': 'aten'},
 'shape_vars': {'B': [1, 8], 'C': [1, 8]},
 'params': {'input': {'kind': 'tensor',
                      'dtype_choices': ['float32', 'float64'],
                      'shape_spec': ['TODO_SHAPE']},
            'num_groups': {'kind': 'int', 'range': [-1, 8]},
            'weight': {'kind': 'tensor_optional',
                       'dtype_choices': ['float32', 'float64'],
                       'shape_spec': ['C'],
                       'has_default': True,
                       'default_repr': 'None',
                       'shape_spec_by_rank': {'1': ['C']}},
            'bias': {'kind': 'tensor_optional',
                     'dtype_choices': ['float32', 'float64'],
                     'shape_spec': ['C'],
                     'has_default': True,
                     'default_repr': 'None'},
            'eps': {'kind': 'float',
                    'range': [1e-12, 0.1],
                    'default': 1e-05,
                    'has_default': True,
                    'default_repr': '1e-05'},
            'cudnn_enabled': {'kind': 'bool',
                              'default': True,
                              'has_default': True,
                              'default_repr': 'True'}}}

# Global constraints from YAML
CONSTRAINTS = ['len(input.shape) >= 2',
 'input.shape[0] == B',
 'input.shape[1] == C',
 'num_groups > 0',
 'C % num_groups == 0']


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
    """
    Check constraints for a given cfg.

    Checks TWO sources:
      1) CONSTRAINTS: global constraints from YAML top-level
      2) cfg['_rank_constraints']: per-rank constraints injected by param_sampler
         (from params[x].constraints_by_rank[active_rank])
    """
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
