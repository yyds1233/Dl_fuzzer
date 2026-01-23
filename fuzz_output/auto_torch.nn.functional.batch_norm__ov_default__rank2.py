import os
import sys
import importlib
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
SPEC = {'api_name': 'torch.nn.functional.batch_norm',
 'category': 'batch_norm',
 'rank_hints': {'marker': '__RANK_FROM_DOC__',
                'status': 'unassigned',
                'rank_candidates': [2, 3, 4, 5],
                'rank_any': False,
                'rank_min': None,
                'rank_max': None},
 'aten': {'aten_name': 'batch_norm',
          'overload': 'default',
          'schema_str': 'aten::batch_norm(Tensor input, Tensor? weight, '
                        'Tensor? bias, Tensor? running_mean, Tensor? '
                        'running_var, bool training, float momentum, float '
                        'eps, bool cudnn_enabled) -> Tensor'},
 'shape_vars': {'N': [1, 8],
                'C': [1, 64],
                'L': [1, 128],
                'H': [1, 128],
                'W': [1, 128],
                'D': [1, 64]},
 'params': {'input': {'kind': 'tensor',
                      'dtype_choices': ['float32', 'float64'],
                      'shape_spec': ['N', 'C'],
                      'shape_spec_by_rank': {'2': ['N', 'C'],
                                             '3': ['N', 'C', 'L'],
                                             '4': ['N', 'C', 'H', 'W'],
                                             '5': ['N', 'C', 'D', 'H', 'W']}},
            'weight': {'kind': 'tensor_optional',
                       'dtype_choices': ['float32', 'float64'],
                       'shape_spec': ['C']},
            'bias': {'kind': 'tensor_optional',
                     'dtype_choices': ['float32', 'float64'],
                     'shape_spec': ['C']},
            'running_mean': {'kind': 'tensor_optional',
                             'dtype_choices': ['float32', 'float64'],
                             'shape_spec': ['C']},
            'running_var': {'kind': 'tensor_optional',
                            'dtype_choices': ['float32', 'float64'],
                            'shape_spec': ['C']},
            'training': {'kind': 'bool'},
            'momentum': {'kind': 'float', 'range': [-1.0, 1.0]},
            'eps': {'kind': 'float', 'range': [1e-12, 0.1]},
            'cudnn_enabled': {'kind': 'bool'}},
 'generator': {'stage': 'B-normalize', 'version': '2026-01-21-v1'}}

# 从 YAML 中读取的约束表达式（字符串）
CONSTRAINTS = ['input.ndim in (2, 3, 4, 5)',
 'weight is None or weight.shape[0] == input.shape[1]',
 'bias is None or bias.shape[0] == input.shape[1]',
 'running_mean is None or running_mean.shape[0] == input.shape[1]',
 'running_var is None or running_var.shape[0] == input.shape[1]']


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
    """通用约束检查函数（预条件）。

    - 把 cfg['_shape_vars'] 和 cfg 本身都摊平成局部变量
    - 自动构造 padding_tuple / stride_tuple / dilation_tuple
    - 每条 CONSTRAINTS 表达式用 eval(expr, {}, locs) 检查

    NOTE:
      - locs 注入 torch/math，方便你写 torch.isfinite / math.gcd 等约束
    """
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
    """多次尝试生成满足约束的 cfg。"""
    if max_tries is None:
        max_tries = SEED_TRIES
    for _ in range(max_tries):
        cfg = gen_config_for_api(spec, fdp)
        if constraint_func(cfg):
            return cfg
    return None


def _call_target_api(cfg):
    """根据 SPEC['api_name'] 和 SPEC['params'] 自动调用目标 API。"""
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
