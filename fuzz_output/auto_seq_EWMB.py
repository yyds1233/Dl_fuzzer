import sys
import os
import importlib
import atheris
import torch

from utils.param_sampler import gen_config_for_api, mutate_cfg
from utils.seq_env import SequenceEnv
from utils.oracle_runtime import eval_sequence_oracles


# ===== 自动嵌入的 API 规格、约束和序列定义 =====

API_SPECS = {'torch.mul': {'api_name': 'torch.mul',
               'category': 'mul',
               'shape_vars': {'N': [1, 8], 'D': [1, 256]},
               'params': {'input': {'kind': 'tensor_or_scalar',
                                    'shape_spec': ['N', 'D'],
                                    'dtype_choices': ['float32',
                                                      'float64',
                                                      'int64']},
                          'other': {'kind': 'tensor_or_scalar',
                                    'shape_spec': ['N', 'D'],
                                    'dtype_choices': ['float32',
                                                      'float64',
                                                      'int64']}}},
 'torch.where': {'api_name': 'torch.where',
                 'category': 'elementwise_select',
                 'shape_vars': {'B0': [1, 4],
                                'B1': [1, 4],
                                'B2': [1, 4],
                                'D': [2, 16]},
                 'params': {'condition': {'kind': 'tensor',
                                          'shape_spec': ['B0', 'D'],
                                          'dtype_choices': ['bool']},
                            'input': {'kind': 'tensor_or_scalar',
                                      'shape_spec': ['B1', 'D'],
                                      'dtype_choices': ['float32',
                                                        'float64',
                                                        'int64']},
                            'other': {'kind': 'tensor_or_scalar',
                                      'shape_spec': ['B2', 'D'],
                                      'dtype_choices': ['float32',
                                                        'float64',
                                                        'int64']}}},
 'torch.add': {'api_name': 'torch.add',
               'category': 'add',
               'shape_vars': {'N': [1, 8], 'D': [1, 256]},
               'params': {'input': {'kind': 'tensor_or_scalar',
                                    'shape_spec': ['N', 'D'],
                                    'dtype_choices': ['float32',
                                                      'float64',
                                                      'int64']},
                          'other': {'kind': 'tensor_or_scalar',
                                    'shape_spec': ['N', 'D'],
                                    'dtype_choices': ['float32',
                                                      'float64',
                                                      'int64']},
                          'alpha': {'kind': 'float', 'range': [-2.0, 2.0]}}},
 'torch.sum': {'api_name': 'torch.sum',
               'category': 'reduction_sum',
               'shape_vars': {'N': [1, 8], 'D': [1, 256]},
               'params': {'input': {'kind': 'tensor',
                                    'shape_spec': ['N', 'D'],
                                    'dtype_choices': ['float32',
                                                      'float64',
                                                      'int64']},
                          'dim': {'kind': 'int', 'range': [0, 1]},
                          'keepdim': {'kind': 'bool'}}}}
API_CONSTRAINTS = {'torch.mul': ['not isinstance(input, torch.Tensor) or input.shape == (N, D)',
               'not isinstance(other, torch.Tensor) or other.shape == (N, D)'],
 'torch.where': ['condition.dtype == condition.bool().dtype'],
 'torch.add': ['not isinstance(input, torch.Tensor) or input.shape == (N, D)',
               'not isinstance(other, torch.Tensor) or other.shape == (N, D)'],
 'torch.sum': ['input.shape == (N, D)',
               'input.dim() == 2',
               '-input.dim() <= dim < input.dim()']}
SEQUENCE_SPEC = {'sequence_name': 'elementwise_where_mix_block',
 'length_range': [4, 4],
 'steps': [{'api': 'torch.where', 'outputs': [{'name': 'where_out'}]},
           {'api': 'torch.add',
            'inputs': {'input': '@tensor:where_out',
                       'other': '@tensor:where_out'},
            'outputs': [{'name': 'add_out'}]},
           {'api': 'torch.mul',
            'inputs': {'input': '@tensor:add_out', 'other': '@tensor:add_out'},
            'outputs': [{'name': 'mul_out'}]},
           {'api': 'torch.sum',
            'inputs': {'input': '@tensor:mul_out'},
            'outputs': [{'name': 'sum_out'}]}],
 'oracles': [{'name': 'add_is_double',
              'kind': 'numeric',
              'mode': 'metamorphic',
              'expr': 'torch.allclose(add_out, where_out + where_out, '
                      'atol=1e-4, rtol=1e-4)',
              'metrics': [{'id': 'add_double_error',
                           'expr': 'float(\n'
                                   '  torch.norm(\n'
                                   '    (add_out - (where_out + '
                                   'where_out)).float()\n'
                                   '  ).item()\n'
                                   ')\n',
                           'lower': 0.0,
                           'upper': 0.001,
                           'near_eps': 0.0001}]},
             {'name': 'mul_is_square',
              'kind': 'numeric',
              'mode': 'metamorphic',
              'expr': 'torch.allclose(mul_out, add_out * add_out, atol=1e-4, '
                      'rtol=1e-4)',
              'metrics': [{'id': 'mul_square_error',
                           'expr': 'float(\n'
                                   '  torch.norm(\n'
                                   '    (mul_out - (add_out * '
                                   'add_out)).float()\n'
                                   '  ).item()\n'
                                   ')\n',
                           'lower': 0.0,
                           'upper': 0.001,
                           'near_eps': 0.0001}]},
             {'name': 'sum_scalar_when_dim_none',
              'kind': 'hard_bool',
              'expr': "(('dim' in cfg_sum) and (cfg_sum['dim'] is not None)) "
                      'or (sum_out.dim() == 0)'}]}
SEQUENCE_ORACLES = [{'name': 'add_is_double',
  'kind': 'numeric',
  'mode': 'metamorphic',
  'expr': 'torch.allclose(add_out, where_out + where_out, atol=1e-4, '
          'rtol=1e-4)',
  'metrics': [{'id': 'add_double_error',
               'expr': 'float(\n'
                       '  torch.norm(\n'
                       '    (add_out - (where_out + where_out)).float()\n'
                       '  ).item()\n'
                       ')\n',
               'lower': 0.0,
               'upper': 0.001,
               'near_eps': 0.0001}]},
 {'name': 'mul_is_square',
  'kind': 'numeric',
  'mode': 'metamorphic',
  'expr': 'torch.allclose(mul_out, add_out * add_out, atol=1e-4, rtol=1e-4)',
  'metrics': [{'id': 'mul_square_error',
               'expr': 'float(\n'
                       '  torch.norm(\n'
                       '    (mul_out - (add_out * add_out)).float()\n'
                       '  ).item()\n'
                       ')\n',
               'lower': 0.0,
               'upper': 0.001,
               'near_eps': 0.0001}]},
 {'name': 'sum_scalar_when_dim_none',
  'kind': 'hard_bool',
  'expr': "(('dim' in cfg_sum) and (cfg_sum['dim'] is not None)) or "
          '(sum_out.dim() == 0)'}]


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
    """通用约束检查函数（预条件）。"""
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
    """根据序列 YAML 里的 inputs 规则，覆盖 cfg 中的部分参数。"""
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
    """为某个 step 生成一次可用 cfg（带重试 + 可控变异）。"""
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
    """执行一条 API 序列。返回 (env, ok, step_cfgs)。"""
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
