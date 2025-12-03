import sys
import importlib
import atheris
import torch

from param_sampler import gen_config_for_api
from seq_env import SequenceEnv
from oracle_runtime import eval_sequence_oracles


# ===== 自动嵌入的 API 规格、约束和序列定义 =====

API_SPECS = {'torch.nn.functional.conv2d': {'api_name': 'torch.nn.functional.conv2d',
                                'category': 'conv2d',
                                'shape_vars': {'N': [1, 8],
                                               'C_in': [1, 64],
                                               'C_out': [1, 64],
                                               'H': [8, 64],
                                               'W': [8, 64],
                                               'kH': [1, 5],
                                               'kW': [1, 5],
                                               'C_per_group': [1, 64]},
                                'params': {'input': {'kind': 'tensor',
                                                     'shape_spec': ['N',
                                                                    'C_in',
                                                                    'H',
                                                                    'W'],
                                                     'dtype_choices': ['float32',
                                                                       'float64',
                                                                       'complex64']},
                                           'weight': {'kind': 'tensor',
                                                      'shape_spec': ['C_out',
                                                                     'C_per_group',
                                                                     'kH',
                                                                     'kW'],
                                                      'dtype_choices': ['float32',
                                                                        'float64',
                                                                        'complex64']},
                                           'bias': {'kind': 'tensor_optional',
                                                    'shape_spec': ['C_out'],
                                                    'dtype_choices': ['float32',
                                                                      'float64',
                                                                      'complex64']},
                                           'stride': {'kind': 'int_or_tuple',
                                                      'values': [1, 2, 3]},
                                           'padding': {'kind': 'int_or_tuple',
                                                       'values': [0,
                                                                  1,
                                                                  2,
                                                                  [1, 1],
                                                                  [2, 2]]},
                                           'dilation': {'kind': 'int_or_tuple',
                                                        'values': [1, 2, 3]},
                                           'groups': {'kind': 'int',
                                                      'range': [1, 16]}}},
 'torch.nn.functional.relu': {'api_name': 'torch.nn.functional.relu',
                              'category': 'activation',
                              'shape_vars': {'N': [1, 8], 'D': [1, 256]},
                              'params': {'input': {'kind': 'tensor',
                                                   'shape_spec': ['N', 'D'],
                                                   'dtype_choices': ['float16',
                                                                     'float32',
                                                                     'float64']},
                                         'inplace': {'kind': 'bool'}}},
 'torch.nn.functional.batch_norm': {'api_name': 'torch.nn.functional.batch_norm',
                                    'category': 'batch_norm',
                                    'shape_vars': {'N': [1, 8],
                                                   'C': [1, 64],
                                                   'H': [4, 64],
                                                   'W': [4, 64]},
                                    'params': {'input': {'kind': 'tensor',
                                                         'shape_spec': ['N',
                                                                        'C',
                                                                        'H',
                                                                        'W'],
                                                         'dtype_choices': ['float32',
                                                                           'float64']},
                                               'running_mean': {'kind': 'tensor_optional',
                                                                'shape_spec': ['C'],
                                                                'dtype_choices': ['float32',
                                                                                  'float64']},
                                               'running_var': {'kind': 'tensor_optional',
                                                               'shape_spec': ['C'],
                                                               'dtype_choices': ['float32',
                                                                                 'float64']},
                                               'weight': {'kind': 'tensor_optional',
                                                          'shape_spec': ['C'],
                                                          'dtype_choices': ['float32',
                                                                            'float64']},
                                               'bias': {'kind': 'tensor_optional',
                                                        'shape_spec': ['C'],
                                                        'dtype_choices': ['float32',
                                                                          'float64']},
                                               'training': {'kind': 'bool'},
                                               'momentum': {'kind': 'float',
                                                            'range': [0.01,
                                                                      0.99]},
                                               'eps': {'kind': 'float',
                                                       'range': [1e-06,
                                                                 0.001]}}},
 'torch.nn.functional.max_pool2d': {'api_name': 'torch.nn.functional.max_pool2d',
                                    'category': 'pool2d',
                                    'shape_vars': {'N': [1, 8],
                                                   'C': [1, 64],
                                                   'H': [4, 64],
                                                   'W': [4, 64]},
                                    'params': {'input': {'kind': 'tensor',
                                                         'shape_spec': ['N',
                                                                        'C',
                                                                        'H',
                                                                        'W'],
                                                         'dtype_choices': ['float16',
                                                                           'float32',
                                                                           'float64']},
                                               'kernel_size': {'kind': 'int_or_tuple',
                                                               'values': [1,
                                                                          2,
                                                                          3,
                                                                          [2,
                                                                           2],
                                                                          [3,
                                                                           3]]},
                                               'stride': {'kind': 'int_or_tuple',
                                                          'values': [1, 2, 3]},
                                               'padding': {'kind': 'int_or_tuple',
                                                           'values': [0,
                                                                      1,
                                                                      2,
                                                                      [1, 1]]},
                                               'dilation': {'kind': 'int_or_tuple',
                                                            'values': [1, 2]},
                                               'ceil_mode': {'kind': 'bool'},
                                               'return_indices': {'kind': 'bool'}}}}

API_CONSTRAINTS = {'torch.nn.functional.conv2d': ['C_in % groups == 0',
                                'C_out % groups == 0',
                                'C_per_group * groups == C_in',
                                'input.shape == (N, C_in, H, W)',
                                'weight.shape == (C_out, C_per_group, kH, kW)',
                                'bias is None or bias.shape == (C_out,)',
                                'all(isinstance(p, int) and p >= 0 for p in '
                                'padding_tuple)',
                                'all(isinstance(s, int) and s >= 1 for s in '
                                'stride_tuple)',
                                'all(isinstance(d, int) and d >= 1 for d in '
                                'dilation_tuple)'],
 'torch.nn.functional.relu': ['input.shape == (N, D)'],
 'torch.nn.functional.batch_norm': ['input.shape == (N, C, H, W)',
                                    'running_mean is None or '
                                    'running_mean.shape == (C,)',
                                    'running_var is None or running_var.shape '
                                    '== (C,)',
                                    'weight is None or weight.shape == (C,)',
                                    'bias is None or bias.shape == (C,)',
                                    '0.0 < momentum < 1.0',
                                    'eps > 0.0'],
 'torch.nn.functional.max_pool2d': ['input.shape == (N, C, H, W)',
                                    'all(isinstance(k, int) and k >= 1 for k '
                                    'in kernel_size_tuple)',
                                    'all(isinstance(s, int) and s >= 1 for s '
                                    'in stride_tuple)',
                                    'all(isinstance(p, int) and p >= 0 for p '
                                    'in padding_tuple)',
                                    'all(isinstance(d, int) and d >= 1 for d '
                                    'in dilation_tuple)']}

SEQUENCE_SPEC = {'sequence_name': 'conv_bn_relu_pool_block',
 'length_range': [4, 4],
 'steps': [{'api': 'torch.nn.functional.conv2d',
            'outputs': [{'name': 'conv_out'}]},
           {'api': 'torch.nn.functional.batch_norm',
            'inputs': {'input': '@tensor:conv_out'},
            'outputs': [{'name': 'bn_out'}]},
           {'api': 'torch.nn.functional.relu',
            'inputs': {'input': '@tensor:bn_out'},
            'outputs': [{'name': 'relu_out'}]},
           {'api': 'torch.nn.functional.max_pool2d',
            'inputs': {'input': '@tensor:relu_out'},
            'outputs': [{'name': 'pool_out'}]}],
 'oracles': [{'name': 'bn_shape_preserve',
              'kind': 'hard_bool',
              'expr': 'bn_out.shape == conv_out.shape'},
             {'name': 'relu_shape_preserve',
              'kind': 'hard_bool',
              'expr': 'relu_out.shape == bn_out.shape'},
             {'name': 'relu_non_negative',
              'kind': 'numeric',
              'mode': 'margin_ge',
              'expr': 'torch.all(relu_out >= 0)',
              'metrics': [{'id': 'relu_min',
                           'expr': 'float(relu_out.min().item())',
                           'lower': 0.0,
                           'upper': None,
                           'near_eps': 0.1}]},
             {'name': 'pool_preserve_nc',
              'kind': 'hard_bool',
              'expr': 'pool_out.shape[0] == relu_out.shape[0] and \\\n'
                      'pool_out.shape[1] == relu_out.shape[1]\n'},
             {'name': 'pool_hw_not_expand',
              'kind': 'numeric',
              'mode': 'discrete_ineq',
              'expr': 'pool_out.shape[2] <= relu_out.shape[2] and '
                      'pool_out.shape[3] <= relu_out.shape[3]',
              'metrics': [{'id': 'pool_hw_violation',
                           'expr': 'float(\n'
                                   '  max(pool_out.shape[2] - '
                                   'relu_out.shape[2], 0) +\n'
                                   '  max(pool_out.shape[3] - '
                                   'relu_out.shape[3], 0)\n'
                                   ')\n',
                           'lower': None,
                           'upper': 0.0,
                           'near_eps': 0.0}]}]}

SEQUENCE_ORACLES = [{'name': 'bn_shape_preserve',
  'kind': 'hard_bool',
  'expr': 'bn_out.shape == conv_out.shape'},
 {'name': 'relu_shape_preserve',
  'kind': 'hard_bool',
  'expr': 'relu_out.shape == bn_out.shape'},
 {'name': 'relu_non_negative',
  'kind': 'numeric',
  'mode': 'margin_ge',
  'expr': 'torch.all(relu_out >= 0)',
  'metrics': [{'id': 'relu_min',
               'expr': 'float(relu_out.min().item())',
               'lower': 0.0,
               'upper': None,
               'near_eps': 0.1}]},
 {'name': 'pool_preserve_nc',
  'kind': 'hard_bool',
  'expr': 'pool_out.shape[0] == relu_out.shape[0] and \\\n'
          'pool_out.shape[1] == relu_out.shape[1]\n'},
 {'name': 'pool_hw_not_expand',
  'kind': 'numeric',
  'mode': 'discrete_ineq',
  'expr': 'pool_out.shape[2] <= relu_out.shape[2] and pool_out.shape[3] <= '
          'relu_out.shape[3]',
  'metrics': [{'id': 'pool_hw_violation',
               'expr': 'float(\n'
                       '  max(pool_out.shape[2] - relu_out.shape[2], 0) +\n'
                       '  max(pool_out.shape[3] - relu_out.shape[3], 0)\n'
                       ')\n',
               'lower': None,
               'upper': 0.0,
               'near_eps': 0.0}]}]


def constraint_func(cfg, constraints):
    """通用约束检查函数（预条件）。

    - 把 cfg['_shape_vars'] 和 cfg 本身都摊平成局部变量，约束里可以直接写 N / C_in / input 等名字。
    - 自动构造 padding_tuple / stride_tuple / dilation_tuple。
    - constraints 是一组字符串表达式，用 eval(expr, {}, locs) 检查。
    """
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
    """根据序列 YAML 里的 inputs 规则，覆盖 cfg 中的部分参数。

    支持几种简单 DSL：
      - "@tensor:name" -> 从 env 里拿名字为 name 的张量
      - "@env:any"     -> 从 env 里随机挑一个张量
      - 其他值         -> 按字面值写入（预留将来扩展）
    """
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
    """根据字符串 api_name 加载并调用对应的函数。"""
    mod_name, func_name = api_name.rsplit(".", 1)
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, func_name)
    return fn(**call_kwargs)


def execute_sequence_once(sequence_spec: dict,
                          api_specs: dict,
                          api_constraints: dict,
                          fdp: atheris.FuzzedDataProvider):
    """执行一条 API 序列。

    返回 (env, ok, step_cfgs):
      - env: SequenceEnv，里面存了所有中间 Tensor
      - ok: False 表示中途失败（约束不满足 / API 抛错等）
      - step_cfgs: {short_name: cfg}，用于 oracle 中访问 cfg_xxx
    """
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


# ========== Atheris Harness ==========

@atheris.instrument_func
def TestOneInput(data: bytes) -> None:
    """Atheris 入口：从 bytes 生成并执行一次 API 序列。"""
    fdp = atheris.FuzzedDataProvider(data)

    env, ok, step_cfgs = execute_sequence_once(SEQUENCE_SPEC, API_SPECS, API_CONSTRAINTS, fdp)
    if not ok:
        return

    # 1) 基于序列级 oracle 做 reward 计算
    oracle_reward, oracle_violation = eval_sequence_oracles(env, step_cfgs, SEQUENCE_ORACLES)

    # 2) 额外：简单 NaN/Inf 检查，仍然保留
    has_nan_or_inf = False
    for ti in env.tensors:
        t = ti.tensor
        if torch.isnan(t).any() or torch.isinf(t).any():
            has_nan_or_inf = True
            break

    total_reward = oracle_reward
    if has_nan_or_inf:
        total_reward += 1.0

    # 这里先简单把 reward 分桶为几种分支，让 fuzzer 能“看到”差异。
    if total_reward <= 0.0:
        bucket = 0
    elif total_reward < 1.0:
        bucket = 1
    elif total_reward < 5.0:
        bucket = 2
    else:
        bucket = 3

    if bucket == 0:
        dummy = 0
    elif bucket == 1:
        dummy = 1
    elif bucket == 2:
        dummy = 2
    else:
        dummy = 3

    _ = dummy  # 避免未使用变量的警告


def main():
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
