import os
import sys
import importlib
import atheris
import torch

from fuzz_output.utils.param_sampler import gen_config_for_api, mutate_cfg

# 由 YAML 自动生成的 API 规格
SPEC = {'api_name': 'torch.nn.functional.relu',
 'category': 'activation',
 'shape_vars': {'N': [1, 8], 'D': [1, 256]},
 'params': {'input': {'kind': 'tensor',
                      'shape_spec': ['N', 'D'],
                      'dtype_choices': ['float16', 'float32', 'float64']},
            'inplace': {'kind': 'bool'}}}

# 从 YAML 中读取的约束表达式（字符串）
CONSTRAINTS = ['input.shape == (N, D)']


def constraint_func(cfg):
    """通用约束检查函数（预条件）。

    - 把 cfg['_shape_vars'] 和 cfg 本身都摊平成局部变量，
      这样约束里写 C_in / groups / input 等名字可以直接用。
    - 自动构造 padding_tuple / stride_tuple / dilation_tuple，方便 conv2d 这类 API 使用。
    - 对于 CONSTRAINTS 里每一个表达式 expr，用 eval(expr, {}, locs) 检查。
    """
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
    """多次尝试生成满足约束的 cfg。"""
    from utils.param_sampler import gen_config_for_api
    for _ in range(max_tries):
        cfg = gen_config_for_api(spec, fdp)
        if constraint_func(cfg):
            return cfg
    return None


def _call_target_api(cfg):
    """根据 SPEC['api_name'] 和 SPEC['params'] 自动调用目标 API。

    约定：YAML 里 params 的 key 必须和真实 API 的参数名一致，
    我们统一用关键字调用：target(**call_kwargs)。
    """
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
    """Atheris 入口：从 bytes 生成一次 API 调用。"""
    fdp = atheris.FuzzedDataProvider(data)
    cfg = gen_valid_config(SPEC, fdp)
    if cfg is None:
        # 这一串 bytes 很难凑出满足约束的参数，直接丢弃
        return
    # ===== FreeFuzz-style cfg mutation =====
    # steps 可以按参数个数决定：比如 1~len(params) 之间
    n_params = len(SPEC.get("params", {}))
    steps = fdp.ConsumeIntInRange(1, max(1, min(6, n_params)))  # 你可以调上限
    cfg = mutate_cfg(
        SPEC, cfg, fdp,
        constraint_func=constraint_func,
        steps=steps,
        max_attempts_per_step=6,
        p_type_mut=0.35,
        p_shape_mut=0.10,
    )

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
