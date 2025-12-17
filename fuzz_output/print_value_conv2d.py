import os
import sys
import importlib
import atheris
import torch

from param_sampler import gen_config_for_api

SPEC = {'api_name': 'torch.nn.functional.conv2d',
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
                      'shape_spec': ['N', 'C_in', 'H', 'W'],
                      'dtype_choices': ['float32', 'float64', 'complex64']},
            'weight': {'kind': 'tensor',
                       'shape_spec': ['C_out', 'C_per_group', 'kH', 'kW'],
                       'dtype_choices': ['float32', 'float64', 'complex64']},
            'bias': {'kind': 'tensor_optional',
                     'shape_spec': ['C_out'],
                     'dtype_choices': ['float32', 'float64', 'complex64']},
            'stride': {'kind': 'int_or_tuple', 'values': [1, 2, 3]},
            'padding': {'kind': 'int_or_tuple',
                        'values': [0, 1, 2, [1, 1], [2, 2]]},
            'dilation': {'kind': 'int_or_tuple', 'values': [1, 2, 3]},
            'groups': {'kind': 'int', 'range': [1, 16]}}}

CONSTRAINTS = []


def constraint_func(cfg):
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

    for expr in CONSTRAINTS:
        try:
            if not eval(expr, {}, locs):
                return False
        except Exception:
            return False
    return True


def gen_valid_config(spec, fdp, max_tries: int = 8):
    for _ in range(max_tries):
        cfg = gen_config_for_api(spec, fdp)
        if constraint_func(cfg):
            return cfg
    return None


def _call_target_api(cfg):
    api_name = SPEC.get("api_name")
    if not api_name:
        raise RuntimeError("SPEC missing 'api_name'")
    mod_name, func_name = api_name.rsplit(".", 1)
    mod = importlib.import_module(mod_name)
    target = getattr(mod, func_name)

    call_kwargs = {}
    for pname in SPEC.get("params", {}).keys():
        if pname in cfg:
            call_kwargs[pname] = cfg[pname]
    return target(**call_kwargs)


def _tensor_summary(x: torch.Tensor, max_print_elems: int = 10) -> str:
    # 注意：complex tensor 也能 min/max，但这里用 abs 的统计更直观
    with torch.no_grad():
        flat = x.reshape(-1)
        n = flat.numel()
        if n == 0:
            return "empty"
        # 取前几个元素做展示，避免输出爆炸
        show = flat[:max_print_elems].cpu()
        if x.is_complex():
            absx = x.abs()
            stats = f"abs(mean)={absx.mean().item():.6g}, abs(min)={absx.min().item():.6g}, abs(max)={absx.max().item():.6g}"
        elif x.dtype.is_floating_point:
            stats = f"mean={x.mean().item():.6g}, min={x.min().item():.6g}, max={x.max().item():.6g}"
        else:
            stats = f"min={x.min().item()}, max={x.max().item()}"
        return f"shape={tuple(x.shape)}, dtype={x.dtype}, {stats}, head={show}"


def print_cfg(cfg):
    print("==== CFG (shape vars) ====")
    print(cfg.get("_shape_vars", {}))

    print("\n==== CFG (non-tensor params) ====")
    for k in ["stride", "padding", "dilation", "groups"]:
        if k in cfg:
            print(f"{k} = {cfg[k]!r}")

    print("\n==== CFG (tensor params) ====")
    for name in SPEC.get("params", {}).keys():
        v = cfg.get(name, None)
        if isinstance(v, torch.Tensor):
            print(f"{name}: {_tensor_summary(v)}")
        else:
            # bias 可能是 None
            print(f"{name}: {v!r}")


def run_once(seed: int = 1234, data_size: int = 4096, max_tries: int = 8, call_api: bool = True):
    # 用确定性的随机字节，保证复现（也可以改成 os.urandom 做非确定性）
    import random as pyrandom
    pyrandom.seed(seed)
    data = bytes(pyrandom.getrandbits(8) for _ in range(data_size))

    fdp = atheris.FuzzedDataProvider(data)
    cfg = gen_valid_config(SPEC, fdp, max_tries=max_tries)
    if cfg is None:
        print(f"[!] Failed to generate valid cfg within {max_tries} tries.")
        return 1

    print_cfg(cfg)

    if call_api:
        try:
            out = _call_target_api(cfg)
            if isinstance(out, torch.Tensor):
                print("\n==== API result (tensor) ====")
                print(_tensor_summary(out))
            else:
                print("\n==== API result ====")
                print(repr(out))
        except Exception as e:
            print("\n[!] API call raised exception (still printed cfg above):")
            print(repr(e))
            return 2

    return 0


def main():
    # 用法：
    #   python harness.py
    #   python harness.py --seed=1 --nbytes=8192 --tries=32 --no-call
    seed = 1234
    nbytes = 4096
    tries = 8
    call_api = True

    for arg in sys.argv[1:]:
        if arg.startswith("--seed="):
            seed = int(arg.split("=", 1)[1])
        elif arg.startswith("--nbytes="):
            nbytes = int(arg.split("=", 1)[1])
        elif arg.startswith("--tries="):
            tries = int(arg.split("=", 1)[1])
        elif arg == "--no-call":
            call_api = False

    return run_once(seed=seed, data_size=nbytes, max_tries=tries, call_api=call_api)


if __name__ == "__main__":
    raise SystemExit(main())
