#!/usr/bin/env python3
import sys
import random
import math
import atheris

with atheris.instrument_imports():
    import torch
    import torch.nn.functional as F

# --- 1. 定义 arms（模式 / 配置集合） ---
ARMS = [
    { "name": "small_kernel",       "conv_k_max": 3,  "conv_stride_max": 1, "pool": True  },
    { "name": "medium_kernel",      "conv_k_max": 5,  "conv_stride_max": 2, "pool": True  },
    { "name": "small_kernel_nopool","conv_k_max": 3,  "conv_stride_max": 1, "pool": False },
    { "name": "large_kernel_pool",  "conv_k_max": 7,  "conv_stride_max": 2, "pool": True  },
]

NUM_ARMS = len(ARMS)

# --- 2. 去掉 bandit：不再有 stats / policy / 更新逻辑 ---
# 这里只保留“语义覆盖”相关集合，便于做 flags 和 dummy 分支

seen_configs = set()
seen_shapes = set()

def make_config_sig(batch, in_c, out_c, h, w,
                    k_h, k_w, stride, padding,
                    did_pool, ph, pw, pool_stride, pool_padding):
    """将一次 conv+pool 的参数打成一个 signature，用于检测“新配置”"""
    return (batch, in_c, out_c, h, w,
            k_h, k_w, stride, padding,
            did_pool, ph, pw, pool_stride, pool_padding)

# --- 3. harness 主逻辑 ---
@atheris.instrument_func
def TestOneInput(data: bytes) -> None:
    fdp = atheris.FuzzedDataProvider(data)

    # 由 fuzz 输入的一个字节决定 arm_idx（不再有 policy / 学习）
    raw_id = fdp.ConsumeIntInRange(0, 255)
    arm_idx = raw_id % NUM_ARMS
    arm = ARMS[arm_idx]
    arm_name = arm["name"]  # 若需要调试打印可用

    # pool 相关参数的占位，便于构造 config_sig
    ph = pw = pool_stride = pool_padding = 0
    did_pool = False

    try:
        # 随机构造基础 dim / shape
        batch = fdp.ConsumeIntInRange(1, 3)
        in_c = fdp.ConsumeIntInRange(1, 4)
        out_c = fdp.ConsumeIntInRange(1, 4)
        h = fdp.ConsumeIntInRange(5, 32)
        w = fdp.ConsumeIntInRange(5, 32)

        # conv2d kernel/stride/padding 根据 arm
        max_k = arm.get("conv_k_max", 3)
        k_h = fdp.ConsumeIntInRange(1, min(max_k, h))
        k_w = fdp.ConsumeIntInRange(1, min(max_k, w))

        input_tensor = torch.randn(batch, in_c, h, w, dtype=torch.float32)
        weight = torch.randn(out_c, in_c, k_h, k_w, dtype=torch.float32)
        bias = None
        if fdp.ConsumeBool():
            bias = torch.randn(out_c, dtype=torch.float32)

        stride = fdp.ConsumeIntInRange(1, arm.get("conv_stride_max", 1))
        padding = fdp.ConsumeIntInRange(0, 2)

        out = F.conv2d(input_tensor, weight, bias=bias, stride=stride, padding=padding)
        out2 = F.relu(out)

        out3 = out2
        if arm.get("pool", False) and fdp.ConsumeBool():
            ph = fdp.ConsumeIntInRange(1, max(1, out2.shape[2] // 2))
            pw = fdp.ConsumeIntInRange(1, max(1, out2.shape[3] // 2))
            pool_stride = fdp.ConsumeIntInRange(1, ph)
            pool_padding = fdp.ConsumeIntInRange(0, 1)
            out3 = F.max_pool2d(
                out2,
                kernel_size=(ph, pw),
                stride=pool_stride,
                padding=pool_padding,
            )
            did_pool = True

        # --- 保留 NaN/Inf + “语义 flags” + dummy 分支（但不再算 reward） ---

        has_nan = torch.isnan(out3).any()
        has_inf = torch.isinf(out3).any()

        # 参数配置 signature
        config_sig = make_config_sig(
            batch, in_c, out_c, h, w,
            k_h, k_w, stride, padding,
            did_pool, ph, pw, pool_stride, pool_padding,
        )
        is_new_config = False
        if config_sig not in seen_configs:
            seen_configs.add(config_sig)
            is_new_config = True

        # 输出 shape
        shape = tuple(out3.shape)
        is_new_shape = False
        if shape not in seen_shapes:
            seen_shapes.add(shape)
            is_new_shape = True

        # 组合成 flags，走不同分支（让 fuzzer 看到一些行为差异）
        score_flags = 0
        if is_new_config:
            score_flags |= 1
        if is_new_shape:
            score_flags |= 2
        if has_nan or has_inf:
            score_flags |= 4

        if score_flags & 1:
            dummy = 1
        elif score_flags & 2:
            dummy = 2
        elif score_flags & 4:
            dummy = 3
        else:
            dummy = 0

        _ = dummy  # 避免未使用变量警告

    except Exception as e:
        # 这里不做 reward，只是简单吞掉异常，继续 fuzz
        # 如果你想在普通版本里保留“crash 就退出”的行为，可以改成 raise
        # raise
        pass

    return

def main():
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()

if __name__ == "__main__":
    main()
