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

# --- 2. bandit 统计 & policy ---
# stats: 按 index 记，不再用 name 做 key，方便和 policy 对齐
stats = { i: { "n": 0, "sum_r": 0.0 } for i in range(NUM_ARMS) }

# policy: 一个简单的概率分布，用来把 raw_id -> arm_idx
policy = [1.0 / NUM_ARMS for _ in range(NUM_ARMS)]

# 全局步数，用来控制多久更新一次 policy
global_step = 0

def update_arm_stats(arm_idx: int, r: float):
    """更新某个 arm 的统计信息"""
    s = stats[arm_idx]
    s["n"] += 1
    s["sum_r"] += r

def recompute_policy_from_stats():
    """
    用平均 reward 做 softmax，更新 policy。
    这是一个非常简单的 bandit 策略，可以根据需要换成 UCB 等。
    """
    global policy

    avgs = []
    for i in range(NUM_ARMS):
        s = stats[i]
        if s["n"] == 0:
            avgs.append(0.0)
        else:
            avgs.append(s["sum_r"] / s["n"])

    # 温度参数 tau 控制“贪心”程度，越小越贪心
    tau = 0.5
    exps = [math.exp(a / tau) for a in avgs]
    Z = sum(exps) or 1.0

    policy = [e / Z for e in exps]

def sample_arm_from_policy(raw_id: int) -> int:
    """
    将 fuzz 数据的 raw_id (0~255) + policy 共同决定 arm_idx。
    - raw_id 由 Atheris/libFuzzer 决定；
    - policy 慢慢向高 reward 的 arm 倾斜。
    """
    u = raw_id / 255.0  # 映射到 [0,1)
    cum = 0.0
    for i, p in enumerate(policy):
        cum += p
        if u < cum:
            return i
    return NUM_ARMS - 1

# --- 3. “语义覆盖”相关的集合，用来给 reward 加信号 ---
seen_configs = set()
seen_shapes = set()

def make_config_sig(batch, in_c, out_c, h, w,
                    k_h, k_w, stride, padding,
                    did_pool, ph, pw, pool_stride, pool_padding):
    """将一次 conv+pool 的参数打成一个 signature，用于检测“新配置”"""
    return (batch, in_c, out_c, h, w,
            k_h, k_w, stride, padding,
            did_pool, ph, pw, pool_stride, pool_padding)

# --- 4. harness 主逻辑 ---
@atheris.instrument_func
def TestOneInput(data: bytes) -> None:
    global global_step

    fdp = atheris.FuzzedDataProvider(data)
    global_step += 1

    # ---- (B) 由 fuzz 输入的一个字节 + policy 共同决定 arm_idx ----
    raw_id = fdp.ConsumeIntInRange(0, 255)
    arm_idx = sample_arm_from_policy(raw_id)
    arm = ARMS[arm_idx]
    arm_name = arm["name"]  # 如果想打印/调试用得到

    # 这些是 pool 相关参数的占位，便于构造 config_sig
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

        # ---- (A) reward 设计：带“语义覆盖”的信号 ----
        reward = 0.0

        # 1. NaN/Inf：潜在数值稳定性 bug，给高奖励
        has_nan = torch.isnan(out3).any()
        has_inf = torch.isinf(out3).any()
        if has_nan or has_inf:
            reward += 1.0

        # 2. 新的参数配置：第一次出现则奖励
        config_sig = make_config_sig(
            batch, in_c, out_c, h, w,
            k_h, k_w, stride, padding,
            did_pool, ph, pw, pool_stride, pool_padding,
        )
        is_new_config = False
        if config_sig not in seen_configs:
            seen_configs.add(config_sig)
            reward += 0.5
            is_new_config = True

        # 3. 新的输出 shape：第一次出现则奖励
        shape = tuple(out3.shape)
        is_new_shape = False
        if shape not in seen_shapes:
            seen_shapes.add(shape)
            reward += 0.3
            is_new_shape = True

        # 4. 输出规模：越大略微增加奖励
        numel = out3.numel()
        reward += min(numel / 1e5, 0.3)

        # 5. 成功执行的基础奖励
        reward += 0.1

        # （可选）将“语义 flags”转换成几段简单的分支，让 fuzzer 也感知到
        score_flags = 0
        if is_new_config:
            score_flags |= 1
        if is_new_shape:
            score_flags |= 2
        if has_nan or has_inf:
            score_flags |= 4

        # 这些分支本身没有逻辑意义，只是增加可见的控制流差异
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
        # 异常/崩溃视为高价值样本
        reward = 2.0
        # 不 re-raise，让 fuzzer 继续；若想 crash 退出，可改成 raise

    # --- 更新 bandit stats & policy ---
    update_arm_stats(arm_idx, reward)

    # 每隔一定步数重新估计一次 policy
    if global_step % 1000 == 0:
        recompute_policy_from_stats()

    return

def main():
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()

if __name__ == "__main__":
    main()
