# oracle_runtime.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import torch


def _compute_margin_and_near(
    v: float,
    lower: float | None,
    upper: float | None,
    eps: float,
) -> Tuple[float, float]:
    """
    统一处理 margin / interval 的距离度量。

    返回 (viol, near):
      - viol: 违反程度（越大越严重），>=0
      - near: 满足约束但接近边界的程度（0~1），用于给一点小奖励
    """
    viol = 0.0
    near = 0.0

    # 无约束
    if lower is None and upper is None:
        return viol, near

    # 只有下界: v >= lower
    if lower is not None and upper is None:
        margin = v - lower
        if margin < 0:
            viol = -margin
        else:
            if eps > 0.0 and margin < eps:
                near = (eps - margin) / eps
        return viol, near

    # 只有上界: v <= upper
    if lower is None and upper is not None:
        margin = upper - v
        if margin < 0:
            viol = -margin
        else:
            if eps > 0.0 and margin < eps:
                near = (eps - margin) / eps
        return viol, near

    # 区间 [lower, upper]
    assert lower is not None and upper is not None
    if v < lower:
        viol = lower - v
        return viol, near
    if v > upper:
        viol = v - upper
        return viol, near

    # 在区间内部，越靠近任一边界 near 越大
    if eps > 0.0:
        dist_to_lower = v - lower
        dist_to_upper = upper - v
        boundary_dist = min(dist_to_lower, dist_to_upper)
        if boundary_dist < eps:
            near = (eps - boundary_dist) / eps

    return viol, near


def eval_sequence_oracles(
    env: Any,
    step_cfgs: Dict[str, Dict[str, Any]],
    sequence_oracles: List[Dict[str, Any]],
) -> Tuple[float, bool]:
    """
    在整条序列执行完后，按照统一 YAML schema 计算 oracle reward。

    约定：
      - env.tensors 是一个 list，每个元素有 .name 和 .tensor
      - step_cfgs: { short_name: cfg }，例如:
          layer_norm -> "layer_norm"   (cfg_layer_norm)
          dropout    -> "dropout"      (cfg_dropout)
          add        -> "add"          (cfg_add)
      - sequence_oracles 是 YAML 里的 oracles 列表（已经 embed 到 harness 里）

    返回:
      (total_reward, has_violation)

      total_reward: float，供 RL / bandit 等使用
      has_violation: bool，只要有一个 hard_bool False 或 metric 违反，就为 True
    """
    locs: Dict[str, Any] = {}

    # 1) 把所有命名 tensor 放进作用域
    for ti in getattr(env, "tensors", []):
        # 假设 ti 有 .name 和 .tensor 属性
        locs[ti.name] = ti.tensor

    # 2) 把每个 step 的 cfg 放进作用域，名为 cfg_<short_name>
    for short_name, cfg in step_cfgs.items():
        locs[f"cfg_{short_name}"] = cfg

    # 3) 提供 torch
    locs["torch"] = torch

    total_reward = 0.0
    has_violation = False

    for idx, oracle in enumerate(sequence_oracles):
        name = oracle.get("name", f"oracle_{idx}")
        kind = oracle.get("kind", "hard_bool")
        expr = oracle.get("expr")
        metrics = oracle.get("metrics", []) or []
        oracle_weight = float(oracle.get("weight", 1.0))

        # ---------- 先检查布尔 expr（如果有） ----------
        expr_ok = True
        if expr:
            try:
                expr_ok = bool(eval(expr, {}, locs))
            except Exception:
                expr_ok = False

        if kind == "hard_bool":
            if not expr_ok:
                # 纯 hard 布尔违反：给一个固定奖励
                total_reward += oracle_weight * 1.0
                has_violation = True
            # hard_bool 不看 metrics
            continue

        # ---------- numeric 类 oracle ----------
        oracle_reward = 0.0

        # 如果 expr 是 False，也算 violation，加一点基础奖励
        if not expr_ok:
            oracle_reward += 1.0
            has_violation = True

        for midx, m in enumerate(metrics):
            m_id = m.get("id", f"{name}_metric_{midx}")
            m_expr = m.get("expr")
            if not m_expr:
                continue

            try:
                v = float(eval(m_expr, {}, locs))
            except Exception:
                # metric 计算失败，当作严重违反
                oracle_reward += 1.0
                has_violation = True
                continue

            lower = m.get("lower", None)
            upper = m.get("upper", None)
            eps = float(m.get("near_eps", 0.0) or 0.0)
            m_weight = float(m.get("weight", 1.0))

            # 统一的 margin/interval 距离计算
            viol, near = _compute_margin_and_near(v, lower, upper, eps)

            # 一个简单的 shaping：viol 优先，其次 near
            if viol > 0.0:
                # 违反了：基础 1.0 + viol 本身
                oracle_reward += m_weight * (1.0 + viol)
                has_violation = True
            elif near > 0.0:
                # 没违反但接近边界：给一点小奖励
                oracle_reward += m_weight * (0.1 * near)
            else:
                # 在安全区：不给奖励（也可以给极小正值，看你喜好）
                oracle_reward += 0.0

        total_reward += oracle_weight * oracle_reward

    return total_reward, has_violation
