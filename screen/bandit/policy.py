# screen/bandit/policy.py
from __future__ import annotations

from typing import Dict

from screen.bandit.bandit_core import DualChannelUCBSoftElim  # 先复用旧文件，后续你再迁移进来


def make_bandit(
    *,
    c_fast: float,
    c_slow: float,
    epsilon: float,
    elim_margin: float,
    elim_patience: int,
    elim_min_pulls: int,
    alpha_min: float,
    cooldown_steps: int,
    seed: int,
) -> DualChannelUCBSoftElim:
    return DualChannelUCBSoftElim(
        c_fast=c_fast,
        c_slow=c_slow,
        epsilon=epsilon,
        elim_margin=elim_margin,
        elim_patience=elim_patience,
        elim_min_pulls=elim_min_pulls,
        alpha_min=alpha_min,
        cooldown_steps=cooldown_steps,
        seed=seed,
    )
