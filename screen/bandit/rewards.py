# screen/bandit/rewards.py
from __future__ import annotations

import math
from typing import Optional


def compute_proxy_reward(
    delta_ft: int,
    delta_cov: int,
    exec_s_last: Optional[float],
    mix: float = 0.25,
) -> float:
    """Fast proxy reward with stronger Δcov preference.

    mix controls the Δft weight. Therefore:
      - mix=0.25 means 75% Δcov + 25% Δft
      - mix=0.70 means 30% Δcov + 70% Δft

    Compared with the old PyTorch version:
      1. speed uses sqrt(exec/s), so very fast but low-coverage harnesses do not dominate;
      2. any positive Δcov gets a 2x bonus;
      3. the default mix is Δcov-oriented.
    """
    if exec_s_last is None or exec_s_last <= 0:
        exec_s_last = 1.0

    cov_score = math.log(1.0 + max(0, int(delta_cov)))
    ft_score = math.log(1.0 + max(0, int(delta_ft)))
    speed_term = math.sqrt(float(exec_s_last))
    cov_bonus = 2.0 if int(delta_cov) > 0 else 1.0

    return (((1.0 - mix) * cov_score + mix * ft_score) * speed_term) * cov_bonus


def compute_fast_reward(proxy: float, delta_files: int) -> float:
    """Do not zero-out reward just because no new corpus file was materialized."""
    return float(proxy) * (1.0 + 0.1 * math.log(1.0 + max(0, int(delta_files))))
