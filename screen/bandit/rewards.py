# screen/bandit/rewards.py
from __future__ import annotations

import math
from typing import Optional


def compute_proxy_reward(delta_ft: int, delta_cov: int, exec_s_last: Optional[float], mix: float = 0.7) -> float:
    """
    proxy = ((1-mix)*log(1+Δcov) + mix*log(1+Δft)) * exec/s
    """
    if exec_s_last is None or exec_s_last <= 0:
        exec_s_last = 1.0
    a = math.log(1.0 + max(0, delta_cov))
    b = math.log(1.0 + max(0, delta_ft))
    return ((1.0 - mix) * a + mix * b) * float(exec_s_last)


def compute_fast_reward(proxy: float, delta_files: int) -> float:
    # integrate "seed evolution" into fast signal (avoid exec/s-only illusion)
    return float(proxy) * math.log(1.0 + max(0, int(delta_files)))
