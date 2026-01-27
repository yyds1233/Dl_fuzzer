# screen/metrics/compute.py
from __future__ import annotations

from typing import Optional, Tuple


def compute_deltas(
    cov_first: Optional[float],
    cov_last: Optional[float],
    ft_first: Optional[float],
    ft_last: Optional[float],
) -> Tuple[int, int]:
    delta_cov = int((cov_last - cov_first) if (cov_first is not None and cov_last is not None) else 0)
    delta_ft = int((ft_last - ft_first) if (ft_first is not None and ft_last is not None) else 0)
    return delta_ft, delta_cov


def normalize_exec_s(exec_s_last: Optional[float]) -> float:
    return float(exec_s_last or 1.0)
