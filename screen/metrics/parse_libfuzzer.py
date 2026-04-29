from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional

COV_RE = re.compile(r"\bcov:\s*([0-9]+)")
FT_RE = re.compile(r"\bft:\s*([0-9]+)")
EXECS_RE = re.compile(r"\bexec/s:\s*([0-9]+(?:\.[0-9]+)?)([kKmM]?)")
INITED_COV_FT_RE = re.compile(r"^\s*#\d+\s+INITED\b.*\bcov:\s*(\d+)\s+ft:\s*(\d+)\b", re.M)
AFTER_INITED_COV_FT_RE = re.compile(r"^\s*#\d+\s+\w+\b.*\bcov:\s*(\d+)\s+ft:\s*(\d+)\b", re.M)
EVENT_CODE_RE = re.compile(r"^\s*#\d+\s+([A-Za-z]+)\b", re.M)
ATHERIS_COV_RE = re.compile(r"\btotal_coverage[=:]\s*([0-9]+)")
ATHERIS_FEAT_RE = re.compile(r"\bfeatures[=:]\s*([0-9]+)")

def _parse_num_with_suffix(x: str, suffix: str) -> float:
    v = float(x)
    if suffix.lower() == "k":
        return v * 1_000.0
    if suffix.lower() == "m":
        return v * 1_000_000.0
    return v

def parse_fuzzer_log(log_path: Path) -> Dict[str, Optional[float]]:
    text = log_path.read_text(errors="ignore")
    exec_pairs = EXECS_RE.findall(text)
    exec_s_last: Optional[float] = None
    if exec_pairs:
        x, suf = exec_pairs[-1]
        exec_s_last = _parse_num_with_suffix(x, suf)
    m = INITED_COV_FT_RE.search(text)
    if m:
        tail = text[m.start():]
        pairs = [(int(a), int(b)) for a, b in AFTER_INITED_COV_FT_RE.findall(tail)]
        if pairs:
            cov_first, ft_first = pairs[0]
            cov_last, ft_last = pairs[-1]
        else:
            cov_first = cov_last = int(m.group(1))
            ft_first = ft_last = int(m.group(2))
        return {"cov_first": cov_first, "cov_last": cov_last, "ft_first": ft_first, "ft_last": ft_last, "exec_s_last": exec_s_last}
    covs = [int(x) for x in COV_RE.findall(text)]
    fts = [int(x) for x in FT_RE.findall(text)]
    if covs and fts:
        return {"cov_first": int(covs[0]), "cov_last": int(covs[-1]), "ft_first": int(fts[0]), "ft_last": int(fts[-1]), "exec_s_last": exec_s_last}
    a_covs = [int(x) for x in ATHERIS_COV_RE.findall(text)]
    a_fts = [int(x) for x in ATHERIS_FEAT_RE.findall(text)]
    if a_covs and a_fts:
        return {"cov_first": int(a_covs[0]), "cov_last": int(a_covs[-1]), "ft_first": int(a_fts[0]), "ft_last": int(a_fts[-1]), "exec_s_last": exec_s_last}
    return {"cov_first": 0, "cov_last": 0, "ft_first": 0, "ft_last": 0, "exec_s_last": exec_s_last}

def parse_fuzzer_event_mix(log_path: Path) -> Dict[str, float]:
    text = log_path.read_text(errors="ignore")
    events = EVENT_CODE_RE.findall(text)
    total = len(events)
    if total == 0:
        return {"event_total": 0, "new_count": 0, "reduce_count": 0, "pulse_count": 0, "reduce_pulse_ratio": 0.0}
    new_count = sum(1 for e in events if e == "NEW")
    reduce_count = sum(1 for e in events if e == "REDUCE")
    pulse_count = sum(1 for e in events if e == "pulse")
    return {
        "event_total": total,
        "new_count": new_count,
        "reduce_count": reduce_count,
        "pulse_count": pulse_count,
        "reduce_pulse_ratio": float(reduce_count + pulse_count) / float(total),
    }
