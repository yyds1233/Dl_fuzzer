# screen/metrics/parse_libfuzzer.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional


COV_RE = re.compile(r"\bcov:\s*([0-9]+)")
FT_RE = re.compile(r"\bft:\s*([0-9]+)")
EXECS_RE = re.compile(r"\bexec/s:\s*([0-9]+(?:\.[0-9]+)?)([kKmM]?)")

# 只在 INITED(含) 之后抓 cov/ft；这样会自动跳过 seed replay/pulse
INITED_COV_FT_RE = re.compile(r"^\s*#\d+\s+INITED\b.*\bcov:\s*(\d+)\s+ft:\s*(\d+)\b", re.M)
AFTER_INITED_COV_FT_RE = re.compile(r"^\s*#\d+\s+\w+\b.*\bcov:\s*(\d+)\s+ft:\s*(\d+)\b", re.M)


def _parse_num_with_suffix(x: str, suffix: str) -> float:
    v = float(x)
    if suffix.lower() == "k":
        return v * 1_000.0
    if suffix.lower() == "m":
        return v * 1_000_000.0
    return v


def parse_fuzzer_log(log_path: Path) -> Dict[str, Optional[float]]:
    text = log_path.read_text(errors="ignore")

    exec_s = None
    hits = EXECS_RE.findall(text)
    if hits:
        x, suf = hits[-1]
        exec_s = _parse_num_with_suffix(x, suf)

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
        return {
            "cov_first": cov_first,
            "cov_last": cov_last,
            "ft_first": ft_first,
            "ft_last": ft_last,
            "exec_s_last": exec_s,
        }

    covs = [int(x) for x in COV_RE.findall(text)]
    fts = [int(x) for x in FT_RE.findall(text)]
    return {
        "cov_first": covs[0] if covs else None,
        "cov_last": covs[-1] if covs else None,
        "ft_first": fts[0] if fts else None,
        "ft_last": fts[-1] if fts else None,
        "exec_s_last": exec_s,
    }
