#!/usr/bin/env python3
# screen_single_api_round1.py
import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------
# Config: profile grid
# ---------------------------
PROFILE_GRID = {
    # ===== 你原来的 =====
    "MUT_STEPS_MAX": [2, 6, 10],
    "P_TYPE_MUT": [0.2, 0.5, 0.8],
    "P_SHAPE_MUT": [0.0, 0.05, 0.15],
    "SEED_TRIES": [3, 8, 12],
    "MUT_ATTEMPTS": [3, 6, 10],

    # ===== C) non-contiguous / stride diversity =====
    # 默认建议从 0 开始，先保证执行率，再逐步开
    "P_NONCONTIG": [0.0, 0.02, 0.05],
    # 作为对照组（在非连续分支里小概率 contiguous()），一般不用太大
    "P_RECONTIG": [0.0, 0.05, 0.10],

    # ===== D) empty tensor / 0-length dims =====
    # 注意：ALLOW_EMPTY 是开关，不是概率
    "ALLOW_EMPTY": [0, 1],
    # 开启后再用低概率注入 0-length，别太猛
    "P_EMPTY_DIM": [0.0, 0.005, 0.01],
    # N/C 置 0 很危险，默认建议保持 0（不启用）
    "P_EMPTY_NC": [0.0, 0.0005, 0.001],

    # =====（可选）序列 harness 也能复用的开关 =====
    # 如果你在序列 harness 里用了 ENABLE_MUT / SEQ_ENABLE_MUT，这里也可以一起扫
    "ENABLE_MUT": [0, 1],
}

# Default libFuzzer/Atheris flags (you can override via CLI)
DEFAULT_FUZZ_FLAGS = [
    "-ignore_timeouts=1",
    "-rss_limit_mb=4096",
    "-use_value_profile=1",
    "-entropic=1",
]

# ---------------------------
# Log parsing regex
# ---------------------------
COV_RE = re.compile(r"\bcov:\s*([0-9]+)")
FT_RE = re.compile(r"\bft:\s*([0-9]+)")
CORP_RE = re.compile(r"\bcorp:\s*([0-9]+)")
EXECS_RE = re.compile(r"\bexec/s:\s*([0-9]+(?:\.[0-9]+)?)([kKmM]?)")
ARTIFACT_RE = re.compile(r"Test unit written to\s+(\S+)")

EPS = 1e-9


def _parse_num_with_suffix(x: str, suffix: str) -> float:
    v = float(x)
    if suffix.lower() == "k":
        return v * 1_000.0
    if suffix.lower() == "m":
        return v * 1_000_000.0
    return v


def sanitize_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def canonical_profile_id(profile: Dict[str, object], length: int = 10) -> str:
    payload = json.dumps(profile, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


def sample_profiles(n: int, seed: int) -> List[Dict[str, object]]:
    rnd = random.Random(seed)
    keys = list(PROFILE_GRID.keys())
    profiles = []
    seen = set()
    tries = 0
    while len(profiles) < n and tries < n * 50:
        tries += 1
        p = {k: rnd.choice(PROFILE_GRID[k]) for k in keys}
        pid = canonical_profile_id(p)
        if pid in seen:
            continue
        seen.add(pid)
        p["profile_id"] = pid
        profiles.append(p)
    return profiles


@dataclass
class Round1Result:
    profile_id: str
    profile: Dict[str, object]
    returncode: int
    seconds: float

    cov_first: Optional[int]
    cov_last: Optional[int]
    ft_first: Optional[int]
    ft_last: Optional[int]
    corp_first: Optional[int]
    corp_last: Optional[int]
    exec_s_last: Optional[float]

    crash_files: int
    artifact_lines: int

    delta_cov: int
    delta_ft: int
    delta_corp: int
    score: float

    corpus_dir: str
    crash_dir: str
    log_path: str

    # NEW: for paper/debug (filled only for mops mode)
    score_components: Optional[Dict[str, float]] = field(default=None)


def parse_fuzzer_log(log_path: Path) -> Dict[str, object]:
    text = log_path.read_text(errors="ignore")

    covs = [int(x) for x in COV_RE.findall(text)]
    fts = [int(x) for x in FT_RE.findall(text)]
    corps = [int(x) for x in CORP_RE.findall(text)]

    exec_s = None
    exec_hits = EXECS_RE.findall(text)
    if exec_hits:
        x, suf = exec_hits[-1]
        exec_s = _parse_num_with_suffix(x, suf)

    artifacts = ARTIFACT_RE.findall(text)

    return {
        "cov_first": covs[0] if covs else None,
        "cov_last": covs[-1] if covs else None,
        "ft_first": fts[0] if fts else None,
        "ft_last": fts[-1] if fts else None,
        "corp_first": corps[0] if corps else None,
        "corp_last": corps[-1] if corps else None,
        "exec_s_last": exec_s,
        "artifact_lines": len(artifacts),
    }


def _robust_norm(x: float, p5: float, p95: float) -> float:
    x = max(min(x, p95), p5)
    return (x - p5) / (p95 - p5 + EPS)


def compute_score(delta_cov: int, delta_ft: int, exec_s_last: Optional[float], mode: str) -> float:
    # score = log(1+Δcov) * exec/s
    # or:    log(1+Δft)  * exec/s
    if exec_s_last is None or exec_s_last <= 0:
        exec_s_last = 1.0
    if mode == "cov":
        return math.log(1.0 + max(0, delta_cov)) * float(exec_s_last)
    if mode == "ft":
        return math.log(1.0 + max(0, delta_ft)) * float(exec_s_last)
    raise ValueError(f"Unknown score mode: {mode}")


def compute_mops_fast(
    *,
    delta_ft: int,
    delta_cov: int,
    delta_corp: int,
    exec_s_last: Optional[float],
    artifact_lines: int,
    pct: Dict[str, Dict[str, float]],
    w_ft: float = 0.7,
    w_cov: float = 0.3,
    lam: float = 0.8,
    alpha: float = 0.7,
    ft_deltas_over_time: Optional[List[int]] = None,
) -> Dict[str, float]:
    if exec_s_last is None or exec_s_last <= 0:
        exec_s_last = 1.0

    ft_n = _robust_norm(float(max(0, delta_ft)), pct["delta_ft"]["p5"], pct["delta_ft"]["p95"])
    cov_n = _robust_norm(float(max(0, delta_cov)), pct["delta_cov"]["p5"], pct["delta_cov"]["p95"])
    corp_n = _robust_norm(float(max(0, delta_corp)), pct["delta_corp"]["p5"], pct["delta_corp"]["p95"])
    ex_n = _robust_norm(float(exec_s_last), pct["exec_s"]["p5"], pct["exec_s"]["p95"])

    novelty = w_ft * ft_n + w_cov * cov_n
    corpus_growth = math.sqrt(corp_n)
    efficiency = ex_n ** alpha

    stability = 1.0
    if ft_deltas_over_time and len(ft_deltas_over_time) >= 3:
        m = sum(ft_deltas_over_time) / (len(ft_deltas_over_time) + EPS)
        if m > 0:
            v = sum((x - m) ** 2 for x in ft_deltas_over_time) / len(ft_deltas_over_time)
            std = math.sqrt(v)
            cv = std / (m + EPS)
            stability = 1.0 / (1.0 + cv)

    crash_flag = 1.0 if artifact_lines > 0 else 0.0
    score = efficiency * (lam * novelty + (1.0 - lam) * corpus_growth) * stability

    return {
        "score": float(score),
        "novelty": float(novelty),
        "corpus_growth": float(corpus_growth),
        "efficiency": float(efficiency),
        "stability": float(stability),
        "crash_flag": float(crash_flag),
        "ft_n": float(ft_n),
        "cov_n": float(cov_n),
        "corp_n": float(corp_n),
        "ex_n": float(ex_n),
    }


def ensure_empty_dir(p: Path):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


# ---------------------------
# Percentile helpers (no numpy)
# ---------------------------
def _percentile_sorted(vals_sorted: List[float], q: float) -> float:
    """
    q in [0, 100]. Linear interpolation between closest ranks.
    """
    n = len(vals_sorted)
    if n == 0:
        return 0.0
    if n == 1:
        return float(vals_sorted[0])
    q = max(0.0, min(100.0, q))
    pos = (q / 100.0) * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(vals_sorted[lo])
    w = pos - lo
    return float(vals_sorted[lo] * (1.0 - w) + vals_sorted[hi] * w)


def _p5_p95(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"p5": 0.0, "p95": 1.0}
    s = sorted(float(x) for x in vals)
    p5 = _percentile_sorted(s, 5.0)
    p95 = _percentile_sorted(s, 95.0)
    # avoid degenerate range
    if abs(p95 - p5) < EPS:
        p95 = p5 + 1.0
    return {"p5": float(p5), "p95": float(p95)}


def build_pct(results: List[Round1Result]) -> Dict[str, Dict[str, float]]:
    return {
        "delta_ft": _p5_p95([float(r.delta_ft) for r in results]),
        "delta_cov": _p5_p95([float(r.delta_cov) for r in results]),
        "delta_corp": _p5_p95([float(r.delta_corp) for r in results]),
        "exec_s": _p5_p95([float(r.exec_s_last or 0.0) for r in results]),
    }


def run_one_campaign(
    python: str,
    harness_path: Path,
    api_id: str,
    profile: Dict[str, object],
    root: Path,
    epoch_sec: int,
    fuzz_flags: List[str],
    keep_dirs: bool,
) -> Round1Result:
    profile_id = str(profile["profile_id"])

    corpus_dir = root / "Corpus" / api_id / profile_id
    crash_dir = root / "Crash" / api_id / profile_id
    run_dir = root / "screen_runs" / api_id / profile_id
    log_path = run_dir / "round1_fuzzer.log"
    meta_path = run_dir / "profile.json"

    run_dir.mkdir(parents=True, exist_ok=True)

    ensure_empty_dir(corpus_dir)
    ensure_empty_dir(crash_dir)

    meta_path.write_text(json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8")

    env = os.environ.copy()
    for k in PROFILE_GRID.keys():
        env[k] = str(profile[k])

    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("TORCH_NUM_THREADS", "1")

    cmd = [
        python,
        str(harness_path),
        str(corpus_dir),
        f"-artifact_prefix={str(crash_dir)}/",
        f"-max_total_time={int(epoch_sec)}",
        *fuzz_flags,
    ]

    t0 = time.time()
    with log_path.open("w", encoding="utf-8", errors="ignore") as lf:
        proc = subprocess.Popen(cmd, stderr=lf, stdout=lf, env=env)
        try:
            proc.wait(timeout=epoch_sec + 20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    t1 = time.time()

    parsed = parse_fuzzer_log(log_path)

    cov_first = parsed["cov_first"]
    cov_last = parsed["cov_last"]
    ft_first = parsed["ft_first"]
    ft_last = parsed["ft_last"]
    corp_first = parsed["corp_first"]
    corp_last = parsed["corp_last"]
    exec_s_last = parsed["exec_s_last"]
    artifact_lines = int(parsed["artifact_lines"])

    delta_cov = (cov_last - cov_first) if (cov_first is not None and cov_last is not None) else 0
    delta_ft = (ft_last - ft_first) if (ft_first is not None and ft_last is not None) else 0
    delta_corp = (corp_last - corp_first) if (corp_first is not None and corp_last is not None) else 0

    crash_files = 0
    if crash_dir.exists():
        crash_files = sum(1 for _ in crash_dir.glob("*") if _.is_file())

    # IMPORTANT: do NOT compute score here anymore (needs pct over all candidates)
    score = 0.0

    if not keep_dirs:
        shutil.rmtree(corpus_dir, ignore_errors=True)
        shutil.rmtree(crash_dir, ignore_errors=True)

    return Round1Result(
        profile_id=profile_id,
        profile={k: profile[k] for k in PROFILE_GRID.keys()},
        returncode=proc.returncode,
        seconds=(t1 - t0),
        cov_first=cov_first,
        cov_last=cov_last,
        ft_first=ft_first,
        ft_last=ft_last,
        corp_first=corp_first,
        corp_last=corp_last,
        exec_s_last=exec_s_last,
        crash_files=crash_files,
        artifact_lines=artifact_lines,
        delta_cov=int(delta_cov),
        delta_ft=int(delta_ft),
        delta_corp=int(delta_corp),
        score=float(score),
        corpus_dir=str(corpus_dir),
        crash_dir=str(crash_dir),
        log_path=str(log_path),
        score_components=None,
    )


def pick_top_fraction(results: List[Round1Result], frac: float) -> List[Round1Result]:
    results_sorted = sorted(results, key=lambda r: r.score, reverse=True)
    k = max(1, int(math.ceil(len(results_sorted) * frac)))
    return results_sorted[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="fuzz_output", help="Your fuzz_output directory")
    ap.add_argument("--harness", required=True, help="Path to a single harness .py (e.g., fuzz_output/auto_conv2d.py)")
    ap.add_argument("--api", default=None, help="API id/name for directory naming (default: from harness filename)")
    ap.add_argument("--n_profiles", type=int, default=60, help="How many random profiles to sample")
    ap.add_argument("--seed", type=int, default=1337, help="Random seed for profile sampling")
    ap.add_argument("--epoch", type=int, default=60, help="Round1 fuzz seconds per campaign (30-60 recommended)")
    ap.add_argument(
        "--score_mode",
        choices=["cov", "ft", "mops"],
        default="mops",
        help="Scoring: cov/ft (simple) or mops (mixed robust score using pct)",
    )
    ap.add_argument("--top_frac", type=float, default=0.333, help="Keep top fraction for next round (default 1/3)")
    ap.add_argument("--python", default=sys.executable, help="Python executable (default current)")
    ap.add_argument("--keep_dirs", action="store_true", help="Keep corpus/crash dirs for ALL campaigns (disk heavy)")
    ap.add_argument("--fuzz_flags", default=" ".join(DEFAULT_FUZZ_FLAGS),
                    help="Extra libFuzzer flags (space-separated), default matches your current flags")

    args = ap.parse_args()

    root = Path(args.root).resolve()
    harness_path = Path(args.harness).resolve()
    if not harness_path.exists():
        print(f"[!] harness not found: {harness_path}")
        sys.exit(2)

    if args.api:
        api_id = sanitize_name(args.api)
    else:
        stem = harness_path.stem
        api_id = stem[len("auto_"):] if stem.startswith("auto_") else stem
        api_id = sanitize_name(api_id)

    fuzz_flags = [x for x in args.fuzz_flags.split(" ") if x.strip()]
    print(f"[+] root={root}")
    print(f"[+] harness={harness_path}")
    print(f"[+] api_id={api_id}")
    print(f"[+] n_profiles={args.n_profiles} epoch={args.epoch}s score_mode={args.score_mode}")
    print(f"[+] fuzz_flags={' '.join(fuzz_flags)}")

    profiles = sample_profiles(args.n_profiles, seed=args.seed)
    print(f"[+] sampled {len(profiles)} unique profiles")

    # 1) run all campaigns, collect raw metrics only
    results: List[Round1Result] = []
    for i, prof in enumerate(profiles, 1):
        pid = prof["profile_id"]
        print(f"    [{i:03d}/{len(profiles):03d}] run profile_id={pid} profile={ {k: prof[k] for k in PROFILE_GRID.keys()} }")
        try:
            r = run_one_campaign(
                python=args.python,
                harness_path=harness_path,
                api_id=api_id,
                profile=prof,
                root=root,
                epoch_sec=args.epoch,
                fuzz_flags=fuzz_flags,
                keep_dirs=args.keep_dirs,
            )
        except Exception as e:
            print(f"      [!] campaign failed: {e}")
            continue
        results.append(r)
        print(f"      raw: Δft={r.delta_ft} Δcov={r.delta_cov} Δcorp={r.delta_corp} exec/s={r.exec_s_last} crashes={r.crash_files}")

    if not results:
        print("[!] No results collected.")
        sys.exit(1)

    # 2) compute pct over all candidates (per-API, per-round)
    pct = build_pct(results)

    # 3) assign final scores AFTER pct is available
    for r in results:
        if args.score_mode in ("cov", "ft"):
            r.score = float(compute_score(r.delta_cov, r.delta_ft, r.exec_s_last, args.score_mode))
            r.score_components = None
        else:
            out = compute_mops_fast(
                delta_ft=r.delta_ft,
                delta_cov=r.delta_cov,
                delta_corp=r.delta_corp,
                exec_s_last=r.exec_s_last,
                artifact_lines=r.artifact_lines,
                pct=pct,
            )
            r.score = float(out["score"])
            r.score_components = out

    # 4) save artifacts
    out_dir = root / "screen_runs" / api_id
    out_dir.mkdir(parents=True, exist_ok=True)

    pct_json = out_dir / "round1_pct.json"
    pct_json.write_text(json.dumps(pct, indent=2), encoding="utf-8")

    all_json = out_dir / "round1_all_results.json"
    all_json.write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")

    top = pick_top_fraction(results, args.top_frac)
    top_json = out_dir / "round1_top_results.json"
    top_json.write_text(json.dumps([asdict(r) for r in top], indent=2), encoding="utf-8")

    print("\n===== Round1 Leaderboard (Top) =====")
    for rank, r in enumerate(sorted(top, key=lambda x: x.score, reverse=True), 1):
        print(
            f"{rank:02d}. profile_id={r.profile_id} score={r.score:.4f} "
            f"Δft={r.delta_ft} Δcov={r.delta_cov} Δcorp={r.delta_corp} "
            f"exec/s={r.exec_s_last} crashes={r.crash_files}"
        )

    print(f"\n[+] wrote: {pct_json}")
    print(f"[+] wrote: {all_json}")
    print(f"[+] wrote: {top_json}")
    print("[+] Next round candidates are in round1_top_results.json")


if __name__ == "__main__":
    main()
