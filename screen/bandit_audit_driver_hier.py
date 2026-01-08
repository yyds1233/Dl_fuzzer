#!/usr/bin/env python3
# bandit_audit_driver_hier.py
#
# Increment-aware adaptive fuzz scheduler (Scheme 1):
# - Corpus is shared per harness (continuous fuzz process).
# - Two manifests per harness: current vs audit_base.
# - Slow audit uses delta since last audit (audit_base -> current).
# - Delta seeds are tagged with profile_id prefix post-epoch for attribution:
#     <profile_id>__<orig_name>
# - Slow audit runs ONCE per harness per audit window (single llvm-cov export).
# - Slow credit to profiles is DISTRIBUTED by attribution stats (top-k optional),
#   NO per-profile llvm-cov re-run.
#
import argparse
import json
import math
import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, DefaultDict
from collections import defaultdict
import random

from bandit_core import DualChannelUCBSoftElim


# ---------------------------
# Log parsing regex (libFuzzer/Atheris-like)
# ---------------------------
COV_RE = re.compile(r"\bcov:\s*([0-9]+)")
FT_RE = re.compile(r"\bft:\s*([0-9]+)")
EXECS_RE = re.compile(r"\bexec/s:\s*([0-9]+(?:\.[0-9]+)?)([kKmM]?)")


def _parse_num_with_suffix(x: str, suffix: str) -> float:
    v = float(x)
    if suffix.lower() == "k":
        return v * 1_000.0
    if suffix.lower() == "m":
        return v * 1_000_000.0
    return v


def parse_fuzzer_log(log_path: Path) -> Dict[str, Optional[float]]:
    text = log_path.read_text(errors="ignore")
    covs = [int(x) for x in COV_RE.findall(text)]
    fts = [int(x) for x in FT_RE.findall(text)]
    exec_s = None
    hits = EXECS_RE.findall(text)
    if hits:
        x, suf = hits[-1]
        exec_s = _parse_num_with_suffix(x, suf)
    return {
        "cov_first": covs[0] if covs else None,
        "cov_last": covs[-1] if covs else None,
        "ft_first": fts[0] if fts else None,
        "ft_last": fts[-1] if fts else None,
        "exec_s_last": exec_s,
    }


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


# ---------------------------
# Data model
# ---------------------------
@dataclass
class ProfileArm:
    profile_id: str
    profile: Dict[str, object]


@dataclass
class HarnessCandidate:
    harness_id: str
    harness_path: Path
    profiles: List[ProfileArm]


@dataclass
class StepResult:
    t: int
    harness_id: str
    profile_id: str
    delta_ft: int
    delta_cov: int
    exec_s: float
    proxy_reward: float
    fast_reward: float
    delta_files_epoch: int

    audited_harnesses: int
    slow_harness: Optional[int]          # harness-level (true global increment attribution per harness audit)
    slow_profile_credit: Optional[float] # profile-level CREDIT (distributed, may be float)


# ---------------------------
# Fuzz epoch runner (fuzz env)
# ---------------------------
def run_one_epoch(
    *,
    python: str,
    harness_path: Path,
    corpus_dir: Path,
    crash_dir: Path,
    log_path: Path,
    epoch_sec: int,
    fuzz_flags: List[str],
    profile_env: Dict[str, str],
) -> None:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    crash_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(profile_env)

    cmd = [
        python,
        str(harness_path),
        str(corpus_dir),
        f"-artifact_prefix={str(crash_dir)}/",
        f"-max_total_time={int(epoch_sec)}",
        *fuzz_flags,
    ]
    with log_path.open("w", encoding="utf-8", errors="ignore") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=lf, env=env)
        try:
            proc.wait(timeout=epoch_sec + 20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


# ---------------------------
# Corpus snapshot/manifest utils
# ---------------------------
def _iter_corpus_files(corpus_dir: Path) -> List[Tuple[str, int, int]]:
    items: List[Tuple[str, int, int]] = []
    if not corpus_dir.exists():
        return items
    for p in corpus_dir.rglob("*"):
        if not p.is_file():
            continue
        rel = str(p.relative_to(corpus_dir))
        st = p.stat()
        items.append((rel, int(st.st_size), int(st.st_mtime_ns)))
    items.sort(key=lambda x: x[0])
    return items


def load_manifest(path: Path) -> Dict[str, Dict[str, int]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(path: Path, manifest: Dict[str, Dict[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def build_manifest_from_items(items: List[Tuple[str, int, int]]) -> Dict[str, Dict[str, int]]:
    m: Dict[str, Dict[str, int]] = {}
    for rel, size, mtime_ns in items:
        m[rel] = {"size": int(size), "mtime_ns": int(mtime_ns)}
    return m


def diff_manifest(old: Dict[str, Dict[str, int]], new: Dict[str, Dict[str, int]]) -> List[str]:
    delta: List[str] = []
    for rel, meta in new.items():
        prev = old.get(rel)
        if prev is None:
            delta.append(rel)
        else:
            if int(prev.get("size", -1)) != int(meta.get("size", -2)) or int(prev.get("mtime_ns", -1)) != int(meta.get("mtime_ns", -2)):
                delta.append(rel)
    return delta


def _safe_tag_name(profile_id: str, basename: str) -> str:
    if basename.startswith(profile_id + "__"):
        return basename
    return f"{profile_id}__{basename}"


def tag_delta_files_with_profile(
    corpus_dir: Path,
    delta_relpaths: List[str],
    profile_id: str,
) -> List[str]:
    """
    Rename newly created/changed files by prefixing profile id for attribution.
    Returns new relpaths after rename.
    """
    new_rels: List[str] = []
    for rel in delta_relpaths:
        src = corpus_dir / rel
        if not src.exists() or not src.is_file():
            continue
        parent = src.parent
        base = src.name
        new_base = _safe_tag_name(profile_id, base)
        if new_base == base:
            new_rels.append(rel)
            continue

        dst = parent / new_base
        if dst.exists():
            stem = dst.stem
            suf = dst.suffix
            i = 0
            while dst.exists():
                i += 1
                dst = parent / f"{stem}__r{i}{suf}"
        src.rename(dst)
        new_rels.append(str(dst.relative_to(corpus_dir)))
    return new_rels


def update_current_manifest_and_tag_epoch_delta(
    *,
    corpus_dir: Path,
    current_manifest_path: Path,
    epoch_profile_id: str,
) -> List[str]:
    """
    1) Read old current manifest
    2) Snapshot current corpus => compute epoch-delta relpaths
    3) Tag epoch-delta files with profile prefix (rename)
    4) Resnapshot corpus & write NEW current manifest
    5) Return tagged epoch-delta relpaths
    """
    old = load_manifest(current_manifest_path)
    items_before_tag = _iter_corpus_files(corpus_dir)
    new_m_before = build_manifest_from_items(items_before_tag)
    epoch_delta = diff_manifest(old, new_m_before)

    tagged_delta = tag_delta_files_with_profile(corpus_dir, epoch_delta, epoch_profile_id)

    items_after = _iter_corpus_files(corpus_dir)
    new_m_after = build_manifest_from_items(items_after)
    save_manifest(current_manifest_path, new_m_after)

    return tagged_delta


def diff_audit_window_delta(
    *,
    audit_base_manifest_path: Path,
    current_manifest_path: Path,
) -> List[str]:
    base = load_manifest(audit_base_manifest_path)
    cur = load_manifest(current_manifest_path)
    return diff_manifest(base, cur)


def advance_audit_base_to_current(
    *,
    audit_base_manifest_path: Path,
    current_manifest_path: Path,
) -> None:
    cur = load_manifest(current_manifest_path)
    save_manifest(audit_base_manifest_path, cur)


# ---------------------------
# materialize corpus subset
# ---------------------------
def materialize_subset_corpus(
    corpus_dir: Path,
    relpaths: List[str],
    out_dir: Path,
    *,
    max_inputs: int = 2000,
) -> int:
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    rels = list(relpaths)

    # cap by newest mtime
    if max_inputs > 0 and len(rels) > max_inputs:
        tmp: List[Tuple[int, str]] = []
        for rel in rels:
            src = corpus_dir / rel
            if not src.exists():
                continue
            try:
                tmp.append((int(src.stat().st_mtime_ns), rel))
            except FileNotFoundError:
                continue
        tmp.sort(key=lambda x: x[0], reverse=True)
        rels = [r for _, r in tmp[:max_inputs]]

    count = 0
    for rel in rels:
        src = corpus_dir / rel
        if not src.exists() or not src.is_file():
            continue
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        count += 1
    return count


# ---------------------------
# Cov env audit via subprocess
# ---------------------------
def _extract_json_from_stdout(stdout: str) -> Dict[str, Any]:
    s = stdout.strip()
    if not s:
        raise ValueError("empty stdout from cov audit script")
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        idx = s.rfind("{")
        if idx == -1:
            raise
        return json.loads(s[idx:])


def run_cov_audit_in_cov_env(
    *,
    cov_venv_activate: Path,
    cov_audit_script: Path,
    harness_path: Path,
    corpus_dir: Path,
    work_dir: Path,
    global_dir: Path,
    primary_object: str,
    extra_objects: List[str],
    ignore_filename_regex: Optional[str],
    replay_extra: str,
) -> Dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    global_dir.mkdir(parents=True, exist_ok=True)

    parts: List[str] = []
    parts.append("source " + shlex.quote(str(cov_venv_activate)))
    parts.append("python3 " + shlex.quote(str(cov_audit_script)))
    parts.append("--python python3")
    parts.append("--harness " + shlex.quote(str(harness_path)))
    parts.append("--corpus " + shlex.quote(str(corpus_dir)))
    parts.append("--work_dir " + shlex.quote(str(work_dir)))
    parts.append("--global_dir " + shlex.quote(str(global_dir)))
    parts.append("--primary_object " + shlex.quote(primary_object))
    for obj in extra_objects:
        parts.append("--extra_object " + shlex.quote(obj))
    if ignore_filename_regex:
        parts.append("--ignore_filename_regex " + shlex.quote(ignore_filename_regex))
    if replay_extra:
        parts.append("--replay_extra " + shlex.quote(replay_extra))

    cmd_str = " && ".join([parts[0], " ".join(parts[1:])])
    cmd = ["bash", "-lc", cmd_str]

    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            "cov audit failed:\n"
            f"  cmd: {cmd}\n"
            f"  rc: {p.returncode}\n"
            f"  stdout:\n{p.stdout}\n"
            f"  stderr:\n{p.stderr}\n"
        )
    return _extract_json_from_stdout(p.stdout)


# ---------------------------
# Loading harness candidates
# ---------------------------
def _load_profiles_from_top_json(top_json: Path) -> List[ProfileArm]:
    top = json.loads(top_json.read_text(encoding="utf-8"))
    arms: List[ProfileArm] = []
    for r in top:
        arms.append(ProfileArm(profile_id=r["profile_id"], profile=r["profile"]))
    return arms


def load_harness_candidates(args) -> List[HarnessCandidate]:
    if args.harnesses_json:
        data = json.loads(Path(args.harnesses_json).read_text(encoding="utf-8"))
        cands: List[HarnessCandidate] = []
        for h in data:
            hid = str(h["harness_id"])
            hpath = Path(h["harness_path"]).resolve()
            profiles = [ProfileArm(profile_id=p["profile_id"], profile=p["profile"]) for p in h["profiles"]]
            if profiles:
                cands.append(HarnessCandidate(harness_id=hid, harness_path=hpath, profiles=profiles))
        return cands

    # legacy: single harness + top_json
    hpath = Path(args.harness).resolve()
    hid = args.harness_id or hpath.stem
    profiles = _load_profiles_from_top_json(Path(args.top_json).resolve())
    return [HarnessCandidate(harness_id=hid, harness_path=hpath, profiles=profiles)]


# ---------------------------
# Attribution helpers
# ---------------------------
def producer_from_relpath(rel: str) -> Optional[str]:
    base = Path(rel).name
    if "__" not in base:
        return None
    pid, _ = base.split("__", 1)
    return pid or None


def group_relpaths_by_profile(relpaths: List[str]) -> Dict[str, List[str]]:
    buckets: DefaultDict[str, List[str]] = defaultdict(list)
    for rel in relpaths:
        pid = producer_from_relpath(rel) or "unknown"
        buckets[pid].append(rel)
    return dict(buckets)


def _select_topk_profiles_by_count(
    buckets: Dict[str, List[str]],
    *,
    topk: int,
) -> List[Tuple[str, int]]:
    items: List[Tuple[str, int]] = []
    for pid, rels in buckets.items():
        if pid in ("unknown", "other"):
            continue
        items.append((pid, len(rels)))
    items.sort(key=lambda x: x[1], reverse=True)
    if topk > 0:
        items = items[:topk]
    return items


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()

    # Inputs
    ap.add_argument("--harnesses_json", default=None)
    ap.add_argument("--harness", default=None)
    ap.add_argument("--harness_id", default=None)
    ap.add_argument("--top_json", default=None)

    # Common
    ap.add_argument("--root", default="fuzz_output")
    ap.add_argument("--python", default=os.sys.executable)
    ap.add_argument("--epoch", type=int, default=60)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--audit_every", type=int, default=10)
    ap.add_argument("--fuzz_flags", default="-ignore_timeouts=1 -rss_limit_mb=4096 -use_value_profile=1 -entropic=1")
    ap.add_argument("--mix", type=float, default=0.7)

    # Dual-channel UCB params
    ap.add_argument("--c_fast", type=float, default=2.0)
    ap.add_argument("--c_slow", type=float, default=2.0)
    ap.add_argument("--epsilon_harness", type=float, default=0.02)
    ap.add_argument("--epsilon_profile", type=float, default=0.05)
    ap.add_argument("--alpha_min", type=float, default=0.2)

    # Soft elimination => cooldown params (bandit_core must support these if you pass them)
    ap.add_argument("--elim_margin", type=float, default=0.0)
    ap.add_argument("--elim_patience", type=int, default=3)
    ap.add_argument("--elim_min_pulls", type=int, default=8)
    ap.add_argument("--cooldown_steps", type=int, default=50)

    # Manifests
    ap.add_argument("--manifest_dir", default="manifests")

    # Audit control
    ap.add_argument("--full_corpus_audit", action="store_true",
                    help="Audit full corpus since last audit. Usually keep OFF to use audit window delta.")
    ap.add_argument("--audit_max_inputs", type=int, default=2000)
    ap.add_argument("--audit_profile_topk", type=int, default=5,
                    help="When distributing slow credit, only top-k producer profiles by count are credited (0=all).")

    # Cov env + audit script
    ap.add_argument("--cov_venv_activate", default="/root/pytorch_cov/bin/activate")
    ap.add_argument("--cov_audit_script", required=True)
    ap.add_argument("--global_dir", default="global_union",
                    help="TRUE global union directory shared across all harnesses (global.profdata).")
    ap.add_argument("--primary_object", required=True)
    ap.add_argument("--extra_object", action="append", default=[])
    ap.add_argument("--ignore_filename_regex", default=None)
    ap.add_argument("--cov_replay_extra", default="")

    # Slow metric
    ap.add_argument("--slow_metric", choices=["BRH", "LH", "FNH"], default="BRH")

    # Optional: penalize zero-contribution profile when it produced enough inputs
    ap.add_argument("--min_credit_inputs", type=int, default=20)
    ap.add_argument("--zero_slow_penalty", type=float, default=0.0,
                    help="If slow increment for a harness audit is 0 and a profile produced >=min_credit_inputs "
                         "in the audit window, apply negative slow update to that profile (default 0 disables).")

    # Repro
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    if not args.harnesses_json:
        if not args.harness or not args.top_json:
            raise SystemExit("Need --harnesses_json OR legacy (--harness and --top_json)")

    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    candidates = load_harness_candidates(args)
    if not candidates:
        raise SystemExit("no harness candidates loaded")

    fuzz_flags = [x for x in args.fuzz_flags.split(" ") if x.strip()]
    cov_venv_activate = Path(args.cov_venv_activate).resolve()
    cov_audit_script = Path(args.cov_audit_script).resolve()

    manifest_root = Path(args.manifest_dir)
    if not manifest_root.is_absolute():
        manifest_root = (root / manifest_root).resolve()
    manifest_root.mkdir(parents=True, exist_ok=True)

    global_root = Path(args.global_dir)
    if not global_root.is_absolute():
        global_root = (root / global_root).resolve()
    global_root.mkdir(parents=True, exist_ok=True)

    # candidates index
    harness_ids = [c.harness_id for c in candidates]
    harness_path_by_id: Dict[str, Path] = {c.harness_id: c.harness_path for c in candidates}
    profiles_by_harness: Dict[str, List[ProfileArm]] = {c.harness_id: c.profiles for c in candidates}

    # Bandits
    harness_bandit = DualChannelUCBSoftElim(
        c_fast=args.c_fast, c_slow=args.c_slow,
        epsilon=args.epsilon_harness,
        elim_margin=args.elim_margin,
        elim_patience=args.elim_patience,
        elim_min_pulls=args.elim_min_pulls,
        alpha_min=args.alpha_min,
        cooldown_steps=args.cooldown_steps,
        seed=args.seed,
    )

    profile_bandits: Dict[str, DualChannelUCBSoftElim] = {}
    for hid in harness_ids:
        profile_bandits[hid] = DualChannelUCBSoftElim(
            c_fast=args.c_fast, c_slow=args.c_slow,
            epsilon=args.epsilon_profile,
            elim_margin=args.elim_margin,
            elim_patience=args.elim_patience,
            elim_min_pulls=args.elim_min_pulls,
            alpha_min=args.alpha_min,
            cooldown_steps=args.cooldown_steps,
            seed=args.seed + (hash(hid) & 0xFFFF),
        )

    results: List[StepResult] = []

    for t in range(1, args.steps + 1):
        # 1) select harness
        hid = harness_bandit.select(harness_ids)
        harness_path = harness_path_by_id[hid]

        # 2) select profile within harness
        prof_arms = profiles_by_harness[hid]
        prof_ids = [p.profile_id for p in prof_arms]
        pb = profile_bandits[hid]
        pid = pb.select(prof_ids)
        arm = next(p for p in prof_arms if p.profile_id == pid)

        profile_env = {k: str(v) for k, v in arm.profile.items()}
        profile_env.setdefault("OMP_NUM_THREADS", "1")
        profile_env.setdefault("MKL_NUM_THREADS", "1")
        profile_env.setdefault("TORCH_NUM_THREADS", "1")

        # dirs (shared corpus per harness)
        corpus_dir = root / "corpus" / hid
        run_dir = root / "runs" / hid / pid / f"t{t:04d}"
        crash_dir = run_dir / "crash"
        log_path = run_dir / "fuzzer.log"
        run_dir.mkdir(parents=True, exist_ok=True)

        # manifests
        cur_manifest = manifest_root / hid / "current.json"
        audit_base_manifest = manifest_root / hid / "audit_base.json"
        cur_manifest.parent.mkdir(parents=True, exist_ok=True)

        # 3) fuzz epoch
        run_one_epoch(
            python=args.python,
            harness_path=harness_path,
            corpus_dir=corpus_dir,
            crash_dir=crash_dir,
            log_path=log_path,
            epoch_sec=args.epoch,
            fuzz_flags=fuzz_flags,
            profile_env=profile_env,
        )

        # 4) update current manifest + tag this epoch delta with pid
        epoch_delta_relpaths = update_current_manifest_and_tag_epoch_delta(
            corpus_dir=corpus_dir,
            current_manifest_path=cur_manifest,
            epoch_profile_id=pid,
        )
        (run_dir / "epoch_delta_files.json").write_text(json.dumps(epoch_delta_relpaths, indent=2), encoding="utf-8")
        delta_files_epoch = len(epoch_delta_relpaths)

        # 5) fast reward (proxy + seed evolution)
        p = parse_fuzzer_log(log_path)
        cov_first, cov_last = p["cov_first"], p["cov_last"]
        ft_first, ft_last = p["ft_first"], p["ft_last"]
        exec_s = float(p["exec_s_last"] or 1.0)

        delta_cov = int((cov_last - cov_first) if (cov_first is not None and cov_last is not None) else 0)
        delta_ft = int((ft_last - ft_first) if (ft_first is not None and ft_last is not None) else 0)

        proxy_reward = float(compute_proxy_reward(delta_ft, delta_cov, exec_s, mix=args.mix))
        fast_reward = float(compute_fast_reward(proxy_reward, delta_files_epoch))

        # update fast both levels
        pb.update_fast(pid, fast_reward)
        harness_bandit.update_fast(hid, fast_reward)

        # 6) slow audit window (low frequency)
        audited_harnesses = 0
        slow_harness_selected: Optional[int] = None
        slow_profile_credit_selected: Optional[float] = None

        do_audit = (args.audit_every > 0 and (t % args.audit_every == 0))
        if do_audit:
            # audit all harnesses that have delta since last audit
            for ahid in harness_ids:
                acorpus = root / "corpus" / ahid
                acur = manifest_root / ahid / "current.json"
                abase = manifest_root / ahid / "audit_base.json"
                if not acur.exists():
                    continue

                # bootstrap audit base on first audit (no cost)
                if not abase.exists():
                    advance_audit_base_to_current(audit_base_manifest_path=abase, current_manifest_path=acur)
                    continue

                # compute audit window delta
                if args.full_corpus_audit:
                    curm = load_manifest(acur)
                    window_delta = list(curm.keys())
                else:
                    window_delta = diff_audit_window_delta(audit_base_manifest_path=abase, current_manifest_path=acur)

                if not window_delta:
                    continue

                # materialize ONE audit corpus (cap inside helper)
                audit_root = root / "audits" / ahid / f"t{t:04d}"
                audit_corpus_dir = audit_root / "window_corpus"
                audited_inputs = materialize_subset_corpus(
                    acorpus, window_delta, audit_corpus_dir, max_inputs=args.audit_max_inputs
                )
                if audited_inputs <= 0:
                    # still advance base to avoid re-auditing same window endlessly
                    advance_audit_base_to_current(audit_base_manifest_path=abase, current_manifest_path=acur)
                    continue

                # run cov audit ONCE (true global union)
                audit_json = run_cov_audit_in_cov_env(
                    cov_venv_activate=cov_venv_activate,
                    cov_audit_script=cov_audit_script,
                    harness_path=harness_path_by_id[ahid],
                    corpus_dir=audit_corpus_dir,
                    work_dir=audit_root / "work",
                    global_dir=global_root,  # TRUE global union shared
                    primary_object=args.primary_object,
                    extra_objects=args.extra_object,
                    ignore_filename_regex=args.ignore_filename_regex,
                    replay_extra=args.cov_replay_extra,
                )
                delta = audit_json.get("delta", {}) or {}
                slow_h = int(delta.get(args.slow_metric, 0)) if isinstance(delta, dict) else 0

                # update harness slow
                harness_bandit.update_slow(ahid, float(slow_h))

                # distribute slow credit to profiles by attribution counts (top-k optional)
                buckets = group_relpaths_by_profile(window_delta)
                top_items = _select_topk_profiles_by_count(buckets, topk=int(args.audit_profile_topk))

                # denom = sum counts among credited profiles
                denom = sum(cnt for _p, cnt in top_items)
                pb2 = profile_bandits[ahid]

                # if slow_h > 0: proportional credit
                if slow_h > 0 and denom > 0:
                    for p2, cnt in top_items:
                        credit = float(slow_h) * (float(cnt) / float(denom))
                        pb2.update_slow(p2, credit)
                        if ahid == hid and p2 == pid:
                            slow_profile_credit_selected = credit
                else:
                    # slow_h == 0: optional penalty for heavy producers
                    if args.zero_slow_penalty > 0.0:
                        for p2, cnt in top_items:
                            if int(cnt) >= int(args.min_credit_inputs):
                                pb2.update_slow(p2, -float(args.zero_slow_penalty))
                                if ahid == hid and p2 == pid:
                                    slow_profile_credit_selected = -float(args.zero_slow_penalty)

                audited_harnesses += 1

                # advance audit base to current (commit window)
                advance_audit_base_to_current(audit_base_manifest_path=abase, current_manifest_path=acur)

                # capture selected ones for logging (only the harness chosen this step)
                if ahid == hid:
                    slow_harness_selected = slow_h
                    # if this pid not in credited topk, keep None (or 0)
                    if slow_profile_credit_selected is None and slow_h > 0:
                        # if user wants explicit 0 when not in topk:
                        slow_profile_credit_selected = 0.0

            # after auditing, run soft elimination once
            harness_bandit.maybe_soft_eliminate(harness_ids)
            for ahid in harness_ids:
                profile_bandits[ahid].maybe_soft_eliminate([p.profile_id for p in profiles_by_harness[ahid]])

        # record
        sr = StepResult(
            t=t,
            harness_id=hid,
            profile_id=pid,
            delta_ft=delta_ft,
            delta_cov=delta_cov,
            exec_s=exec_s,
            proxy_reward=proxy_reward,
            fast_reward=fast_reward,
            delta_files_epoch=delta_files_epoch,
            audited_harnesses=audited_harnesses,
            slow_harness=slow_harness_selected,
            slow_profile_credit=slow_profile_credit_selected,
        )
        results.append(sr)

        # print
        hu, hl = harness_bandit.ucb_lcb(hid)
        pu, pl = pb.ucb_lcb(pid)

        print(
            f"[t={t:04d}] harness={hid} profile={pid} "
            f"Δft={delta_ft} Δcov={delta_cov} exec/s={exec_s:.1f} "
            f"proxy={proxy_reward:.3f} fast={fast_reward:.3f} delta_files={delta_files_epoch} "
            + (f"| audit_h={audited_harnesses} slow_{args.slow_metric}(H/P)=({slow_harness_selected}/{slow_profile_credit_selected}) "
               if do_audit else "")
            + f"| H(UCB/LCB)=({hu:.3f}/{hl:.3f}) active={harness_bandit.is_active(hid)} "
              f"P(UCB/LCB)=({pu:.3f}/{pl:.3f}) active={pb.is_active(pid)}"
        )

        # persist state
        out_state = {
            "harness_bandit": harness_bandit.to_jsonable(),
            "profile_bandits": {x: profile_bandits[x].to_jsonable() for x in profile_bandits},
            "results_tail": [asdict(x) for x in results[-50:]],
        }
        (root / "state").mkdir(parents=True, exist_ok=True)
        (root / "state" / "bandit_state.json").write_text(json.dumps(out_state, indent=2), encoding="utf-8")

    final = root / "state" / "bandit_all_results.json"
    final.write_text(json.dumps([asdict(x) for x in results], indent=2), encoding="utf-8")
    print(f"[+] wrote {final}")
    print(f"[+] global union dir: {global_root}")
    print(f"[+] manifest root: {manifest_root}")
    print(f"[+] corpus root: {root / 'corpus'}")


if __name__ == "__main__":
    main()
