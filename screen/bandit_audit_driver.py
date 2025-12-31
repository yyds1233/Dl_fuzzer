#!/usr/bin/env python3
# bandit_audit_driver.py
# - Fuzz env: run harness with profiles, parse Δft/Δcov/exec/s, use UCB1 on proxy reward.
# - Cov env (separate venv): low-frequency audit by replaying ONLY new/changed corpus inputs,
#   updating global.profdata union, and reading totals delta (e.g., ΔBRH) as slow reward.
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
from typing import Dict, List, Optional, Tuple, Any


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


# ---------------------------
# Bandit (UCB1)
# ---------------------------
@dataclass
class Arm:
    profile_id: str
    profile: Dict[str, object]


class UCB1:
    def __init__(self, c: float = 2.0):
        self.c = c
        self.n: Dict[str, int] = {}
        self.mean: Dict[str, float] = {}
        self.t = 0

    def select(self, arms: List[Arm]) -> Arm:
        # Ensure each arm is tried at least once
        for a in arms:
            if self.n.get(a.profile_id, 0) == 0:
                return a

        self.t += 1
        best_arm: Optional[Arm] = None
        best_score = -1e99
        for a in arms:
            pid = a.profile_id
            n = self.n[pid]
            mu = self.mean[pid]
            score = mu + self.c * math.sqrt(math.log(self.t + 1.0) / n)
            if score > best_score:
                best_score = score
                best_arm = a
        assert best_arm is not None
        return best_arm

    def update(self, profile_id: str, reward: float) -> None:
        n = self.n.get(profile_id, 0) + 1
        old = self.mean.get(profile_id, 0.0)
        new = old + (reward - old) / n
        self.n[profile_id] = n
        self.mean[profile_id] = new


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

    # IMPORTANT: DO NOT set LLVM_PROFILE_FILE here (this is fuzz env; cov env is separate)
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
# Corpus delta tracking (manifest)
# ---------------------------
def _iter_corpus_files(corpus_dir: Path) -> List[Tuple[str, int, int]]:
    """
    Return sorted list of (relpath, size, mtime_ns).
    Using size+mtime_ns avoids hashing full files (fast).
    """
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


def update_manifest_and_get_delta(corpus_dir: Path, manifest_path: Path) -> List[str]:
    """
    Returns relpaths that are new or changed since last snapshot.
    Then writes new snapshot.
    """
    old = load_manifest(manifest_path)
    cur_items = _iter_corpus_files(corpus_dir)

    delta: List[str] = []
    new_manifest: Dict[str, Dict[str, int]] = {}

    for rel, size, mtime_ns in cur_items:
        new_manifest[rel] = {"size": size, "mtime_ns": mtime_ns}
        prev = old.get(rel)
        if prev is None:
            delta.append(rel)
        else:
            if int(prev.get("size", -1)) != size or int(prev.get("mtime_ns", -1)) != mtime_ns:
                delta.append(rel)

    save_manifest(manifest_path, new_manifest)
    return delta


def materialize_delta_corpus(
    corpus_dir: Path,
    delta_relpaths: List[str],
    out_dir: Path,
    *,
    max_inputs: int = 2000,
) -> int:
    """
    Create out_dir containing only delta inputs.
    Prefer hardlink; fallback to copy.
    Optionally cap number of inputs (pick newest by mtime_ns).
    """
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    rels = delta_relpaths[:]
    if max_inputs > 0 and len(rels) > max_inputs:
        tmp: List[Tuple[int, str]] = []
        for rel in rels:
            src = corpus_dir / rel
            try:
                tmp.append((int(src.stat().st_mtime_ns), rel))
            except FileNotFoundError:
                continue
        tmp.sort(key=lambda x: x[0], reverse=True)
        rels = [rel for _, rel in tmp[:max_inputs]]

    count = 0
    for rel in rels:
        src = corpus_dir / rel
        if not src.exists() or not src.is_file():
            continue
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, dst)  # hardlink (fast)
        except OSError:
            shutil.copy2(src, dst)  # fallback
        count += 1
    return count


# ---------------------------
# Cov env audit via subprocess
# ---------------------------
def _extract_json_from_stdout(stdout: str) -> Dict[str, Any]:
    """
    cov_global_union_audit.py should print JSON to stdout.
    To be robust, we parse from the last '{' occurrence.
    """
    s = stdout.strip()
    if not s:
        raise ValueError("empty stdout from cov audit script")
    # Try direct parse
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # Try parse last JSON object
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
    """
    Runs in a fresh shell:
      source <cov_venv_activate>
      python3 <cov_audit_script> --harness ... --corpus ... --work_dir ... --global_dir ...
    Returns parsed JSON.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    global_dir.mkdir(parents=True, exist_ok=True)

    # Build a safe shell command string
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
# Results
# ---------------------------
@dataclass
class StepResult:
    t: int
    profile_id: str
    delta_ft: int
    delta_cov: int
    exec_s: float
    proxy_reward: float
    delta_files: int
    audited_inputs: int
    slow_delta: Optional[Dict[str, int]]  # e.g. {"BRH": +12, "LH": +30}
    slow_reward: Optional[int]            # default: ΔBRH


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()

    # Fuzz side
    ap.add_argument("--harness", required=True, help="path to fuzz harness.py")
    ap.add_argument("--top_json", required=True, help="round1_top_results.json")
    ap.add_argument("--root", default="fuzz_output")
    ap.add_argument("--python", default=os.sys.executable)
    ap.add_argument("--epoch", type=int, default=60)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--audit_every", type=int, default=10)

    ap.add_argument("--fuzz_flags", default="-ignore_timeouts=1 -rss_limit_mb=4096 -use_value_profile=1 -entropic=1")
    ap.add_argument("--mix", type=float, default=0.7, help="proxy mix weight on ft (0..1)")
    ap.add_argument("--ucb_c", type=float, default=2.0)

    # Manifest/delta audit control
    ap.add_argument("--manifest_dir", default="bandit_manifest", help="relative to --root unless absolute")
    ap.add_argument("--full_corpus_audit", action="store_true",
                    help="if set, audit replays full corpus; otherwise replay only new/changed inputs (recommended)")
    ap.add_argument("--audit_max_inputs", type=int, default=2000,
                    help="cap number of inputs replayed per audit (delta-only mode)")

    # Cov env + audit script (global.profdata union)
    ap.add_argument("--cov_venv_activate", default="/root/pytorch_cov/bin/activate")
    ap.add_argument("--cov_audit_script", required=True, help="path to cov_global_union_audit.py")
    ap.add_argument("--global_dir", default="global_union", help="relative to --root unless absolute")
    ap.add_argument("--primary_object", required=True, help="instrumented .so/.bin used by llvm-cov in cov env")
    ap.add_argument("--extra_object", action="append", default=[], help="repeatable -object for llvm-cov")
    ap.add_argument("--ignore_filename_regex", default=None, help="pass to llvm-cov -ignore-filename-regex")
    ap.add_argument("--cov_replay_extra", default="", help="extra args for replay in cov env (space-separated)")

    # Slow reward selection
    ap.add_argument("--slow_metric", choices=["BRH", "LH", "FNH"], default="BRH",
                    help="which delta field to use as slow reward (default: ΔBRH)")

    args = ap.parse_args()

    root = Path(args.root).resolve()
    harness_path = Path(args.harness).resolve()
    top_json = Path(args.top_json).resolve()

    # Load arms
    top = json.loads(top_json.read_text(encoding="utf-8"))
    arms: List[Arm] = [Arm(profile_id=r["profile_id"], profile=r["profile"]) for r in top]
    if not arms:
        raise SystemExit("no arms from top_json")

    fuzz_flags = [x for x in args.fuzz_flags.split(" ") if x.strip()]
    bandit = UCB1(c=args.ucb_c)

    # Paths
    cov_venv_activate = Path(args.cov_venv_activate).resolve()
    cov_audit_script = Path(args.cov_audit_script).resolve()

    manifest_dir = Path(args.manifest_dir)
    if not manifest_dir.is_absolute():
        manifest_dir = (root / manifest_dir).resolve()

    global_dir = Path(args.global_dir)
    if not global_dir.is_absolute():
        global_dir = (root / global_dir).resolve()

    results: List[StepResult] = []

    for t in range(1, args.steps + 1):
        arm = bandit.select(arms)

        # dirs (fuzz)
        run_dir = root / "bandit_runs" / arm.profile_id / f"t{t:04d}"
        corpus_dir = root / "bandit_corpus" / arm.profile_id
        crash_dir = run_dir / "crash"
        log_path = run_dir / "fuzzer.log"
        run_dir.mkdir(parents=True, exist_ok=True)

        # profile env
        profile_env = {k: str(v) for k, v in arm.profile.items()}
        profile_env.setdefault("OMP_NUM_THREADS", "1")
        profile_env.setdefault("MKL_NUM_THREADS", "1")
        profile_env.setdefault("TORCH_NUM_THREADS", "1")

        # 1) fuzz epoch
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

        # 2) update manifest & compute delta files
        manifest_path = manifest_dir / f"{arm.profile_id}.json"
        delta_relpaths = update_manifest_and_get_delta(corpus_dir, manifest_path)
        (run_dir / "delta_files.json").write_text(json.dumps(delta_relpaths, indent=2), encoding="utf-8")
        delta_files_count = len(delta_relpaths)

        # 3) parse proxy
        p = parse_fuzzer_log(log_path)
        cov_first, cov_last = p["cov_first"], p["cov_last"]
        ft_first, ft_last = p["ft_first"], p["ft_last"]
        exec_s = float(p["exec_s_last"] or 1.0)

        delta_cov = int((cov_last - cov_first) if (cov_first is not None and cov_last is not None) else 0)
        delta_ft = int((ft_last - ft_first) if (ft_first is not None and ft_last is not None) else 0)

        proxy_reward = compute_proxy_reward(delta_ft, delta_cov, exec_s, mix=args.mix)
        bandit.update(arm.profile_id, proxy_reward)

        # 4) slow audit (cov env) - low frequency
        slow_delta: Optional[Dict[str, int]] = None
        slow_reward: Optional[int] = None
        audited_inputs = 0

        do_audit = (args.audit_every > 0 and (t % args.audit_every == 0))
        if do_audit:
            if (not args.full_corpus_audit) and delta_files_count == 0:
                # nothing new => skip expensive replay/cov
                slow_delta = {"BRH": 0, "BRF": 0, "LH": 0, "LF": 0, "FNH": 0, "FNF": 0}
                slow_reward = 0
                audited_inputs = 0
            else:
                if args.full_corpus_audit:
                    audit_corpus_dir = corpus_dir
                    audited_inputs = -1  # means full corpus
                else:
                    audit_corpus_dir = run_dir / "audit_delta_corpus"
                    audited_inputs = materialize_delta_corpus(
                        corpus_dir, delta_relpaths, audit_corpus_dir, max_inputs=args.audit_max_inputs
                    )
                    if audited_inputs == 0:
                        slow_delta = {"BRH": 0, "BRF": 0, "LH": 0, "LF": 0, "FNH": 0, "FNF": 0}
                        slow_reward = 0

                if slow_reward is None:
                    audit_work_dir = root / "audits" / arm.profile_id / f"t{t:04d}"
                    audit_json = run_cov_audit_in_cov_env(
                        cov_venv_activate=cov_venv_activate,
                        cov_audit_script=cov_audit_script,
                        harness_path=harness_path,
                        corpus_dir=audit_corpus_dir,
                        work_dir=audit_work_dir,
                        global_dir=global_dir,
                        primary_object=args.primary_object,
                        extra_objects=args.extra_object,
                        ignore_filename_regex=args.ignore_filename_regex,
                        replay_extra=args.cov_replay_extra,
                    )
                    slow_delta = audit_json.get("delta", None)
                    if isinstance(slow_delta, dict):
                        slow_reward = int(slow_delta.get(args.slow_metric, 0))
                    else:
                        slow_reward = 0

        # 5) record
        sr = StepResult(
            t=t,
            profile_id=arm.profile_id,
            delta_ft=delta_ft,
            delta_cov=delta_cov,
            exec_s=exec_s,
            proxy_reward=float(proxy_reward),
            delta_files=delta_files_count,
            audited_inputs=audited_inputs,
            slow_delta=slow_delta,
            slow_reward=slow_reward,
        )
        results.append(sr)

        print(
            f"[t={t:04d}] profile={arm.profile_id} "
            f"Δft={delta_ft} Δcov={delta_cov} exec/s={exec_s:.1f} proxy={proxy_reward:.3f} "
            f"delta_files={delta_files_count}"
            + (f" | audit_inputs={'full' if audited_inputs==-1 else audited_inputs} slow_{args.slow_metric}={slow_reward}"
               if do_audit else "")
        )

        # 6) persist state
        out_state = {
            "bandit": {"n": bandit.n, "mean": bandit.mean, "t": bandit.t, "c": bandit.c},
            "results_tail": [asdict(x) for x in results[-50:]],
        }
        (root / "bandit_runs").mkdir(parents=True, exist_ok=True)
        (root / "bandit_runs" / "bandit_state.json").write_text(json.dumps(out_state, indent=2), encoding="utf-8")

    final = root / "bandit_runs" / "bandit_all_results.json"
    final.write_text(json.dumps([asdict(x) for x in results], indent=2), encoding="utf-8")
    print(f"[+] wrote {final}")
    print(f"[+] global union dir: {global_dir}")
    print(f"[+] manifest dir: {manifest_dir}")


if __name__ == "__main__":
    main()
