#!/usr/bin/env python3
# cov_global_union_audit.py
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

# ---------- utils ----------
def run(cmd: List[str], *, env: Optional[Dict[str, str]] = None, cwd: Optional[Path] = None) -> str:
    p = subprocess.run(
        cmd,
        env=env,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if p.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  rc: {p.returncode}\n"
            f"  stdout:\n{p.stdout}\n"
            f"  stderr:\n{p.stderr}\n"
        )
    return p.stdout

def parse_lcov_summary(text: str) -> Dict[str, int]:
    """
    For lcov trace, totals can appear per file; we sum across files:
      LH/LF (lines hit/found)
      FNH/FNF (functions hit/found)
      BRH/BRF (branches hit/found)
    """
    out = {"LH": 0, "LF": 0, "FNH": 0, "FNF": 0, "BRH": 0, "BRF": 0}
    for line in text.splitlines():
        line = line.strip()
        for k in list(out.keys()):
            if line.startswith(k + ":"):
                try:
                    out[k] += int(line.split(":", 1)[1])
                except ValueError:
                    pass
    return out

def diff_totals(new: Dict[str, int], old: Dict[str, int]) -> Dict[str, int]:
    keys = sorted(set(new.keys()) | set(old.keys()))
    return {k: int(new.get(k, 0) - old.get(k, 0)) for k in keys}

def atomic_replace(src: Path, dst: Path) -> None:
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)

# ---------- core ----------
def replay_and_collect_profraw(
    *,
    python: str,
    harness: Path,
    corpus_dir: Path,
    out_dir: Path,
    extra_args: List[str],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    profraw_pattern = str(out_dir / "epoch_%p.profraw")

    env = os.environ.copy()
    env["LLVM_PROFILE_FILE"] = profraw_pattern

    cmd = [python, str(harness), str(corpus_dir), "-runs=0", "-print_final_stats=1", *extra_args]
    run(cmd, env=env)

    return out_dir

# def merge_profraw_to_profdata(llvm_profdata: str, profraw_dir: Path, out_profdata: Path) -> None:
#     glob = str(profraw_dir / "epoch_*.profraw")
#     run([llvm_profdata, "merge", "-sparse", glob, "-o", str(out_profdata)])
def merge_profraw_to_profdata(llvm_profdata: str, profraw_dir: Path, out_profdata: Path) -> None:
    # 1) 在 Python 里展开 glob
    profraws = sorted(profraw_dir.glob("epoch_*.profraw"))

    # 2) 可选：等一小会儿（极少数情况下进程退出后落盘稍慢）
    if not profraws:
        for _ in range(20):  # 最多等 2 秒
            time.sleep(0.1)
            profraws = sorted(profraw_dir.glob("epoch_*.profraw"))
            if profraws:
                break

    if not profraws:
        raise RuntimeError(f"No profraw found in {profraw_dir} (pattern epoch_*.profraw)")

    # 3) 把所有 profraw 文件路径作为参数传给 llvm-profdata
    cmd = [llvm_profdata, "merge", "-sparse", *[str(p) for p in profraws], "-o", str(out_profdata)]
    run(cmd)

def merge_into_global(llvm_profdata: str, global_profdata: Path, epoch_profdata: Path, out_new_global: Path) -> None:
    if global_profdata.exists():
        run([llvm_profdata, "merge", "-sparse", str(global_profdata), str(epoch_profdata), "-o", str(out_new_global)])
    else:
        shutil.copy2(epoch_profdata, out_new_global)

def cov_summary_lcov(
    llvm_cov: str,
    profdata: Path,
    primary_object: Path,
    extra_objects: List[Path],
    ignore_filename_regex: Optional[str],
) -> Dict[str, int]:
    cmd = [
        llvm_cov, "export",
        "-summary-only",
        "-format=lcov",
        f"-instr-profile={str(profdata)}",
        str(primary_object),
    ]
    for obj in extra_objects:
        cmd += ["-object", str(obj)]
    if ignore_filename_regex:
        cmd += [f"-ignore-filename-regex={ignore_filename_regex}"]

    text = run(cmd)
    return parse_lcov_summary(text)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default="python3")
    ap.add_argument("--harness", default="/root/fuzz_output/auto_conv2d.py")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--work_dir", required=True, help="where to put profraw/profdata/logs for this audit")
    ap.add_argument("--global_dir", required=True, help="stores global.profdata + global_totals.json")

    ap.add_argument("--llvm_cov", default="llvm-cov")
    ap.add_argument("--llvm_profdata", default="llvm-profdata")
    ap.add_argument("--primary_object", required=True)
    ap.add_argument("--extra_object", action="append", default=[])

    ap.add_argument("--ignore_filename_regex", default=None)
    ap.add_argument("--replay_extra", default="", help="extra args passed to harness replay (space-separated)")

    args = ap.parse_args()

    harness = Path(args.harness).resolve()
    corpus = Path(args.corpus).resolve()
    work_dir = Path(args.work_dir).resolve()
    global_dir = Path(args.global_dir).resolve()
    global_dir.mkdir(parents=True, exist_ok=True)

    global_prof = global_dir / "global.profdata"
    global_totals_path = global_dir / "global_totals.json"

    primary_obj = Path(args.primary_object).resolve()
    extra_objs = [Path(x).resolve() for x in args.extra_object]

    # 0) load old totals
    if global_totals_path.exists():
        old_totals = json.loads(global_totals_path.read_text(encoding="utf-8"))
    else:
        old_totals = {"LH": 0, "LF": 0, "FNH": 0, "FNF": 0, "BRH": 0, "BRF": 0}

    # 1) replay -> profraw
    extra_args = [x for x in args.replay_extra.split(" ") if x.strip()]
    replay_and_collect_profraw(
        python=args.python,
        harness=harness,
        corpus_dir=corpus,
        out_dir=work_dir,
        extra_args=extra_args,
    )

    # 2) profraw -> epoch.profdata
    epoch_prof = work_dir / "epoch.profdata"
    merge_profraw_to_profdata(args.llvm_profdata, work_dir, epoch_prof)

    # 3) merge into new_global.profdata
    new_global = work_dir / "new_global.profdata"
    merge_into_global(args.llvm_profdata, global_prof, epoch_prof, new_global)

    # 4) compute totals from new_global
    new_totals = cov_summary_lcov(
        llvm_cov=args.llvm_cov,
        profdata=new_global,
        primary_object=primary_obj,
        extra_objects=extra_objs,
        ignore_filename_regex=args.ignore_filename_regex,
    )

    delta = diff_totals(new_totals, old_totals)

    # 5) commit: replace global.profdata + update totals
    atomic_replace(new_global, global_prof)
    global_totals_path.write_text(json.dumps(new_totals, indent=2), encoding="utf-8")

    out = {
        "old_totals": old_totals,
        "new_totals": new_totals,
        "delta": delta,
        "global_profdata": str(global_prof),
    }
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
