#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

# 改成你的 PyTorch 覆盖率环境 Python
TORCH_PYTHON = "/root/pytorch_cov/bin/python3"
@dataclass
class ReplayResult:
    script: str
    ok: bool
    exit_code: Optional[int]
    duration_sec: float
    raw_dir: str
    raw_files: List[str]
    stdout_log: str
    stderr_log: str
    done_marker: str
    error: Optional[str] = None
    timeout: bool = False
    skipped_done: bool = False
    reused_raw: bool = False


def sha1_short(s: str, n: int = 12) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def find_scripts(root_dir: Path, name_glob: str, exclude_regex: Optional[str]) -> List[Path]:
    import re

    regex = re.compile(exclude_regex) if exclude_regex else None
    scripts = []

    for p in root_dir.rglob(name_glob):
        if not p.is_file():
            continue
        rel = p.relative_to(root_dir).as_posix()
        if regex and regex.search(rel):
            continue
        scripts.append(p)

    scripts.sort()
    return scripts


def split_batches(items: List[Path], batch_size: int) -> List[List[Path]]:
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def ensure_cmd(cmd: str) -> str:
    if os.path.sep in cmd:
        p = Path(cmd).resolve()
        if not p.exists():
            raise FileNotFoundError(f"找不到命令: {p}")
        return str(p)

    found = shutil.which(cmd)
    if not found:
        raise FileNotFoundError(f"找不到命令: {cmd}")
    return found


def make_tag(script: Path, root_dir: Path) -> str:
    rel = script.relative_to(root_dir).as_posix()
    return f"{script.stem}__{sha1_short(rel)}"


def get_raw_dir(script: Path, root_dir: Path, raw_root: Path) -> Path:
    tag = make_tag(script, root_dir)
    return raw_root / tag


def get_done_marker(script: Path, root_dir: Path, done_root: Path) -> Path:
    tag = make_tag(script, root_dir)
    return done_root / f"{tag}.done.json"


def remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def run_profdata_merge(
    llvm_profdata: str,
    input_files: List[str],
    output_profdata: Path,
    manifest_file: Path,
) -> subprocess.CompletedProcess:
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    output_profdata.parent.mkdir(parents=True, exist_ok=True)

    manifest_file.write_text("\n".join(input_files) + "\n", encoding="utf-8")

    tmp_output = output_profdata.with_suffix(".profdata.tmp")
    remove_path(tmp_output)

    cmd = [
        llvm_profdata,
        "merge",
        "-sparse",
        f"--input-files={manifest_file}",
        f"-o={tmp_output}",
    ]

    cp = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        check=False,
    )

    if cp.returncode == 0:
        tmp_output.replace(output_profdata)
    else:
        remove_path(tmp_output)

    return cp


def build_torch_cmd(
    python_exe: str,
    script: Path,
    torch_threads: int,
    torch_interop_threads: int,
) -> List[str]:
    """
    用一个很小的 wrapper 启动 seed 脚本。
    这样可以在执行 seed 前设置 PyTorch 线程数。
    """

    runner = f"""
import runpy
import sys

script_path = sys.argv[1]
sys.argv = [script_path] + sys.argv[2:]

try:
    import torch

    if {torch_threads} > 0:
        torch.set_num_threads({torch_threads})

    if {torch_interop_threads} > 0:
        try:
            torch.set_num_interop_threads({torch_interop_threads})
        except RuntimeError:
            pass

except Exception as e:
    print(f"[WARN] failed to configure PyTorch threads: {{e}}", file=sys.stderr)

runpy.run_path(script_path, run_name="__main__")
"""

    return [python_exe, "-u", "-c", runner, str(script)]


def run_one_script(
    script: Path,
    root_dir: Path,
    raw_root: Path,
    done_root: Path,
    log_root: Path,
    python_exe: str,
    timeout: int,
    torch_threads: int,
    torch_interop_threads: int,
    merge_failed_with_raw: bool,
) -> ReplayResult:
    tag = make_tag(script, root_dir)

    raw_dir = get_raw_dir(script, root_dir, raw_root)
    done_marker = get_done_marker(script, root_dir, done_root)

    stdout_log = log_root / f"{tag}.stdout.log"
    stderr_log = log_root / f"{tag}.stderr.log"

    if done_marker.exists():
        return ReplayResult(
            script=str(script),
            ok=True,
            exit_code=0,
            duration_sec=0.0,
            raw_dir=str(raw_dir),
            raw_files=[],
            stdout_log=str(stdout_log),
            stderr_log=str(stderr_log),
            done_marker=str(done_marker),
            skipped_done=True,
        )

    existing_raw = sorted(str(p) for p in raw_dir.rglob("*.profraw")) if raw_dir.exists() else []
    if existing_raw:
        return ReplayResult(
            script=str(script),
            ok=True,
            exit_code=0,
            duration_sec=0.0,
            raw_dir=str(raw_dir),
            raw_files=existing_raw,
            stdout_log=str(stdout_log),
            stderr_log=str(stderr_log),
            done_marker=str(done_marker),
            reused_raw=True,
        )

    raw_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()

    # 每个脚本单独一个 raw 目录，避免多进程写同一个 profraw
    env["LLVM_PROFILE_FILE"] = str(raw_dir / f"{tag}_%p_%m.profraw")

    # 避免 Python 缓冲导致日志看不到
    env["PYTHONUNBUFFERED"] = "1"

    # PyTorch / BLAS / OpenMP 线程限制
    # 并发 replay 时非常重要，否则每个进程内部再开很多线程，很容易把机器打满
    env["OMP_NUM_THREADS"] = str(torch_threads)
    env["MKL_NUM_THREADS"] = str(torch_threads)
    env["OPENBLAS_NUM_THREADS"] = str(torch_threads)
    env["NUMEXPR_NUM_THREADS"] = str(torch_threads)
    env["VECLIB_MAXIMUM_THREADS"] = str(torch_threads)
    env["BLIS_NUM_THREADS"] = str(torch_threads)

    # PyTorch C++ 日志降噪，可保留
    env.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")

    cmd = build_torch_cmd(
        python_exe=python_exe,
        script=script,
        torch_threads=torch_threads,
        torch_interop_threads=torch_interop_threads,
    )

    t0 = time.time()
    exit_code = None
    ok = False
    is_timeout = False
    error = None
    stdout = ""
    stderr = ""

    try:
        cp = subprocess.run(
            cmd,
            cwd=str(script.parent),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            timeout=timeout,
            check=False,
        )
        exit_code = cp.returncode
        stdout = cp.stdout
        stderr = cp.stderr
        ok = cp.returncode == 0

    except subprocess.TimeoutExpired as e:
        is_timeout = True
        error = f"timeout({timeout}s)"
        stdout = e.stdout if isinstance(e.stdout, str) else ""
        stderr = e.stderr if isinstance(e.stderr, str) else ""
        stderr += f"\n[TIMEOUT] script exceeded {timeout}s\n"

    except Exception as e:
        error = repr(e)
        stderr = f"[EXCEPTION] {repr(e)}\n"

    duration = time.time() - t0

    write_text(stdout_log, stdout)
    write_text(stderr_log, stderr)

    raw_files = sorted(str(p) for p in raw_dir.rglob("*.profraw"))

    if error is None and not ok:
        error = f"exit_code({exit_code})"

    if ok and not raw_files:
        error = "script exited 0 but emitted no .profraw"

    # 默认只合并成功脚本的 raw
    # 如果你希望失败但已产生 raw 的脚本也计入覆盖率，可以加 --merge-failed-with-raw
    final_ok_for_merge = ok or (merge_failed_with_raw and raw_files and not is_timeout)

    return ReplayResult(
        script=str(script),
        ok=final_ok_for_merge,
        exit_code=exit_code,
        duration_sec=round(duration, 2),
        raw_dir=str(raw_dir),
        raw_files=raw_files,
        stdout_log=str(stdout_log),
        stderr_log=str(stderr_log),
        done_marker=str(done_marker),
        error=error,
        timeout=is_timeout,
    )


def write_done_marker(result: ReplayResult, batch_index: int, batch_profdata: Path) -> None:
    payload = {
        "script": result.script,
        "batch_index": batch_index,
        "batch_profdata": str(batch_profdata),
        "merged_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_text(Path(result.done_marker), json.dumps(payload, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Replay merged PyTorch API seed scripts and collect LLVM coverage."
    )

    ap.add_argument(
        "root_dir",
        help="存放 *_merged.py 的根目录，会递归扫描",
    )

    ap.add_argument(
        "--out-dir",
        default="./replay_cov_out",
        help="输出目录",
    )

    ap.add_argument(
        "--name-glob",
        default="*_merged.py",
        help="要扫描的脚本模式，默认 *_merged.py",
    )

    ap.add_argument(
        "--exclude-regex",
        default=None,
        help="排除文件的正则，例如 'bad|deprecated'",
    )

    ap.add_argument(
        "--python",
        default=TORCH_PYTHON,
        help="用于重放的 Python 解释器，建议填你的 PyTorch 覆盖率环境 python",
    )

    ap.add_argument(
        "--llvm-profdata",
        default="llvm-profdata",
        help="llvm-profdata 路径",
    )

    ap.add_argument(
        "--workers",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="并发重放进程数",
    )

    ap.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="每多少个 merged 脚本合并一次 profraw",
    )

    ap.add_argument(
        "--timeout",
        type=int,
        default=36000,
        help="单个 merged 脚本超时时间，单位秒",
    )

    ap.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="每个 PyTorch 进程的 intra-op / OpenMP 线程数",
    )

    ap.add_argument(
        "--torch-interop-threads",
        type=int,
        default=1,
        help="每个 PyTorch 进程的 inter-op 线程数",
    )

    ap.add_argument(
        "--keep-raw",
        action="store_true",
        help="合并成功后不删除 .profraw，调试时可用",
    )

    ap.add_argument(
        "--merge-failed-with-raw",
        action="store_true",
        help="失败脚本如果产生了 profraw，也尝试合并进去",
    )

    return ap.parse_args()


def main() -> int:
    args = parse_args()

    root_dir = Path(args.root_dir).resolve()
    if not root_dir.exists() or not root_dir.is_dir():
        print(f"[ERROR] root_dir 不存在: {root_dir}", file=sys.stderr)
        return 2

    # 注意：这里必须用 args.python，而不是写死 TORCH_PYTHON
    python_exe = args.python
    if not Path(python_exe).exists():
        print(f"[ERROR] Python 不存在: {python_exe}", file=sys.stderr)
        return 2

    llvm_profdata = ensure_cmd(args.llvm_profdata)

    out_dir = Path(args.out_dir).resolve()
    raw_root = out_dir / "raw"
    log_root = out_dir / "logs"
    done_root = out_dir / "done"
    batch_root = out_dir / "batches"
    report_root = out_dir / "reports"
    tmp_root = out_dir / "tmp"

    for d in [raw_root, log_root, done_root, batch_root, report_root, tmp_root]:
        d.mkdir(parents=True, exist_ok=True)

    scripts = find_scripts(root_dir, args.name_glob, args.exclude_regex)

    if not scripts:
        print(f"[WARN] 没找到脚本: {root_dir} / {args.name_glob}")
        return 0

    batches = split_batches(scripts, args.batch_size)

    print("========== Replay Config ==========")
    print(f"root_dir              : {root_dir}")
    print(f"out_dir               : {out_dir}")
    print(f"python                : {python_exe}")
    print(f"llvm-profdata         : {llvm_profdata}")
    print(f"name_glob             : {args.name_glob}")
    print(f"scripts               : {len(scripts)}")
    print(f"batch_size            : {args.batch_size}")
    print(f"batches               : {len(batches)}")
    print(f"workers               : {args.workers}")
    print(f"timeout               : {args.timeout}s")
    print(f"torch threads         : {args.torch_threads}")
    print(f"torch interop threads : {args.torch_interop_threads}")
    print("===================================")

    all_results: List[ReplayResult] = []

    total = len(scripts)
    completed = 0
    success = 0
    failed = 0
    timeout_cnt = 0
    skipped = 0
    reused = 0

    t_all = time.time()

    for batch_idx, batch_scripts in enumerate(batches, start=1):
        print(f"\n>>> [BATCH {batch_idx}/{len(batches)}] start, scripts={len(batch_scripts)}")

        t_batch = time.time()
        batch_results: List[ReplayResult] = []

        local_workers = max(1, min(args.workers, len(batch_scripts)))

        with cf.ThreadPoolExecutor(max_workers=local_workers) as ex:
            futs = [
                ex.submit(
                    run_one_script,
                    script,
                    root_dir,
                    raw_root,
                    done_root,
                    log_root,
                    python_exe,
                    args.timeout,
                    args.torch_threads,
                    args.torch_interop_threads,
                    args.merge_failed_with_raw,
                )
                for script in batch_scripts
            ]

            for fut in cf.as_completed(futs):
                r = fut.result()
                batch_results.append(r)
                all_results.append(r)

                completed += 1

                if r.skipped_done:
                    skipped += 1
                    status = "SKIP"
                elif r.reused_raw:
                    reused += 1
                    status = "REUSE_RAW"
                elif r.timeout:
                    timeout_cnt += 1
                    status = "TIMEOUT"
                elif r.ok:
                    success += 1
                    status = "OK"
                else:
                    failed += 1
                    status = "FAIL"

                percent = completed / total * 100

                print(
                    f"[{status}] {completed}/{total} ({percent:.1f}%) "
                    f"ok={success} fail={failed} timeout={timeout_cnt} "
                    f"skip={skipped} reuse={reused} "
                    f"time={r.duration_sec}s "
                    f"{Path(r.script).name}"
                )

        batch_results.sort(key=lambda x: x.script)

        batch_raw_files: List[str] = []
        merge_candidates: List[ReplayResult] = []

        for r in batch_results:
            if r.skipped_done:
                continue
            if r.ok and r.raw_files:
                merge_candidates.append(r)
                batch_raw_files.extend(r.raw_files)

        batch_profdata = batch_root / f"batch_{batch_idx:06d}.profdata"
        batch_manifest = tmp_root / f"batch_{batch_idx:06d}_inputs.txt"
        batch_report = report_root / f"batch_{batch_idx:06d}.json"

        batch_payload = {
            "batch_index": batch_idx,
            "script_count": len(batch_scripts),
            "raw_input_count": len(batch_raw_files),
            "batch_profdata": str(batch_profdata),
            "results": [asdict(r) for r in batch_results],
        }

        if batch_raw_files:
            merge_inputs = batch_raw_files[:]

            # 如果这个 batch 之前已经有 profdata，说明是断点续跑，
            # 新 raw 要和旧 batch profdata 再合并一次。
            if batch_profdata.exists():
                merge_inputs = [str(batch_profdata)] + merge_inputs

            print(f"[BATCH MERGE] batch={batch_idx}, inputs={len(merge_inputs)}")

            cp = run_profdata_merge(
                llvm_profdata=llvm_profdata,
                input_files=merge_inputs,
                output_profdata=batch_profdata,
                manifest_file=batch_manifest,
            )

            if cp.returncode != 0:
                batch_payload["merge_status"] = "failed"
                batch_payload["merge_stderr"] = cp.stderr
                write_text(batch_report, json.dumps(batch_payload, ensure_ascii=False, indent=2))

                print(f"[ERROR] batch {batch_idx} merge failed", file=sys.stderr)
                print(cp.stderr, file=sys.stderr)
                return 4

            batch_payload["merge_status"] = "ok"
            batch_payload["merge_stdout"] = cp.stdout
            batch_payload["merge_stderr"] = cp.stderr

            for r in merge_candidates:
                write_done_marker(r, batch_idx, batch_profdata)
                if not args.keep_raw:
                    remove_path(Path(r.raw_dir))

            print(
                f"[BATCH MERGE OK] {batch_profdata.name}, "
                f"raw={len(batch_raw_files)}, "
                f"cleaned={0 if args.keep_raw else len(merge_candidates)}, "
                f"time={time.time() - t_batch:.2f}s"
            )

        else:
            if batch_profdata.exists():
                batch_payload["merge_status"] = "reuse_existing_batch_profdata"
                print(f"[BATCH REUSE] {batch_profdata.name}")
            else:
                batch_payload["merge_status"] = "no_raw"
                print(f"[BATCH NO RAW] batch={batch_idx}")

        write_text(batch_report, json.dumps(batch_payload, ensure_ascii=False, indent=2))

    print("\n========== Final Merge ==========")

    batch_profdata_files = sorted(str(p) for p in batch_root.glob("batch_*.profdata"))

    if not batch_profdata_files:
        print("[ERROR] 没有任何 batch profdata，无法最终合并", file=sys.stderr)
        return 5

    total_profdata = out_dir / "merged_total.profdata"
    total_manifest = tmp_root / "all_batch_profdata_manifest.txt"

    cp = run_profdata_merge(
        llvm_profdata=llvm_profdata,
        input_files=batch_profdata_files,
        output_profdata=total_profdata,
        manifest_file=total_manifest,
    )

    result_json = report_root / "run_results.json"
    summary = {
        "total_scripts": total,
        "success": success,
        "failed": failed,
        "timeout": timeout_cnt,
        "skipped_done": skipped,
        "reused_raw": reused,
        "batch_profdata_count": len(batch_profdata_files),
        "total_profdata": str(total_profdata),
        "elapsed_sec": round(time.time() - t_all, 2),
    }

    write_text(
        result_json,
        json.dumps(
            {
                "summary": summary,
                "results": [asdict(r) for r in all_results],
            },
            ensure_ascii=False,
            indent=2,
        ),
    )

    if cp.returncode != 0:
        print("[ERROR] final merge failed", file=sys.stderr)
        print(cp.stderr, file=sys.stderr)
        return 6

    print(f"[FINAL OK] total profdata: {total_profdata}")
    print(f"[REPORT] {result_json}")
    print("========== Summary ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())