#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run Atheris fuzzing across all harnessN.py files found under a root directory.
- Ensures corpus_N and crash_N exist for each harness.
- Runs in parallel with --jobs workers.
- Writes per-harness logs under each subdir/logs/ and a JSON summary at root.
Usage:
  python3 fuzz_all.py --root ./torch-api-fuzz --m 10000 --jobs 4 --timeout 0
"""
import os
import sys
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import time

from utils_fuzz import discover_jobs, run_harness

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root folder containing target subfolders (e.g., ./torch-api-fuzz)")
    ap.add_argument("--fuzz_rounds", type=int, default=10000, help="Value for -atheris_runs (default: 10000)")
    ap.add_argument("--jobs", type=int, default=4, help="Max parallel harnesses (default: 4)")
    ap.add_argument("--timeout", type=int, default=None, help="Per-harness timeout in seconds (default: None)")
    ap.add_argument("--env", action="append", help="Extra env VAR=VALUE (can pass multiple)")
    args = ap.parse_args()

    extra_env = {}
    if args.env:
        for kv in args.env:
            if "=" in kv:
                k, v = kv.split("=", 1)
                extra_env[k] = v

    jobs = discover_jobs(args.root, create_dirs=True)
    if not jobs:
        print(f"No harnesses found under {args.root}. (Expect files like harness1.py)")
        return 1

    print(f"Discovered {len(jobs)} harness jobs under {args.root}. Starting with {args.jobs} workers...")
    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        fut2job = {ex.submit(run_harness, job, args.fuzz_rounds, args.timeout, extra_env): job for job in jobs}
        for fut in as_completed(fut2job):
            job = fut2job[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = {
                    "job": job.__dict__,
                    "error": str(e),
                    "status": "error",
                    "returncode": -99,
                    "duration_s": 0,
                    "log_path": None,
                    "crash_count": 0,
                }
            results.append(res)
            jid = f"{job.name}/h{job.harness_num}"
            print(f"[{res['status']:^7}] {jid:35s} rc={res.get('returncode')} crashes={res.get('crash_count')} time={res.get('duration_s')}s")

    # Summaries
    total = len(results)
    ok = sum(1 for r in results if r.get("status") == "ok")
    timeout = sum(1 for r in results if r.get("status") == "timeout")
    err = sum(1 for r in results if r.get("status") not in ("ok", "timeout"))
    total_crashes = sum(r.get("crash_count", 0) for r in results)

    # 获取当前时间戳
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    summary = {
        "root": os.path.abspath(args.root),
        "timestamp": datetime.now().isoformat(timespec='seconds'),
        "atheris_runs": args.fuzz_rounds,
        "jobs": args.jobs,
        "timeout": args.timeout,
        "totals": {"jobs": total, "ok": ok, "timeout": timeout, "error": err, "crashes": total_crashes},
        "results": results,
    }

    # 输出路径改为 /root/fuzz_output/{subdir}_logs_{timestamp}/fuzz_summary.json
    out_dir = "/root/fuzz_output"
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"fuzz_summary_{timestamp}.json")

    # 保存 summary 到文件
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nWrote summary: {out_path}")
    print(f"OK={ok}, TIMEOUT={timeout}, ERROR={err}, CRASHES={total_crashes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
