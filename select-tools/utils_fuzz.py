#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared utilities for orchestrating Atheris fuzz runs on multiple harnesses.
Assumes each target directory contains harnessN.py files (N = 1..n).
Creates corpus_N and crash_N if missing, runs each harness, and collects results.
"""
import os
import re
import time
import json
import subprocess
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

HARNESS_RE = re.compile(r"^harness(\d+)\.py$")

@dataclass
class HarnessJob:
    subdir: str           # absolute path to target subdir (e.g., /path/to/torch-api-fuzz/fft.fft)
    name: str             # directory name (e.g., 'fft.fft')
    harness_num: int      # N from harnessN.py
    harness_file: str     # absolute path to harnessN.py
    corpus_dir: str       # absolute path to corpus_N
    crash_dir: str        # absolute path to crash_N
    log_dir: str          # absolute path to logs directory inside subdir

    def id(self) -> str:
        return f"{self.name}:h{self.harness_num}"

def is_dir_with_harnesses(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    try:
        for fn in os.listdir(path):
            if HARNESS_RE.match(fn):
                return True
        return False
    except Exception:
        return False

# import os
# import time

def discover_jobs(root: str, create_dirs: bool = True) -> List[HarnessJob]:
    jobs: List[HarnessJob] = []
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Root path not found: {root}")
    
    # 获取当前时间戳
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    output_root = "/root/fuzz_output"
    
    for entry in sorted(os.listdir(root)):
        subdir = os.path.join(root, entry)
        if not is_dir_with_harnesses(subdir):
            continue
        
        # Find harness files
        harness_files = [(int(HARNESS_RE.match(fn).group(1)), fn)
                         for fn in os.listdir(subdir) if HARNESS_RE.match(fn)]
        harness_files.sort(key=lambda x: x[0])
        
        # Optional: Check for gaps
        if harness_files:
            expected = list(range(1, harness_files[-1][0] + 1))
            existing = [n for n, _ in harness_files]
            missing = [n for n in expected if n not in existing]
            if missing:
                print(f"[warn] {entry}: missing harness indices {missing} (will run existing only)")
        
        # Create logs directory under /root/fuzz_output
        log_dir = os.path.join(output_root, f"{entry}_logs_{timestamp}")
        if create_dirs and not os.path.isdir(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        for num, fn in harness_files:
            corpus = os.path.join(subdir, f"corpus_{num}")
            crash  = os.path.join(subdir, f"crash_{num}")
            if create_dirs:
                os.makedirs(corpus, exist_ok=True)
                os.makedirs(crash, exist_ok=True)
            
            jobs.append(HarnessJob(
                subdir=os.path.abspath(subdir),
                name=entry,
                harness_num=num,
                harness_file=os.path.abspath(os.path.join(subdir, fn)),
                corpus_dir=os.path.abspath(corpus),
                crash_dir=os.path.abspath(crash),
                log_dir=os.path.abspath(log_dir),
            ))
    
    return jobs


def count_crashes(crash_dir: str) -> int:
    try:
        return len([f for f in os.listdir(crash_dir) if os.path.isfile(os.path.join(crash_dir, f))])
    except Exception:
        return 0

def run_harness(job: HarnessJob, m: int = 10000, timeout: Optional[int] = None,
                extra_env: Optional[Dict[str, str]] = None) -> Dict:
    """
    Execute one harness fuzzing session:
      python3 harnessN.py corpus_N -artifact_prefix=crash_N -atheris_runs=m
    Captures stdout/stderr into per-harness logs. Returns a result dict.
    """
    t0 = time.time()
    cmd = [
        "python3",
        os.path.basename(job.harness_file),
        os.path.basename(job.corpus_dir),
        f"-artifact_prefix={os.path.basename(job.crash_dir)}/",
        f"-atheris_runs={m}",
        "-use_value_profile=1",
        "-entropic=1",
    ]
    env = os.environ.copy()
    # Reasonable defaults to avoid runaway thread usage
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    if extra_env:
        env.update(extra_env)

    log_path = os.path.join(job.log_dir, f"harness{job.harness_num}.log")
    # Run inside the subdir so relative paths work
    proc = subprocess.Popen(
        cmd, cwd=job.subdir, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        rc = -9  # custom code for timeout
    
    # 过滤掉所有包含 "INFO: Instrumenting" 的行，确保逐行检查
    filtered_stdout = "\n".join(line.strip() for line in stdout.split("\n") if "INFO: Instrumenting" not in line)
    filtered_stderr = "\n".join(line.strip() for line in stderr.split("\n") if "INFO: Instrumenting" not in line)

    # Write logs
    try:
        with open(log_path, "w", encoding="utf-8", errors="ignore") as f:
            f.write("=== CMD ===\n")
            f.write(" ".join(cmd) + "\n")
            f.write(f"cwd: {job.subdir}\n")
            f.write(f"started: {time.ctime(t0)}\n")
            f.write("=== STDOUT ===\n")
            f.write(filtered_stdout or "")
            f.write("\n=== STDERR ===\n")
            f.write(filtered_stderr or "")
            f.write(f"\n=== EXIT {rc} in {time.time()-t0:.2f}s ===\n")
    except Exception as e:
        print(f"[warn] failed writing log for {job.id()}: {e}")

    result = {
        "job": asdict(job),
        "cmd": cmd,
        "returncode": rc,
        "duration_s": round(time.time() - t0, 2),
        "log_path": log_path,
        "stdout_tail": (filtered_stdout or "")[-4000:],  # tail for quick glance
        "stderr_tail": (filtered_stderr or "")[-4000:],
        "crash_count": count_crashes(job.crash_dir),
        "status": "ok" if rc == 0 else ("timeout" if rc == -9 else "crash"),
    }
    return result


