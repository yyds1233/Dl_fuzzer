# screen/runner/fuzz_runner.py
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Dict, List


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
