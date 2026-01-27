# screen/runner/audit_runner.py
from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


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
