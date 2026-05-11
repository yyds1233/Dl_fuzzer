#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import gc
import hashlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_SOURCE_PATTERNS = [
    "corpus/{harness_id}/**/*",
    "audits/{harness_id}/**/window_corpus/**/*",
    "runs/{harness_id}/**/corpus/**/*",
    "runs/{harness_id}/**/queue/**/*",
]


# ---------------- common utilities ----------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts} pid={os.getpid()}] {msg}", flush=True)


def _run(
    cmd: Sequence[str],
    *,
    env: Optional[Dict[str, str]] = None,
    timeout: Optional[int] = None,
) -> str:
    p = subprocess.run(
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        timeout=timeout,
    )
    if p.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  rc: {p.returncode}\n"
            f"  stdout:\n{p.stdout[-4000:]}\n"
            f"  stderr:\n{p.stderr[-4000:]}\n"
        )
    return p.stdout


def _fsync_file(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except Exception:
        # fsync is best effort here; the atomic replace below is still the
        # important correctness boundary.
        pass


def _fsync_parent(path: Path) -> None:
    try:
        fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except Exception:
        pass


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(text, encoding="utf-8")
    _fsync_file(tmp)
    os.replace(tmp, path)
    _fsync_parent(path)


def _atomic_write_json(path: Path, obj: Dict[str, object]) -> None:
    _atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False))


def _append_jsonl(path: Path, obj: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True, ensure_ascii=False) + "\n")
        f.flush()


def _safe_unlink(path: Path) -> None:
    try:
        if path.exists() or path.is_symlink():
            path.unlink()
    except Exception:
        pass


def _safe_rmtree(path: Path) -> None:
    try:
        if path.exists():
            shutil.rmtree(path)
    except Exception:
        pass


class _FileLock:
    """Small POSIX advisory lock used to prevent two script instances from
    merging into the same cumulative profdata at the same time.
    """

    def __init__(self, path: Path):
        self.path = path
        self._fh = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a+")
        try:
            import fcntl

            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
        except Exception:
            # Non-POSIX platforms still get single-parent-process safety.
            pass
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._fh is not None:
            try:
                import fcntl

                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            try:
                self._fh.close()
            except Exception:
                pass


# ---------------- coverage helpers ----------------


def _parse_lcov_summary(text: str) -> Dict[str, int]:
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


def _cov_summary_lcov(
    *,
    llvm_cov: str,
    profdata: Path,
    primary_object: Path,
    extra_objects: Sequence[Path],
    ignore_filename_regex: Optional[str],
) -> Dict[str, int]:
    cmd = [
        llvm_cov,
        "export",
        "-summary-only",
        "-format=lcov",
        f"-instr-profile={profdata}",
        str(primary_object),
    ]
    for obj in extra_objects:
        cmd += ["-object", str(obj)]
    if ignore_filename_regex:
        cmd += [f"-ignore-filename-regex={ignore_filename_regex}"]
    text = _run(cmd)
    return _parse_lcov_summary(text)


def _chunks(xs: Sequence[str], n: int) -> Iterable[List[str]]:
    if n <= 0:
        n = len(xs) or 1
    for i in range(0, len(xs), n):
        yield list(xs[i : i + n])


def _merge_profiles_chunked(
    llvm_profdata: str,
    inputs: Sequence[Path],
    out_profdata: Path,
    *,
    chunk_size: int = 64,
    tmp_dir: Optional[Path] = None,
) -> None:
    """Merge profile files while keeping llvm-profdata command lines bounded.

    Important storage behavior:
    - chunk intermediates are removed after the final merge succeeds;
    - callers are still responsible for deleting their original inputs when it
      is safe to do so.
    """

    inputs = [Path(p) for p in inputs if Path(p).exists()]
    if not inputs:
        raise RuntimeError("No profile files found to merge")

    out_profdata.parent.mkdir(parents=True, exist_ok=True)

    if len(inputs) <= chunk_size:
        cmd = [
            llvm_profdata,
            "merge",
            "-sparse",
            *[str(p) for p in inputs],
            "-o",
            str(out_profdata),
        ]
        _run(cmd)
        _fsync_file(out_profdata)
        return

    created_tmp_dir = False
    if tmp_dir is None:
        tmp_dir = out_profdata.parent / f".merge_tmp_{out_profdata.stem}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        created_tmp_dir = True
    tmp_dir.mkdir(parents=True, exist_ok=True)

    intermediates: List[Path] = []
    try:
        for idx, chunk in enumerate(_chunks([str(p) for p in inputs], chunk_size)):
            chunk_out = tmp_dir / f"{out_profdata.stem}.chunk_{idx:04d}.profdata"
            cmd = [llvm_profdata, "merge", "-sparse", *chunk, "-o", str(chunk_out)]
            _run(cmd)
            _fsync_file(chunk_out)
            intermediates.append(chunk_out)

        _merge_profiles_chunked(
            llvm_profdata,
            intermediates,
            out_profdata,
            chunk_size=chunk_size,
            tmp_dir=tmp_dir,
        )
    finally:
        for p in intermediates:
            _safe_unlink(p)
        if created_tmp_dir:
            _safe_rmtree(tmp_dir)


def _sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _load_harnesses_json(path: Path) -> Tuple[Dict[str, Path], List[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, Path] = {}
    order: List[str] = []
    seen_order = set()

    for row in data:
        hid = str(row["harness_id"])
        hpath = Path(row["harness_path"]).resolve()
        out[hid] = hpath
        stripped = hid.strip()
        if stripped and stripped not in out:
            out[stripped] = hpath
        if hid not in seen_order:
            seen_order.add(hid)
            order.append(hid)

    return out, order


def _safe_harness_file_id(harness_id: str) -> str:
    s = harness_id.strip()
    s = s.replace("/", "__")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_.=-]+", "_", s)
    return s or "unknown_harness"


def _collect_candidate_files(
    root: Path,
    harness_id: str,
    patterns: Sequence[str],
) -> List[Tuple[Path, str]]:
    seen_paths = set()
    out: List[Tuple[Path, str]] = []
    for pat in patterns:
        expanded = pat.format(harness_id=harness_id)
        for p in root.glob(expanded):
            if not p.is_file():
                continue
            rp = str(p.resolve())
            if rp in seen_paths:
                continue
            seen_paths.add(rp)
            out.append((p.resolve(), expanded))
    return out


def _collect_unique_inputs(
    root: Path,
    harness_id: str,
    patterns: Sequence[str],
) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    candidates = _collect_candidate_files(root, harness_id, patterns)
    by_hash: Dict[str, Dict[str, object]] = {}
    source_counts: Dict[str, int] = {}

    for p, pat in candidates:
        source_counts[pat] = source_counts.get(pat, 0) + 1
        try:
            sha1 = _sha1_file(p)
        except Exception:
            continue

        try:
            rel = str(p.relative_to(root))
        except Exception:
            rel = str(p)

        if sha1 not in by_hash:
            by_hash[sha1] = {
                "sha1": sha1,
                "path": str(p),
                "relpath": rel,
                "size": int(p.stat().st_size),
                "sources": [pat],
            }
        else:
            src = by_hash[sha1].setdefault("sources", [])
            if pat not in src:
                src.append(pat)

    items = sorted(by_hash.values(), key=lambda x: (str(x["relpath"]), str(x["sha1"])))
    return items, source_counts


# ---------------- libFuzzer worker subprocess ----------------


def _identity_mutate(spec, cfg, fdp, **kwargs):
    return cfg


def _load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _symlink_or_copy(src: Path, dst: Path) -> None:
    try:
        os.symlink(src, dst)
    except Exception:
        try:
            shutil.copy2(src, dst)
        except Exception:
            dst.write_bytes(src.read_bytes())


def _split_flag_text(flag_text: Optional[str]) -> List[str]:
    if not flag_text:
        return []
    try:
        return shlex.split(flag_text)
    except Exception:
        return str(flag_text).split()


def _run_batch_via_libfuzzer(
    mod,
    harness_path: Path,
    file_list: Sequence[str],
    flag_text: Optional[str],
) -> int:
    import atheris

    with tempfile.TemporaryDirectory(prefix=f"replay_{harness_path.stem}_") as td:
        corpus_dir = Path(td) / "corpus"
        corpus_dir.mkdir(parents=True, exist_ok=True)

        for i, path_str in enumerate(file_list):
            src = Path(path_str)
            suffix = src.suffix or ".bin"
            dst = corpus_dir / f"seed_{i:06d}{suffix}"
            _symlink_or_copy(src, dst)

        fuzz_argv = [
            str(harness_path),
            str(corpus_dir),
            f"-runs={len(file_list)}",
            "-print_final_stats=1",
        ]
        fuzz_argv.extend(_split_flag_text(flag_text))

        atheris.Setup(fuzz_argv, mod.TestOneInput)
        atheris.Fuzz()
        return 0


def _worker_main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--disable-harness-mutation", action="store_true")
    ap.add_argument("--gc-every", type=int, default=16)
    ap.add_argument("--cov_replay_extra", default=None)
    ap.add_argument("--replay_extra", default=None)
    args = ap.parse_args(list(argv))

    flag_text = (
        args.cov_replay_extra
        or args.replay_extra
        or "-rss_limit_mb=8192 -malloc_limit_mb=8192"
    )

    harness_path = Path(args.harness).resolve()
    harness_dir = harness_path.parent
    if str(harness_dir) not in sys.path:
        sys.path.insert(0, str(harness_dir))

    module_name = f"coverage_replay_{harness_path.stem}_{os.getpid()}_{int(time.time() * 1000)}"
    mod = _load_module_from_path(module_name, harness_path)

    if args.disable_harness_mutation:
        mod.mutate_cfg = _identity_mutate

    if not hasattr(mod, "TestOneInput"):
        raise RuntimeError(f"Harness has no TestOneInput: {harness_path}")

    file_list = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if not file_list:
        return 0

    return _run_batch_via_libfuzzer(mod, harness_path, file_list, flag_text)


# ---------------- harness-level parallel job ----------------


def _emit_harness_event(cfg: Dict[str, object], safe_hid: str, event: Dict[str, object]) -> None:
    row = {
        "time": _now_iso(),
        "pid": os.getpid(),
        "harness": safe_hid,
        **event,
    }
    progress_dir = Path(str(cfg["progress_dir"]))
    _append_jsonl(progress_dir / f"{safe_hid}.jsonl", row)

    if bool(cfg.get("verbose", True)):
        msg = str(event.get("message") or event.get("status") or event)
        _log(f"[harness={safe_hid}] {msg}")


def _harness_replay_job(task: Dict[str, object], cfg: Dict[str, object]) -> Dict[str, object]:
    harness_id = str(task["harness_id"])
    harness_path = Path(str(task["harness_path"])).resolve()
    safe_hid = _safe_harness_file_id(harness_id)
    run_id = f"{int(time.time())}_{os.getpid()}_{uuid.uuid4().hex[:8]}"

    root = Path(str(cfg["root"])).resolve()
    out_dir = Path(str(cfg["out_dir"])).resolve()
    manifests_dir = Path(str(cfg["manifests_dir"]))
    per_harness_dir = Path(str(cfg["per_harness_dir"]))
    tmp_root = Path(str(cfg["tmp_dir"]))
    marks_dir = Path(str(cfg["marks_dir"]))

    patterns = list(cfg["patterns"])
    batch_size = int(cfg["batch_size"])
    merge_chunk_size = int(cfg["merge_chunk_size"])
    timeout = int(cfg["timeout"])
    python_exe = str(cfg["python"])
    llvm_profdata = str(cfg["llvm_profdata"])
    llvm_cov = str(cfg["llvm_cov"])
    primary_object = Path(str(cfg["primary_object"])).resolve()
    extra_objects = [Path(str(x)).resolve() for x in cfg.get("extra_object", [])]
    ignore_filename_regex = cfg.get("ignore_filename_regex")
    disable_harness_mutation = bool(cfg.get("disable_harness_mutation", False))
    keep_empty_harnesses = bool(cfg.get("keep_empty_harnesses", False))
    keep_failed_profiles = bool(cfg.get("keep_failed_profiles", False))
    flag_text = (
        cfg.get("cov_replay_extra")
        or cfg.get("replay_extra")
        or "-rss_limit_mb=8192 -malloc_limit_mb=8192"
    )

    mark_path = marks_dir / f"{safe_hid}.json"
    harness_work = tmp_root / safe_hid / run_id
    profraw_work = harness_work / "profraw"
    batch_profdata_dir = harness_work / "batch_profdata"
    batch_manifest_dir = harness_work / "batch_manifests"
    merge_tmp_dir = harness_work / "merge_tmp"

    profraw_work.mkdir(parents=True, exist_ok=True)
    batch_profdata_dir.mkdir(parents=True, exist_ok=True)
    batch_manifest_dir.mkdir(parents=True, exist_ok=True)
    per_harness_dir.mkdir(parents=True, exist_ok=True)
    (manifests_dir / safe_hid).mkdir(parents=True, exist_ok=True)

    start_time = _now_iso()
    _atomic_write_json(
        mark_path,
        {
            "status": "running",
            "harness_id": harness_id,
            "safe_harness_id": safe_hid,
            "harness_path": str(harness_path),
            "started_at": start_time,
            "worker_pid": os.getpid(),
            "run_id": run_id,
        },
    )

    result_base: Dict[str, object] = {
        "harness_id": harness_id,
        "safe_harness_id": safe_hid,
        "harness_path": str(harness_path),
        "started_at": start_time,
        "run_id": run_id,
        "worker_pid": os.getpid(),
    }

    try:
        _emit_harness_event(
            cfg,
            safe_hid,
            {"status": "collecting", "message": "collecting unique corpus inputs"},
        )
        unique_items, source_counts = _collect_unique_inputs(root, harness_id, patterns)

        stable_manifest = manifests_dir / safe_hid / "unique_inputs.json"
        _atomic_write_json(
            stable_manifest,
            {
                "harness_id": harness_id,
                "harness_path": str(harness_path),
                "unique_inputs": unique_items,
                "candidate_sources": source_counts,
                "generated_at": _now_iso(),
            },
        )

        if not unique_items and not keep_empty_harnesses:
            _emit_harness_event(
                cfg,
                safe_hid,
                {"status": "skipped_empty", "message": "no corpus inputs; skipped"},
            )
            _safe_rmtree(harness_work)
            return {
                **result_base,
                "status": "skipped_empty",
                "candidate_sources": source_counts,
                "unique_inputs": 0,
                "batches": 0,
                "profraw_count": 0,
                "batch_profdata_count": 0,
                "per_harness_profdata": None,
                "per_harness_totals": None,
                "finished_at": _now_iso(),
            }

        unique_paths = [str(x["path"]) for x in unique_items]
        batches = list(_chunks(unique_paths, batch_size))
        _emit_harness_event(
            cfg,
            safe_hid,
            {
                "status": "replaying",
                "unique_inputs": len(unique_items),
                "batches": len(batches),
                "message": f"start replay: inputs={len(unique_items)} batches={len(batches)}",
            },
        )

        batch_profdata_files: List[Path] = []
        profraw_count = 0

        for batch_idx, batch_paths in enumerate(batches):
            batch_manifest = batch_manifest_dir / f"batch_{batch_idx:04d}.paths.json"
            _atomic_write_text(batch_manifest, json.dumps(batch_paths, indent=2, ensure_ascii=False))

            # Critical isolation rule:
            # Each harness job writes profraw only into its own run directory.
            # The parent never globs a shared directory, so concurrent jobs cannot
            # collect or merge each other's raw profiles.
            profraw_pattern = profraw_work / f"batch_{batch_idx:04d}_%p.profraw"
            env = os.environ.copy()
            env["LLVM_PROFILE_FILE"] = str(profraw_pattern)

            cmd = [
                python_exe,
                str(Path(__file__).resolve()),
                "--worker",
                "--harness",
                str(harness_path),
                "--manifest",
                str(batch_manifest),
            ]
            if disable_harness_mutation:
                cmd.append("--disable-harness-mutation")
            if flag_text:
                cmd += ["--cov_replay_extra", str(flag_text)]

            _run(cmd, env=env, timeout=timeout)

            batch_profraws = sorted(profraw_work.glob(f"batch_{batch_idx:04d}_*.profraw"))
            profraw_count += len(batch_profraws)

            if batch_profraws:
                batch_profdata = batch_profdata_dir / f"batch_{batch_idx:04d}.profdata"
                _merge_profiles_chunked(
                    llvm_profdata,
                    batch_profraws,
                    batch_profdata,
                    chunk_size=merge_chunk_size,
                    tmp_dir=merge_tmp_dir / f"batch_{batch_idx:04d}",
                )
                batch_profdata_files.append(batch_profdata)

            # Immediately delete raw profile files from this batch. At most
            # roughly --jobs batches' raw files exist at the same time.
            for p in batch_profraws:
                _safe_unlink(p)

            if batch_idx % 8 == 0:
                gc.collect()

            _emit_harness_event(
                cfg,
                safe_hid,
                {
                    "status": "batch_done",
                    "batch": batch_idx + 1,
                    "batches": len(batches),
                    "batch_profraws": len(batch_profraws),
                    "message": f"batch {batch_idx + 1}/{len(batches)} done; profraw={len(batch_profraws)} merged+deleted",
                },
            )

        if not batch_profdata_files:
            _safe_rmtree(harness_work)
            return {
                **result_base,
                "status": "no_profiles",
                "candidate_sources": source_counts,
                "unique_inputs": len(unique_items),
                "batches": len(batches),
                "profraw_count": profraw_count,
                "batch_profdata_count": 0,
                "per_harness_profdata": None,
                "per_harness_totals": None,
                "finished_at": _now_iso(),
            }

        per_profdata_tmp = per_harness_dir / f"{safe_hid}.{run_id}.tmp.profdata"
        per_profdata = per_harness_dir / f"{safe_hid}.profdata"

        _merge_profiles_chunked(
            llvm_profdata,
            batch_profdata_files,
            per_profdata_tmp,
            chunk_size=merge_chunk_size,
            tmp_dir=merge_tmp_dir / "per_harness",
        )
        os.replace(per_profdata_tmp, per_profdata)
        _fsync_file(per_profdata)
        _fsync_parent(per_profdata)

        # Once the per-harness profdata exists, batch profdata files are no
        # longer needed.
        for p in batch_profdata_files:
            _safe_unlink(p)

        per_summary = _cov_summary_lcov(
            llvm_cov=llvm_cov,
            profdata=per_profdata,
            primary_object=primary_object,
            extra_objects=extra_objects,
            ignore_filename_regex=str(ignore_filename_regex) if ignore_filename_regex else None,
        )

        _safe_rmtree(harness_work)
        _emit_harness_event(
            cfg,
            safe_hid,
            {
                "status": "harness_profdata_ready",
                "message": f"per-harness profdata ready: {per_profdata}",
            },
        )

        return {
            **result_base,
            "status": "profdata_ready",
            "candidate_sources": source_counts,
            "unique_inputs": len(unique_items),
            "batches": len(batches),
            "profraw_count": profraw_count,
            "batch_profdata_count": len(batch_profdata_files),
            "per_harness_profdata": str(per_profdata),
            "per_harness_totals": per_summary,
            "finished_at": _now_iso(),
        }

    except BaseException as e:
        err = {
            **result_base,
            "status": "failed",
            "candidate_sources": {},
            "unique_inputs": None,
            "batches": None,
            "profraw_count": None,
            "batch_profdata_count": None,
            "per_harness_profdata": None,
            "per_harness_totals": None,
            "error": repr(e),
            "traceback": traceback.format_exc(),
            "failed_at": _now_iso(),
        }

        if not keep_failed_profiles:
            # On failure, avoid leaving partial raw profiles around. The failed
            # mark preserves enough information to retry the harness later.
            _safe_rmtree(harness_work)

        _emit_harness_event(
            cfg,
            safe_hid,
            {"status": "failed", "message": f"failed: {repr(e)}"},
        )
        return err


# ---------------- cumulative merge transaction ----------------


def _backup_final_profdata(final_profdata: Path, backup_path: Path) -> Optional[Path]:
    if not final_profdata.exists():
        return None
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    _safe_unlink(backup_path)
    try:
        os.link(final_profdata, backup_path)
    except Exception:
        shutil.copy2(final_profdata, backup_path)
    _fsync_file(backup_path)
    _fsync_parent(backup_path)
    return backup_path


def _write_done_mark(
    marks_dir: Path,
    result: Dict[str, object],
    final_profdata: Path,
) -> None:
    safe_hid = str(result["safe_harness_id"])
    mark = {
        **result,
        "status": "done",
        "merged": True,
        "merged_into": str(final_profdata),
        "merged_at": _now_iso(),
    }
    _atomic_write_json(marks_dir / f"{safe_hid}.json", mark)


def _write_terminal_mark(
    marks_dir: Path,
    result: Dict[str, object],
    *,
    status: str,
) -> None:
    safe_hid = str(result["safe_harness_id"])
    mark = {**result, "status": status}
    if status not in mark:
        mark[f"{status}_at"] = _now_iso()
    _atomic_write_json(marks_dir / f"{safe_hid}.json", mark)


def _recover_pending_transactions(
    *,
    final_profdata: Path,
    marks_dir: Path,
    tx_dir: Path,
) -> None:
    tx_dir.mkdir(parents=True, exist_ok=True)
    for tx_file in sorted(tx_dir.glob("*.json")):
        try:
            tx = json.loads(tx_file.read_text(encoding="utf-8"))
        except Exception:
            _log(f"[recovery] ignoring unreadable transaction file: {tx_file}")
            continue

        safe_hid = str(tx.get("safe_harness_id") or tx_file.stem)
        done_mark = marks_dir / f"{safe_hid}.json"
        done = False
        if done_mark.exists():
            try:
                mark = json.loads(done_mark.read_text(encoding="utf-8"))
                done = mark.get("status") == "done"
            except Exception:
                done = False

        backup_raw = tx.get("backup_profdata")
        backup = Path(str(backup_raw)) if backup_raw else None

        if done:
            # The merge completed and the mark was written. Any leftover
            # transaction artifacts are safe to remove.
            if backup is not None:
                _safe_unlink(backup)
            _safe_unlink(tx_file)
            _log(f"[recovery] cleaned completed transaction for {safe_hid}")
            continue

        # No done mark means the cumulative file is uncertain. Restore the
        # pre-merge version, or remove final if there was no previous final.
        if backup is not None and backup.exists():
            os.replace(backup, final_profdata)
            _fsync_file(final_profdata)
            _fsync_parent(final_profdata)
            _log(f"[recovery] restored cumulative profdata before unfinished merge: {safe_hid}")
        else:
            if final_profdata.exists():
                _safe_unlink(final_profdata)
                _fsync_parent(final_profdata)
            _log(f"[recovery] removed cumulative profdata from unfinished first merge: {safe_hid}")

        _safe_unlink(tx_file)


def _merge_one_harness_to_cumulative(
    *,
    result: Dict[str, object],
    final_profdata: Path,
    marks_dir: Path,
    tx_dir: Path,
    backup_dir: Path,
    merge_tmp_dir: Path,
    llvm_profdata: str,
    merge_chunk_size: int,
    lock_path: Path,
) -> None:
    safe_hid = str(result["safe_harness_id"])
    per_raw = result.get("per_harness_profdata")
    if not per_raw:
        raise RuntimeError(f"Harness {safe_hid} has no per-harness profdata to merge")
    per_profdata = Path(str(per_raw))
    if not per_profdata.exists():
        raise RuntimeError(f"Missing per-harness profdata for {safe_hid}: {per_profdata}")

    with _FileLock(lock_path):
        tx_dir.mkdir(parents=True, exist_ok=True)
        backup_dir.mkdir(parents=True, exist_ok=True)
        merge_tmp_dir.mkdir(parents=True, exist_ok=True)

        tx_file = tx_dir / f"{safe_hid}.json"
        backup_path = backup_dir / f"{safe_hid}.{int(time.time())}.{os.getpid()}.before.profdata"
        backup = _backup_final_profdata(final_profdata, backup_path)

        tmp_final = merge_tmp_dir / f"all_corpora.{safe_hid}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp.profdata"
        tx = {
            "status": "merging",
            "safe_harness_id": safe_hid,
            "harness_id": result.get("harness_id"),
            "per_harness_profdata": str(per_profdata),
            "final_profdata": str(final_profdata),
            "tmp_final_profdata": str(tmp_final),
            "backup_profdata": str(backup) if backup is not None else None,
            "started_at": _now_iso(),
        }
        _atomic_write_json(tx_file, tx)

        try:
            merge_inputs: List[Path] = []
            if final_profdata.exists():
                merge_inputs.append(final_profdata)
            merge_inputs.append(per_profdata)

            _merge_profiles_chunked(
                llvm_profdata,
                merge_inputs,
                tmp_final,
                chunk_size=merge_chunk_size,
                tmp_dir=merge_tmp_dir / f"chunked_{safe_hid}",
            )
            os.replace(tmp_final, final_profdata)
            _fsync_file(final_profdata)
            _fsync_parent(final_profdata)
        except BaseException:
            # If anything failed before the done mark, roll back to the
            # pre-merge cumulative file so a resume cannot double-merge.
            if backup is not None and backup.exists():
                os.replace(backup, final_profdata)
                _fsync_file(final_profdata)
                _fsync_parent(final_profdata)
            elif final_profdata.exists():
                _safe_unlink(final_profdata)
                _fsync_parent(final_profdata)
            _safe_unlink(tmp_final)
            raise

        # The done mark is the resume boundary requested by the workflow:
        # only after this mark is written do we delete the per-harness profdata.
        _write_done_mark(marks_dir, result, final_profdata)

        _safe_unlink(per_profdata)
        if backup is not None:
            _safe_unlink(backup)
        _safe_unlink(tx_file)
        _safe_rmtree(merge_tmp_dir / f"chunked_{safe_hid}")
        _safe_unlink(tmp_final)


# ---------------- state management ----------------


def _load_json_if_exists(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _is_done_or_empty(mark: Optional[Dict[str, object]], final_profdata: Path) -> bool:
    if not mark:
        return False
    status = mark.get("status")
    if status == "skipped_empty":
        return True
    if status == "done":
        # Done means merged into cumulative, so the cumulative file must exist.
        return final_profdata.exists()
    return False


def _cleanup_orphans(
    *,
    out_dir: Path,
    profraw_dir: Path,
    tmp_dir: Path,
    per_harness_dir: Path,
    keep_failed_profiles: bool,
) -> None:
    if keep_failed_profiles:
        return
    _safe_rmtree(profraw_dir)
    _safe_rmtree(tmp_dir)
    profraw_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Per-harness profdata is temporary. If it was not already consumed into
    # final_profdata and marked done, rerunning the harness is safer than
    # merging a stale file that might not match its mark.
    if per_harness_dir.exists():
        for p in per_harness_dir.glob("*.profdata"):
            _safe_unlink(p)


def _reset_generated_state(paths: Sequence[Path]) -> None:
    for p in paths:
        if p.is_file() or p.is_symlink():
            _safe_unlink(p)
        else:
            _safe_rmtree(p)


def _auto_jobs() -> int:
    cpu = os.cpu_count() or 1
    # With coverage instrumentation, llvm-profdata, and rss limits, disk and RAM
    # usually become bottlenecks before CPU. On a 128-core machine this defaults
    # to 32 concurrent harnesses; pass --jobs 64 or --jobs 128 if the machine has
    # enough memory and IO headroom.
    if cpu >= 64:
        return min(32, cpu)
    return max(1, min(16, cpu // 2 or 1))


# ---------------- main ----------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--worker":
        return _worker_main(argv[1:])

    ap = argparse.ArgumentParser(
        description=(
            "Replay all discovered corpora through harness TestOneInput, "
            "merge coverage with bounded disk usage, and support resumable "
            "per-harness progress marks."
        )
    )
    ap.add_argument("--root", required=True, help="screen run root")
    ap.add_argument("--harnesses_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--llvm_cov", default="llvm-cov")
    ap.add_argument("--llvm_profdata", default="llvm-profdata")
    ap.add_argument("--primary_object", required=True)
    ap.add_argument("--extra_object", action="append", default=[])
    ap.add_argument("--ignore_filename_regex", default=None)
    ap.add_argument(
        "--pattern",
        action="append",
        default=[],
        help="Additional glob pattern under --root. Use {harness_id} placeholder.",
    )
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--merge_chunk_size", type=int, default=64)
    ap.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="Parallel harness jobs. 0 means auto; on a 128-core host auto defaults to 32.",
    )
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--disable_harness_mutation", action="store_true", default=False)
    ap.add_argument("--keep_empty_harnesses", action="store_true")
    ap.add_argument(
        "--cov_replay_extra",
        default=None,
        help='libFuzzer flags forwarded to replay workers, e.g. "-rss_limit_mb=8192 -malloc_limit_mb=8192".',
    )
    ap.add_argument("--replay_extra", default=None, help="Alias for --cov_replay_extra.")
    ap.add_argument(
        "--rerun_failed",
        action="store_true",
        help="Retry harnesses whose mark is failed. By default failed harnesses are kept as failed and skipped on resume.",
    )
    ap.add_argument(
        "--fail_fast",
        action="store_true",
        help="Stop scheduling new work after the first failed harness result.",
    )
    ap.add_argument(
        "--keep_failed_profiles",
        action="store_true",
        help="Keep partial per-harness temp/profraw files after failures for debugging. Default is to delete them.",
    )
    ap.add_argument(
        "--reset_state",
        action="store_true",
        help="Delete generated state/final profdata and rebuild from scratch. Use this instead of manually deleting mark files.",
    )
    ap.add_argument(
        "--no_clean_orphan_tmp",
        action="store_true",
        help="Do not clean orphan tmp/profraw/per-harness files at startup.",
    )
    ap.add_argument("--quiet", action="store_true", help="Reduce progress logging.")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir).resolve()
    profraw_dir = out_dir / "profraw"
    manifests_dir = out_dir / "input_manifests"
    per_harness_dir = out_dir / "per_harness"
    tmp_dir = out_dir / "tmp"
    progress_dir = out_dir / "progress"
    state_dir = out_dir / "state"
    marks_dir = state_dir / "marks"
    tx_dir = state_dir / "merge_transactions"
    backup_dir = state_dir / "merge_backups"
    cumulative_merge_tmp_dir = state_dir / "merge_tmp"
    lock_path = state_dir / "final_profdata.lock"

    final_profdata = out_dir / "all_corpora.profdata"
    summary_path = out_dir / "summary.json"

    if args.reset_state:
        _reset_generated_state(
            [
                final_profdata,
                summary_path,
                profraw_dir,
                manifests_dir,
                per_harness_dir,
                tmp_dir,
                progress_dir,
                state_dir,
            ]
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    for d in [
        profraw_dir,
        manifests_dir,
        per_harness_dir,
        tmp_dir,
        progress_dir,
        state_dir,
        marks_dir,
        tx_dir,
        backup_dir,
        cumulative_merge_tmp_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    _recover_pending_transactions(
        final_profdata=final_profdata,
        marks_dir=marks_dir,
        tx_dir=tx_dir,
    )

    if not args.no_clean_orphan_tmp:
        _cleanup_orphans(
            out_dir=out_dir,
            profraw_dir=profraw_dir,
            tmp_dir=tmp_dir,
            per_harness_dir=per_harness_dir,
            keep_failed_profiles=bool(args.keep_failed_profiles),
        )

    harness_map, harness_ids = _load_harnesses_json(Path(args.harnesses_json).resolve())
    patterns = list(DEFAULT_SOURCE_PATTERNS) + list(args.pattern or [])

    jobs = int(args.jobs) if int(args.jobs) > 0 else _auto_jobs()
    jobs = max(1, jobs)

    tasks: List[Dict[str, object]] = []
    skipped_done = 0
    skipped_failed = 0
    skipped_missing = 0

    for harness_id in harness_ids:
        harness_path = harness_map.get(harness_id) or harness_map.get(harness_id.strip())
        safe_hid = _safe_harness_file_id(harness_id)
        mark_path = marks_dir / f"{safe_hid}.json"
        mark = _load_json_if_exists(mark_path)

        if harness_path is None:
            skipped_missing += 1
            missing = {
                "status": "failed",
                "harness_id": harness_id,
                "safe_harness_id": safe_hid,
                "harness_path": None,
                "error": "harness_id not found in harnesses_json map",
                "failed_at": _now_iso(),
            }
            _atomic_write_json(mark_path, missing)
            continue

        if _is_done_or_empty(mark, final_profdata):
            skipped_done += 1
            continue

        if mark and mark.get("status") == "failed" and not args.rerun_failed:
            skipped_failed += 1
            continue

        tasks.append({"harness_id": harness_id, "harness_path": str(harness_path)})

    cfg: Dict[str, object] = {
        "root": str(root),
        "out_dir": str(out_dir),
        "profraw_dir": str(profraw_dir),
        "manifests_dir": str(manifests_dir),
        "per_harness_dir": str(per_harness_dir),
        "tmp_dir": str(tmp_dir),
        "progress_dir": str(progress_dir),
        "marks_dir": str(marks_dir),
        "patterns": patterns,
        "batch_size": int(args.batch_size),
        "merge_chunk_size": int(args.merge_chunk_size),
        "timeout": int(args.timeout),
        "python": str(args.python),
        "llvm_profdata": str(args.llvm_profdata),
        "llvm_cov": str(args.llvm_cov),
        "primary_object": str(Path(args.primary_object).resolve()),
        "extra_object": [str(Path(x).resolve()) for x in args.extra_object],
        "ignore_filename_regex": args.ignore_filename_regex,
        "disable_harness_mutation": bool(args.disable_harness_mutation),
        "keep_empty_harnesses": bool(args.keep_empty_harnesses),
        "keep_failed_profiles": bool(args.keep_failed_profiles),
        "cov_replay_extra": args.cov_replay_extra,
        "replay_extra": args.replay_extra,
        "verbose": not bool(args.quiet),
    }

    _log(
        "prepared replay: "
        f"total_harnesses={len(harness_ids)} runnable={len(tasks)} "
        f"skip_done={skipped_done} skip_failed={skipped_failed} "
        f"skip_missing={skipped_missing} jobs={jobs}"
    )

    completed = 0
    merged = 0
    failed = 0
    empty = 0
    no_profiles = 0
    submitted = len(tasks)

    # Submit all runnable harnesses. Each job internally runs batches one at a
    # time and immediately merges/deletes profraw, so the upper bound for live
    # profraw files is roughly --jobs times the number of processes spawned by
    # one harness batch.
    with cf.ProcessPoolExecutor(max_workers=jobs) as ex:
        future_to_task = {ex.submit(_harness_replay_job, task, cfg): task for task in tasks}

        for fut in cf.as_completed(future_to_task):
            task = future_to_task[fut]
            safe_hid = _safe_harness_file_id(str(task["harness_id"]))
            completed += 1

            try:
                result = fut.result()
            except BaseException as e:
                failed += 1
                result = {
                    "status": "failed",
                    "harness_id": task["harness_id"],
                    "safe_harness_id": safe_hid,
                    "harness_path": task.get("harness_path"),
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                    "failed_at": _now_iso(),
                }

            status = str(result.get("status"))

            if status == "profdata_ready":
                try:
                    _merge_one_harness_to_cumulative(
                        result=result,
                        final_profdata=final_profdata,
                        marks_dir=marks_dir,
                        tx_dir=tx_dir,
                        backup_dir=backup_dir,
                        merge_tmp_dir=cumulative_merge_tmp_dir,
                        llvm_profdata=str(args.llvm_profdata),
                        merge_chunk_size=int(args.merge_chunk_size),
                        lock_path=lock_path,
                    )
                    merged += 1
                    _log(
                        f"[global] merged {safe_hid}; progress={completed}/{submitted} "
                        f"merged={merged} failed={failed} empty={empty} no_profiles={no_profiles}"
                    )
                except BaseException as e:
                    failed += 1
                    result = {
                        **result,
                        "status": "failed",
                        "merge_error": repr(e),
                        "merge_traceback": traceback.format_exc(),
                        "failed_at": _now_iso(),
                    }
                    _atomic_write_json(marks_dir / f"{safe_hid}.json", result)
                    _log(
                        f"[global] merge failed {safe_hid}; progress={completed}/{submitted} "
                        f"failed={failed}: {repr(e)}"
                    )
                    if args.fail_fast:
                        break

            elif status == "skipped_empty":
                empty += 1
                _write_terminal_mark(marks_dir, result, status="skipped_empty")
                _log(
                    f"[global] empty {safe_hid}; progress={completed}/{submitted} "
                    f"merged={merged} failed={failed} empty={empty}"
                )

            elif status == "no_profiles":
                no_profiles += 1
                _write_terminal_mark(marks_dir, result, status="no_profiles")
                _log(
                    f"[global] no profiles {safe_hid}; progress={completed}/{submitted} "
                    f"merged={merged} failed={failed} no_profiles={no_profiles}"
                )

            else:
                failed += 1
                result = {**result, "status": "failed"}
                _atomic_write_json(marks_dir / f"{safe_hid}.json", result)
                _log(
                    f"[global] failed {safe_hid}; progress={completed}/{submitted} "
                    f"merged={merged} failed={failed}: {result.get('error')}"
                )
                if args.fail_fast:
                    break

    # Load terminal marks as the authoritative summary.
    per_harness: Dict[str, object] = {}
    status_counts: Dict[str, int] = {}
    total_profraw_count = 0

    for harness_id in harness_ids:
        safe_hid = _safe_harness_file_id(harness_id)
        mark = _load_json_if_exists(marks_dir / f"{safe_hid}.json")
        if mark is None:
            mark = {
                "status": "not_run",
                "harness_id": harness_id,
                "safe_harness_id": safe_hid,
            }
        status = str(mark.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
        try:
            total_profraw_count += int(mark.get("profraw_count") or 0)
        except Exception:
            pass
        per_harness[harness_id] = mark

    final_totals: Optional[Dict[str, int]] = None
    if final_profdata.exists():
        final_totals = _cov_summary_lcov(
            llvm_cov=str(args.llvm_cov),
            profdata=final_profdata,
            primary_object=Path(args.primary_object).resolve(),
            extra_objects=[Path(x).resolve() for x in args.extra_object],
            ignore_filename_regex=args.ignore_filename_regex,
        )

    summary = {
        "root": str(root),
        "harnesses_json": str(Path(args.harnesses_json).resolve()),
        "out_dir": str(out_dir),
        "final_profdata": str(final_profdata) if final_profdata.exists() else None,
        "final_totals": final_totals,
        "status_counts": status_counts,
        "profraw_count": total_profraw_count,
        "jobs": jobs,
        "batch_size": int(args.batch_size),
        "merge_chunk_size": int(args.merge_chunk_size),
        "disable_harness_mutation": bool(args.disable_harness_mutation),
        "cov_replay_extra": (
            args.cov_replay_extra
            or args.replay_extra
            or "-rss_limit_mb=8192 -malloc_limit_mb=8192"
        ),
        "source_patterns": patterns,
        "state_dir": str(state_dir),
        "marks_dir": str(marks_dir),
        "progress_dir": str(progress_dir),
        "per_harness": per_harness,
        "generated_at": _now_iso(),
    }

    _atomic_write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if final_profdata.exists():
        return 0

    raise RuntimeError(
        "No cumulative profdata was produced. Check failed/no_profiles marks under "
        f"{marks_dir} and progress logs under {progress_dir}."
    )


if __name__ == "__main__":
    raise SystemExit(main())
