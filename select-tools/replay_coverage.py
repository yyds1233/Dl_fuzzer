#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import concurrent.futures
import concurrent.futures as futures
import datetime
import os
import re
import subprocess
import sys
from pathlib import Path

HARNESS_RE = re.compile(r"^harness(\d+)\.py$")

def is_nonempty_dir(p: Path) -> bool:
    return p.is_dir() and any(p.iterdir())

def list_corpus_files(corpus_dir: Path):
    """按文件名排序返回 corpus 里的文件路径列表（仅文件）。"""
    files = [p for p in corpus_dir.iterdir() if p.is_file()]
    files.sort(key=lambda x: x.name)
    return files

def find_harnesses(fuzz_root: Path):
    """
    递归查找所有 harnessN.py，返回条目：
    {
      "harness_idx": int,
      "harness_file": Path(绝对),
      "parent_dir": Path(绝对),
      "parent_rel": Path(相对 fuzz_root 的相对路径)
    }
    """
    entries = []
    for py in fuzz_root.rglob("harness*.py"):
        m = HARNESS_RE.match(py.name)
        if not m:
            continue
        idx = int(m.group(1))
        parent = py.parent
        parent_rel = parent.relative_to(fuzz_root)
        entries.append({
            "harness_idx": idx,
            "harness_file": py.resolve(),
            "parent_dir": parent.resolve(),
            "parent_rel": parent_rel,
        })
    return entries

def ensure_dest_dirs(out_root: Path, result_dir: Path, parent_rel: Path):
    """
    创建结果目录里与源目录相同的相对路径
    """
    dest_parent = (result_dir / parent_rel).resolve()
    dest_parent.mkdir(parents=True, exist_ok=True)
    return dest_parent

def glob_profraw_parts(dest_parent: Path, idx: int):
    """返回 C 覆盖阶段产生的所有分片 profraw 列表。"""
    pattern = f"harn_{idx}_"  # 前缀，后面会接 %p 展开出的 pid
    return sorted([p for p in dest_parent.glob(f"{pattern}*.profraw") if p.is_file() and p.stat().st_size > 0])

def run_one(item, fuzz_root: Path, result_dir: Path) -> dict:
    """
    执行单个 harness 的回放：先跑 C 覆盖，再跑 Python 覆盖。
    返回 {"status":"ok/skip/fail", "msg": "...", "idx": int, "rel": Path}
    """
    idx = item["harness_idx"]
    parent_dir = item["parent_dir"]
    parent_rel = item["parent_rel"]
    harness_file = item["harness_file"]

    corpus_dir = parent_dir / f"corpus_{idx}"

    # 只回放有有效语料的 harness
    if not is_nonempty_dir(corpus_dir):
        return {"status": "skip", "msg": f"corpus_{idx} 不存在或为空", "idx": idx, "rel": parent_rel}

    # 收集并排序种子文件；Atheris 建议将每个文件作为位置参数传入，并设置 -atheris_runs=1+N
    seeds = list_corpus_files(corpus_dir)
    if not seeds:
        return {"status": "skip", "msg": f"corpus_{idx} 没有文件", "idx": idx, "rel": parent_rel}
    atheris_runs = 1 + len(seeds)

    dest_parent = ensure_dest_dirs(out_root=None, result_dir=result_dir, parent_rel=parent_rel)

    # 目标文件路径
    # C 覆盖：使用 %p 以避免多进程/DSO 写同一文件导致 0 字节
    profraw_pattern = dest_parent / f"harn_{idx}_%p.profraw"
    merged_profdata = dest_parent / f"harn_{idx}.profdata"  # 合并结果（供 llvm-cov 用）
    final_profraw_single = dest_parent / f"harn_{idx}.profraw"  # 若仅产生一个 raw，就重命名为这个
    cov_data_path = dest_parent / f".coverage.{idx}"       # Python 覆盖（隐藏文件）
    log_path = dest_parent / f"harn_{idx}.log"

    ############################################
    # 第一阶段：C 覆盖（不使用 coverage.py）
    ############################################
    env_c = os.environ.copy()
    env_c["LLVM_PROFILE_FILE"] = str(profraw_pattern)

    cmd_c = [
        sys.executable,
        str(harness_file), 
        f"corpus_{idx}",
        "-runs=0",
        "-print_final_stats=1",
    ]

    ############################################
    # 第二阶段：Python 覆盖（不再动 LLVM_*）
    ############################################
    env_py = os.environ.copy()
    env_py["COVERAGE_FILE"] = str(cov_data_path)

    cmd_py = [
        sys.executable, "-m", "coverage",
        "run",
        "--branch",
        str(harness_file),
        *[str(s) for s in seeds],
        f"-atheris_runs={atheris_runs}",
        "-print_final_stats=1",
    ]

    try:
        with open(log_path, "w", encoding="utf-8") as lg:
            lg.write(f"# Working dir: {parent_dir}\n")
            lg.write(f"# === C COVERAGE PHASE ===\n")
            lg.write(f"# LLVM_PROFILE_FILE={env_c['LLVM_PROFILE_FILE']}\n")
            lg.write(f"# CMD: {' '.join(map(str, cmd_c))}\n")
            lg.write(f"# Seeds({len(seeds)}):\n")
            for s in seeds[:50]:
                lg.write(f"  - {s}\n")
            if len(seeds) > 50:
                lg.write(f"  ... (+{len(seeds)-50} more)\n")
            lg.write(f"# -atheris_runs={atheris_runs}\n\n")
            lg.flush()

            # 运行 C 覆盖阶段
            proc_c = subprocess.run(
                cmd_c,
                cwd=str(parent_dir),
                env=env_c,
                stdout=lg,
                stderr=subprocess.STDOUT,
                check=False,
            )

            lg.write(f"\n# C phase exit code: {proc_c.returncode}\n")

            # 收集 C 覆盖分片
            parts = glob_profraw_parts(dest_parent, idx)
            lg.write(f"# Found profraw parts: {len(parts)}\n")
            for p in parts[:20]:
                lg.write(f"  - {p}\n")
            if len(parts) > 20:
                lg.write(f"  ... (+{len(parts)-20} more)\n")

            # 若只有一个分片，也生成 .profdata，并删除所有 .profraw 分片
            if len(parts) >= 1:
                try:
                    # 合并所有分片（即使只有一个）
                    merge_cmd = ["llvm-profdata", "merge", "-sparse", "-o", str(merged_profdata)] + [str(p) for p in parts]
                    lg.write(f"# Merging to profdata: {' '.join(merge_cmd)}\n")
                    subprocess.run(merge_cmd, stdout=lg, stderr=subprocess.STDOUT, check=False)

                    # 删除所有原始 .profraw 文件
                    for p in parts:
                        try:
                            p.unlink()
                        except Exception as e:
                            lg.write(f"# 删除 profraw 文件失败 {p}: {e}\n")
                    lg.write(f"# 已生成并保留合并产物: {merged_profdata}\n")
                except Exception as e:
                    lg.write(f"# llvm-profdata merge failed: {e}\n")
            else:
                lg.write("# 未找到任何 profraw 分片，无法合并。\n")

            lg.write("\n# === PY COVERAGE PHASE ===\n")
            lg.write(f"# COVERAGE_FILE={env_py['COVERAGE_FILE']}\n")
            lg.write(f"# CMD: {' '.join(map(str, cmd_py))}\n\n")
            lg.flush()

            # 运行 Python 覆盖阶段
            proc_py = subprocess.run(
                cmd_py,
                cwd=str(parent_dir),
                env=env_py,
                stdout=lg,
                stderr=subprocess.STDOUT,
                check=False,
            )

            lg.write(f"\n# PY phase exit code: {proc_py.returncode}\n")
            lg.flush()

        # 统一的结果校验
        missing = []

        # C 覆盖：允许两种有效情形
        parts = glob_profraw_parts(dest_parent, idx)
        c_ok = False
        if final_profraw_single.exists() and final_profraw_single.stat().st_size > 0:
            c_ok = True
        elif len(parts) >= 1:
            c_ok = True  # 分片存在即可；如需严格，可要求合并产物存在
        elif merged_profdata.exists() and merged_profdata.stat().st_size > 0:
            c_ok = True  # 也接受合并产物
        if not c_ok:
            missing.append("harn_%d.profraw(parts)" % idx)

        # Python 覆盖：.coverage.{idx} 必须非空（若落在工作目录，用户可用兜底搬运，但我们已固定 COVERAGE_FILE）
        if not cov_data_path.exists() or cov_data_path.stat().st_size == 0:
            missing.append(cov_data_path.name)

        if missing:
            return {
                "status": "fail",
                "msg": f"覆盖文件缺失或为空：{', '.join(missing)}（详见 {log_path}）",
                "idx": idx,
                "rel": parent_rel,
            }

        return {"status": "ok", "msg": "done", "idx": idx, "rel": parent_rel}

    except Exception as e:
        return {"status": "fail", "msg": f"异常：{e}", "idx": idx, "rel": parent_rel}

def main():
    parser = argparse.ArgumentParser(description="批量回放 fuzz harness 并收集 Python/C 覆盖")
    parser.add_argument("--fuzz-root", type=Path, default=Path("/root/torch-api-fuzz"),
                        help="fuzz 根目录（例如 /path/to/torch-api-fuzz）")
    parser.add_argument("--out-root", type=Path, default=Path("/root/cov-result"),
                        help="输出根目录（默认 /root/cov-result）")
    parser.add_argument("--workers", type=int, default=2,
                        help="并发 worker 数（默认=2）")
    args = parser.parse_args()

    fuzz_root = args.fuzz_root.resolve()
    out_root = args.out_root.resolve()

    if not fuzz_root.exists():
        print(f"[Error] fuzz 根目录不存在：{fuzz_root}", file=sys.stderr)
        sys.exit(2)

    out_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = out_root / f"result_{ts}"
    result_dir.mkdir(parents=True, exist_ok=True)

    latest_file = out_root / "latest_result_dir.txt"
    with open(latest_file, "w", encoding="utf-8") as f:
        f.write(str(result_dir) + "\n")

    print(f"[Info] 本次结果目录：{result_dir}")
    print(f"[Info] 也已写入：{latest_file}")

    harnesses = find_harnesses(fuzz_root)
    if not harnesses:
        print("[Warning] 未发现任何 harnessN.py 文件。")
        return

    for h in harnesses:
        ensure_dest_dirs(out_root=None, result_dir=result_dir, parent_rel=h["parent_rel"])

    print(f"[Info] 发现 {len(harnesses)} 个 harness，开始并行回放（workers={args.workers}）...")

    ok, skip, fail = 0, 0, 0
    fail_list = []
    skip_list = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_one, h, fuzz_root, result_dir) for h in harnesses]
        for fut in concurrent.futures.as_completed(futs):
            r = fut.result()
            rel = str(r["rel"])
            idx = r["idx"]
            if r["status"] == "ok":
                ok += 1
                print(f"[OK] {rel}/harness{idx}.py")
            elif r["status"] == "skip":
                skip += 1
                skip_list.append((rel, idx, r["msg"]))
                print(f"[SKIP] {rel}/harness{idx}.py -> {r['msg']}")
            else:
                fail += 1
                fail_list.append((rel, idx, r["msg"]))
                print(f"[FAIL] {rel}/harness{idx}.py -> {r['msg']}")

    print("\n========== Summary ==========")
    print(f"result：{result_dir}")
    print(f"success：{ok}，skip：{skip}，fail：{fail}")
    if skip_list:
        print("\n[跳过详情]")
        for rel, idx, msg in skip_list:
            print(f"- {rel}/harness{idx}.py : {msg}")
    if fail_list:
        print("\n[失败详情]")
        for rel, idx, msg in fail_list:
            print(f"- {rel}/harness{idx}.py : {msg}")

if __name__ == "__main__":
    main()
