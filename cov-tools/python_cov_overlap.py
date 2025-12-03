#!/usr/bin/env python3
# overlap_cov.py
# 计算两次 coverage 运行的“重复率” = 交集 / 并集（行或分支）
# 依赖：pip install coverage

import argparse, os, sys, fnmatch
from typing import Iterable, Tuple, Set, Dict
import coverage  # type: ignore

def canonicalize(path: str, root: str | None) -> str:
    p = os.path.realpath(path)
    if root:
        try:
            return os.path.relpath(p, os.path.realpath(root))
        except ValueError:
            return p
    return p

def match_includes(path: str, includes: list[str] | None, excludes: list[str] | None) -> bool:
    name = path
    if includes:
        ok = any(fnmatch.fnmatch(name, pat) for pat in includes)
        if not ok: 
            return False
    if excludes:
        if any(fnmatch.fnmatch(name, pat) for pat in excludes):
            return False
    return True

def load_data(data_file: str) -> coverage.CoverageData:
    cov = coverage.Coverage(data_file=data_file)
    cov.load()
    return cov.get_data()

def collect_items(
    data: coverage.CoverageData, 
    mode: str, 
    includes: list[str] | None, 
    excludes: list[str] | None, 
    root: str | None
) -> Tuple[Set[Tuple[str,int]] | Set[Tuple[str,int,int]], Dict[str, Set[int] | Set[Tuple[int,int]]]]:
    """
    返回：
      - overall: {(file,line)} 或 {(file,from,to)}
      - per_file: {file: {line,...} 或 {(from,to),...}}
    """
    overall: Set = set()
    per_file: Dict[str, Set] = {}

    files = list(data.measured_files())
    for f in files:
        cf = canonicalize(f, root)
        if not match_includes(cf, includes, excludes):
            continue

        if mode == "line":
            lines = data.lines(f) or []
            s = set(int(x) for x in lines)
            if not s:
                continue
            per_file[cf] = s
            overall.update((cf, ln) for ln in s)
        else:  # branch
            if not data.has_arcs():
                raise RuntimeError("该 .coverage 文件中没有分支数据（arcs）。请在采集时开启 branch=True 或 `coverage run --branch`。")
            arcs = data.arcs(f) or []
            # 过滤入口/出口 arcs（包含 None 的 arc）
            s = set((int(a), int(b)) for (a,b) in arcs if a is not None and b is not None)
            if not s:
                continue
            per_file[cf] = s
            overall.update((cf, a, b) for (a,b) in s)

    return overall, per_file

def jaccard(a: Set, b: Set) -> float:
    if not a and not b:
        return 1.0  # 两者都为空，定义为完全一致
    u = a | b
    if not u:
        return 0.0
    i = a & b
    return len(i) / len(u)

def main():
    ap = argparse.ArgumentParser(description="计算两次 coverage 数据的重复率（交并比）")
    ap.add_argument("data1", help="第一次运行的 .coverage 文件路径")
    ap.add_argument("data2", help="第二次运行的 .coverage 文件路径")
    ap.add_argument("--mode", choices=["line", "branch"], default="line", help="比较维度：行 或 分支（arcs）")
    ap.add_argument("--include", action="append", dest="includes", help="包含的文件通配（可多次），如 'src/**/*.py'")
    ap.add_argument("--exclude", action="append", dest="excludes", help="排除的文件通配（可多次）")
    ap.add_argument("--root", help="把文件路径相对化到该根目录，以减少绝对路径差异")
    ap.add_argument("--per-file", action="store_true", help="输出逐文件的重复率统计")
    ap.add_argument("--csv", help="把逐文件统计输出到 CSV 文件")
    args = ap.parse_args()

    data1 = load_data(args.data1)
    data2 = load_data(args.data2)

    overall1, per1 = collect_items(data1, args.mode, args.includes, args.excludes, args.root)
    overall2, per2 = collect_items(data2, args.mode, args.includes, args.excludes, args.root)

    overall_score = jaccard(overall1, overall2)
    i_cnt = len(overall1 & overall2)
    u_cnt = len(overall1 | overall2)

    unit = "行" if args.mode == "line" else "分支"
    print(f"[整体] 重复率（Jaccard，{unit}）= {overall_score:.4f}  |  交集={i_cnt}  并集={u_cnt}")

    if args.per_file or args.csv:
        import csv
        rows = []
        all_files = sorted(set(per1.keys()) | set(per2.keys()))
        for f in all_files:
            s1 = per1.get(f, set())
            s2 = per2.get(f, set())
            score = jaccard(s1, s2)
            inter = len(s1 & s2)
            uni = len(s1 | s2)
            rows.append((f, score, inter, uni, len(s1), len(s2)))

        if args.per_file:
            print("\n[逐文件]")
            print("文件, 重复率, 交集, 并集, 运行1计数, 运行2计数")
            for f, score, inter, uni, c1, c2 in rows:
                print(f"{f}, {score:.4f}, {inter}, {uni}, {c1}, {c2}")

        if args.csv:
            with open(args.csv, "w", newline="", encoding="utf-8") as fh:
                cw = csv.writer(fh)
                cw.writerow(["file", "jaccard", "intersection", "union", "count_run1", "count_run2"])
                cw.writerows(rows)
            print(f"\n已写出 CSV：{args.csv}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[错误] {e}", file=sys.stderr)
        sys.exit(2)
