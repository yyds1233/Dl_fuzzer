#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, argparse, sys
import coverage

def load_cov_data(path):
    # 兼容新版/旧版 coverage.py 数据读取
    data = coverage.CoverageData()
    if hasattr(data, "read_file"):
        data.read_file(path)
        return data
    cov = coverage.Coverage(data_file=path)
    cov.load()
    return cov.get_data()

def build_matched_file_list(data, inc_res, exc_res):
    matched = []
    for f in data.measured_files():
        f_posix = f.replace("\\", "/")
        if inc_res and not any(r.search(f_posix) for r in inc_res):
            continue
        if exc_res and any(r.search(f_posix) for r in exc_res):
            continue
        matched.append(f)
    return matched

def write_filtered_coverage_data(src_data, files, out_path):
    """
    可选：把筛选后的数据写成一个新的 .coverage 文件
    （仅包含指定 files 的行/弧信息）
    """
    new_data = coverage.CoverageData()
    add_lines = getattr(new_data, "add_lines", None)
    add_arcs  = getattr(new_data, "add_arcs", None)

    for f in files:
        lines = src_data.lines(f) or []
        arcs  = src_data.arcs(f) or []

        # coverage.py >= 5 有 add_lines/add_arcs；若缺失则退化为只写 arcs（或跳过）
        if add_lines and lines:
            add_lines({f: set(lines)})
        if add_arcs and arcs:
            add_arcs({f: set(arcs)})

    # 覆盖率数据需要包含 file tracer 名称，否则个别版本报告时会忽略
    # 尝试从原数据拷贝（如果可用）
    if hasattr(src_data, "file_tracers"):
        fts = src_data.file_tracers()
        if fts:
            # 只保留我们关心的文件的 tracer
            kept = {f: ft for f, ft in fts.items() if f in files}
            if hasattr(new_data, "add_file_tracers") and kept:
                new_data.add_file_tracers(kept)

    # 写文件
    if hasattr(new_data, "write_file"):
        new_data.write_file(out_path)
    else:
        # 旧版 fallback：通过 Coverage(data_file=...).get_data().[add_*] 再保存
        cov = coverage.Coverage(data_file=out_path)
        cov._init()
        cov._data = new_data  # 作为兼容的“低风险”用法
        cov.save()

def main():
    ap = argparse.ArgumentParser(description="按正则过滤已有 .coverage，并生成覆盖率报告/可选导出过滤数据")
    ap.add_argument("--cov", default=".coverage", help="输入 coverage 数据文件")
    ap.add_argument("--include", action="append", default=[], help="正则(可多次)，匹配需要统计的文件路径")
    ap.add_argument("--exclude", action="append", default=[], help="正则(可多次)，排除的文件路径")
    ap.add_argument("--html", default="", help="输出 HTML 报告目录（留空则不生成）")
    ap.add_argument("--xml", default="", help="输出 XML 文件路径（留空则不生成）")
    ap.add_argument("--json", default="", help="输出 JSON 报告文件（留空则不生成，需 coverage 版本支持）")
    ap.add_argument("--term", action="store_true", help="在终端打印覆盖率表格")
    ap.add_argument("--filtered-data", default="", help="导出过滤后的 .coverage（留空则不导出）")
    ap.add_argument("--fail-under", type=float, default=None, help="低于此总覆盖率则脚本返回非0退出码")
    ap.add_argument("--branch", action="store_true", help="以分支覆盖率为主（建议与你收集时保持一致）")
    args = ap.parse_args()

    if not os.path.exists(args.cov):
        sys.exit(f"[!] coverage data not found: {args.cov}")

    inc_res = [re.compile(p) for p in args.include] if args.include else []
    exc_res = [re.compile(p) for p in args.exclude] if args.exclude else []

    # 1) 读原始数据，计算目标文件集合
    src_data = load_cov_data(args.cov)
    matched_files = build_matched_file_list(src_data, inc_res, exc_res)
    if not matched_files:
        sys.exit("[!] No files matched by include/exclude patterns.")

    # 2) 用 Coverage 对 matched_files 直接生成报告
    cov = coverage.Coverage(data_file=args.cov, branch=args.branch or True)
    cov.load()

    total_percent = None  # 保存总体覆盖率（report 的返回值）

    # 终端报告
    if args.term:
        # morfs=matched_files 只对这部分文件报告
        total_percent = cov.report(morfs=matched_files)

    # HTML 报告
    if args.html:
        cov.html_report(morfs=matched_files, directory=args.html)

    # XML 报告（如 CI / Sonar / Jenkins 等）
    if args.xml:
        cov.xml_report(morfs=matched_files, outfile=args.xml)

    # JSON 报告（需要支持的 coverage 版本：>= 5.x/6.x，某些旧版无此 API）
    if args.json and hasattr(cov, "json_report"):
        cov.json_report(morfs=matched_files, outfile=args.json)

    # 3) 可选：导出过滤后的 .coverage 数据库
    if args.filtered_data:
        write_filtered_coverage_data(src_data, matched_files, args.filtered_data)
        print(f"[+] Wrote filtered coverage data: {args.filtered_data}")

    # 4) fail-under
    if args.fail_under is not None:
        # 若没打印终端报告，就临时跑一次以拿到总覆盖率
        if total_percent is None:
            total_percent = cov.report(morfs=matched_files, show_missing=False)
        if total_percent < args.fail_under:
            print(f"[!] Total coverage {total_percent:.2f}% < fail-under {args.fail_under}%")
            sys.exit(2)

    # 总结
    print(f"[+] Files matched: {len(matched_files)}")
    if total_percent is not None:
        print(f"[+] Total coverage (matched subset): {total_percent:.2f}%")

if __name__ == "__main__":
    main()

