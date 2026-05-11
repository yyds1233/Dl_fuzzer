#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import tempfile


DEFAULT_IGNORE_FILENAME_REGEX = r"(^|/)(third_party)(/|$)"


def run_cmd(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n\nSTDOUT:\n{p.stdout}\n\nSTDERR:\n{p.stderr}"
        )
    return p.stdout


def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_files(obj):
    # 期望结构：{"data":[{"files":[{...}, ...]}]}
    for bundle in obj.get("data", []):
        for f in bundle.get("files", []):
            yield f


def stable_file(f):
    return f.get("filename") or f.get("name") or "<unknown>"


def _int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


def branch_keys_from_file(f):
    """
    生成“命中过的边”的键：
    - 对象形态：key=(file,line,col,edge_idx)；edge_idx=0/1（false/true）或 counts 中的索引
    - 列表形态：key=(file,line,col,edge_idx)
    """
    file_path = stable_file(f)
    keys = set()
    branches = f.get("branches") or []
    for b in branches:
        if isinstance(b, dict):
            line = b.get("line")
            col = b.get("column")
            counts = None

            if "true_count" in b or "false_count" in b:
                counts = [_int(b.get("false_count", 0)), _int(b.get("true_count", 0))]
            elif isinstance(b.get("branch_counts"), list):
                counts = [_int(x) for x in b["branch_counts"]]
            elif isinstance(b.get("counts"), list):
                counts = [_int(x) for x in b["counts"]]

            if counts is None:
                continue

            for eidx, c in enumerate(counts):
                if c > 0:
                    keys.add((file_path, line, col, eidx))
            continue

        if isinstance(b, list) and len(b) >= 3:
            line = b[0]
            col = b[1]
            counts = []
            for x in b[2:]:
                try:
                    counts.append(int(x))
                except Exception:
                    continue
            for eidx, c in enumerate(counts):
                if c > 0:
                    keys.add((file_path, line, col, eidx))

    return keys


def region_keys_from_file(f):
    """
    生成“命中过的区域”的键：
    - 对象形态：{line_start,column_start,line_end,column_end,count}
    - 列表形态：[startLine,startCol,endLine,endCol,count,...]
    - 若只有 segments，可退化为“点键”（end=None）
    """
    file_path = stable_file(f)
    keys = set()

    regions = f.get("regions")
    if isinstance(regions, list) and regions:
        for r in regions:
            if isinstance(r, dict):
                sl, sc = r.get("line_start"), r.get("column_start")
                el, ec = r.get("line_end"), r.get("column_end")
                cnt = _int(r.get("count", 0))
            elif isinstance(r, list) and len(r) >= 5:
                sl, sc, el, ec, cnt = r[:5]
                cnt = _int(cnt)
            else:
                continue

            if cnt > 0:
                keys.add((file_path, sl, sc, el, ec))
        return keys

    segments = f.get("segments")
    if isinstance(segments, list):
        for s in segments:
            if not (isinstance(s, list) and len(s) >= 5):
                continue
            line, col, cnt, has_cnt, _ = s[:5]
            cnt = _int(cnt)
            try:
                has_cnt = bool(has_cnt)
            except Exception:
                has_cnt = True
            if has_cnt and cnt > 0:
                keys.add((file_path, line, col, None, None))
    return keys


def line_keys_from_file(f):
    """
    生成“命中过的行”的键：
    - 优先用 regions：count>0 的 region 覆盖到的所有行
    - 若无 regions，则退化用 segments：hasCount 且 count>0 的 line
    key=(file,line)
    """
    file_path = stable_file(f)
    keys = set()

    regions = f.get("regions")
    if isinstance(regions, list) and regions:
        for r in regions:
            if isinstance(r, dict):
                sl = r.get("line_start")
                el = r.get("line_end")
                cnt = _int(r.get("count", 0))
            elif isinstance(r, list) and len(r) >= 5:
                sl, _, el, _, cnt = r[:5]
                cnt = _int(cnt)
            else:
                continue

            if cnt > 0 and sl is not None and el is not None:
                try:
                    sl = int(sl)
                    el = int(el)
                except Exception:
                    continue
                for line in range(sl, el + 1):
                    keys.add((file_path, line))
        return keys

    segments = f.get("segments")
    if isinstance(segments, list):
        for s in segments:
            if not (isinstance(s, list) and len(s) >= 5):
                continue
            line, _, cnt, has_cnt, _ = s[:5]
            cnt = _int(cnt)
            try:
                has_cnt = bool(has_cnt)
            except Exception:
                has_cnt = True
            if has_cnt and cnt > 0:
                try:
                    keys.add((file_path, int(line)))
                except Exception:
                    pass
    return keys


def load_branch_keys(path):
    obj = load_json(path)
    keys = set()
    for f in iter_files(obj):
        keys |= branch_keys_from_file(f)
    return keys


def load_region_keys(path):
    obj = load_json(path)
    keys = set()
    for f in iter_files(obj):
        keys |= region_keys_from_file(f)
    return keys


def load_line_keys(path):
    obj = load_json(path)
    keys = set()
    for f in iter_files(obj):
        keys |= line_keys_from_file(f)
    return keys


def jaccard(a, b):
    if not a and not b:
        return 1.0
    u = a | b
    if not u:
        return 1.0
    return len(a & b) / len(u)


def print_overlap(title, A, B, unit):
    print(f"== {title} ==")
    print(f"mine {unit}: {len(A)}  baseline {unit}: {len(B)}")
    print(f"Intersection: {len(A & B)}  Union: {len(A | B)}")
    print(f"Jaccard: {jaccard(A, B):.4f}")
    print(f"Only in mine: {len(A - B)}  Only in baseline: {len(B - A)}")
    print()


def export_cov_json(
    llvm_cov,
    binary,
    profdata,
    out_json,
    objects=None,
    ignore_filename_regex=None,
):
    cmd = [llvm_cov, "export", f"-instr-profile={profdata}"]

    if ignore_filename_regex:
        cmd.append(f"-ignore-filename-regex={ignore_filename_regex}")

    cmd.append(binary)

    if objects:
        for obj in objects:
            cmd.extend(["-object", obj])

    output = run_cmd(cmd)
    with open(out_json, "w", encoding="utf-8") as f:
        f.write(output)


def main():
    ap = argparse.ArgumentParser(
        description="Compare two LLVM profdata files using both llvm-profdata overlap and coverage-set overlap."
    )
    ap.add_argument("binary", help="Instrumented binary (with coverage mapping)")
    ap.add_argument("mine", help="mine.profdata")
    ap.add_argument("baseline", help="baseline.profdata")
    ap.add_argument("--llvm-profdata", default="llvm-profdata", help="Path to llvm-profdata")
    ap.add_argument("--llvm-cov", default="llvm-cov", help="Path to llvm-cov")
    ap.add_argument(
        "--object",
        action="append",
        default=[],
        help="Additional object file for llvm-cov export (can repeat)",
    )
    ap.add_argument(
        "--ignore-filename-regex",
        default=DEFAULT_IGNORE_FILENAME_REGEX,
        help="Regex for source files to ignore during llvm-cov export",
    )
    ap.add_argument(
        "--keep-json",
        action="store_true",
        help="Keep exported JSON files in current directory",
    )
    args = ap.parse_args()

    for p in [args.binary, args.mine, args.baseline]:
        if not os.path.exists(p):
            print(f"File not found: {p}", file=sys.stderr)
            sys.exit(2)

    print("== llvm-profdata overlap ==")
    try:
        overlap_out = run_cmd([args.llvm_profdata, "overlap", args.mine, args.baseline])
        print(overlap_out.strip())
    except Exception as e:
        print(f"[WARN] llvm-profdata overlap failed:\n{e}")
    print()

    if args.keep_json:
        j1 = os.path.abspath("mine.export.json")
        j2 = os.path.abspath("baseline.export.json")
        export_cov_json(
            args.llvm_cov,
            args.binary,
            args.mine,
            j1,
            args.object,
            args.ignore_filename_regex,
        )
        export_cov_json(
            args.llvm_cov,
            args.binary,
            args.baseline,
            j2,
            args.object,
            args.ignore_filename_regex,
        )
    else:
        td = tempfile.TemporaryDirectory()
        j1 = os.path.join(td.name, "mine.export.json")
        j2 = os.path.join(td.name, "baseline.export.json")
        export_cov_json(
            args.llvm_cov,
            args.binary,
            args.mine,
            j1,
            args.object,
            args.ignore_filename_regex,
        )
        export_cov_json(
            args.llvm_cov,
            args.binary,
            args.baseline,
            j2,
            args.object,
            args.ignore_filename_regex,
        )

    B1 = load_branch_keys(j1)
    B2 = load_branch_keys(j2)
    R1 = load_region_keys(j1)
    R2 = load_region_keys(j2)
    L1 = load_line_keys(j1)
    L2 = load_line_keys(j2)

    print_overlap("Branch overlap", B1, B2, "edges")
    print_overlap("Region overlap", R1, R2, "regions")
    print_overlap("Line overlap",   L1, L2, "lines")

    if args.keep_json:
        print(f"Exported JSON kept at:\n  {j1}\n  {j2}")


if __name__ == "__main__":
    main()
