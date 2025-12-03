#!/usr/bin/env python3
import json
import sys

def load_json(p):
    with open(p, "r") as f:
        return json.load(f)

def iter_files(obj):
    # 期望结构：{"data":[{"files":[{...}, ...]}]}
    for bundle in obj.get("data", []):
        for f in bundle.get("files", []):
            yield f

def stable_file(f):
    return f.get("filename") or f.get("name")

def _int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default

def branch_keys_from_file(f):
    """
    生成“命中过的边”的键：
    - 对象形态：key=(file,line,col,edge_idx)；edge_idx=0/1（false/true）
    - 列表形态：key=(file,line,col,edge_idx)；edge_idx = 在 counts 中的顺序索引
    """
    file_path = stable_file(f)
    keys = set()
    branches = f.get("branches") or []
    for b in branches:
        # 形态 1：字典
        if isinstance(b, dict):
            line = b.get("line")
            col = b.get("column")
            counts = None
            # a) true/false 字段
            if "true_count" in b or "false_count" in b:
                counts = [_int(b.get("false_count", 0)), _int(b.get("true_count", 0))]
            # b) branch_counts 或 counts: [false, true]（常见）
            elif isinstance(b.get("branch_counts"), list):
                counts = [ _int(x) for x in b["branch_counts"] ]
            elif isinstance(b.get("counts"), list):
                counts = [ _int(x) for x in b["counts"] ]
            if counts is None:
                continue
            for eidx, c in enumerate(counts):
                if c > 0:
                    keys.add((file_path, line, col, eidx))
            continue

        # 形态 2：列表，如 [line, col, c0, c1, ...]
        if isinstance(b, list) and len(b) >= 3:
            line = b[0]
            col  = b[1]
            # 取第 3 项及以后所有“可转 int 的数字”当作各 edge 的计数
            counts = []
            for x in b[2:]:
                try:
                    counts.append(int(x))
                except Exception:
                    # 碰到非数字就忽略
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
    - 若只有 segments，可退化为“点键”（end= None）
    """
    file_path = stable_file(f)
    keys = set()

    regions = f.get("regions")
    if isinstance(regions, list) and regions:
        for r in regions:
            if isinstance(r, dict):
                sl, sc = r.get("line_start"), r.get("column_start")
                el, ec = r.get("line_end"),   r.get("column_end")
                cnt    = _int(r.get("count", 0))
            elif isinstance(r, list) and len(r) >= 5:
                sl, sc, el, ec, cnt = r[:5]
                cnt = _int(cnt)
            else:
                continue
            if cnt > 0:
                keys.add((file_path, sl, sc, el, ec))
        return keys

    # 退化：用 segments 近似
    segments = f.get("segments")
    if isinstance(segments, list):
        for s in segments:
            # 常见形态：[line, col, count, hasCount, isRegionEntry, ...]
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

def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)

def main():
    if len(sys.argv) != 3:
        print("Usage: cov_overlap.py run1.json run2.json", file=sys.stderr)
        sys.exit(2)
    p1, p2 = sys.argv[1], sys.argv[2]

    B1 = load_branch_keys(p1)
    B2 = load_branch_keys(p2)
    R1 = load_region_keys(p1)
    R2 = load_region_keys(p2)

    print("== Branch overlap ==")
    print(f"Run1 edges: {len(B1)}  Run2 edges: {len(B2)}")
    print(f"Intersection: {len(B1 & B2)}  Union: {len(B1 | B2)}")
    print(f"Jaccard: {jaccard(B1, B2):.4f}")
    print(f"Only in Run1: {len(B1 - B2)}  Only in Run2: {len(B2 - B1)}")

    print("\n== Region overlap ==")
    print(f"Run1 regions: {len(R1)}  Run2 regions: {len(R2)}")
    print(f"Intersection: {len(R1 & R2)}  Union: {len(R1 | R2)}")
    print(f"Jaccard: {jaccard(R1, R2):.4f}")
    print(f"Only in Run1: {len(R1 - R2)}  Only in Run2: {len(R2 - R1)}")

if __name__ == "__main__":
    main()
