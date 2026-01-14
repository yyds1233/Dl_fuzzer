#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

DEFAULT_THREADS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "TORCH_NUM_THREADS": "1",
}

def load_top_results(p: Path) -> List[Dict[str, Any]]:
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{p} is not a list json")
    # 有的文件可能不是严格按 score 排序，保险起见再排一次
    data.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return data

def merge_env(profile_env: Dict[str, Any], default_threads: Dict[str, str]) -> Dict[str, Any]:
    out = dict(profile_env or {})
    # 补默认线程设置（不覆盖已有值）
    for k, v in default_threads.items():
        out.setdefault(k, v)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output harnesses.json path")
    ap.add_argument("--topk", type=int, default=10, help="top N profiles per harness")
    ap.add_argument("--default_threads", action="store_true", help="append OMP/MKL/TORCH default threads")
    # 一个 harness 配置： --harness <id> <harness_path> <round1_top_results.json>
    ap.add_argument("--harness", nargs=3, action="append", metavar=("ID", "HARNESS_PY", "TOP_JSON"),
                    required=True, help="repeatable: (harness_id, harness_path, top_json)")
    args = ap.parse_args()

    out_list: List[Dict[str, Any]] = []

    for hid, harness_py, top_json in args.harness:
        harness_path = str(Path(harness_py).resolve())
        top_path = Path(top_json).resolve()

        rows = load_top_results(top_path)
        rows = rows[: max(1, int(args.topk))]

        profiles = []
        for r in rows:
            pid = r.get("profile_id")
            prof = r.get("profile") or {}
            if not pid:
                continue
            if args.default_threads:
                prof = merge_env(prof, DEFAULT_THREADS)
            profiles.append({"profile_id": pid, "profile": prof})

        out_list.append({
            "harness_id": hid,
            "harness_path": harness_path,
            "profiles": profiles,
        })

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_list, indent=2), encoding="utf-8")
    print(f"[+] wrote {out_path} with {len(out_list)} harnesses")

if __name__ == "__main__":
    main()
