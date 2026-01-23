#!/usr/bin/env python3
# doc_rank_extractor.py
# Stage-A version: per-API rank file output (one file per api).
#
# Input:
#   - single: --api_name + --doc_txt/--doc_dir
#   - batch:  --mapping_json {api_name: [doc_paths...]}
#
# Output:
#   out_dir/<safe_name(api_name)>.rank.json
#   schema:
#     {
#       "api_name": "...",
#       "rank_candidates": [...],   # from extracted fixed_ranks
#       "rank_any": bool,
#       "rank_min": int|null,
#       "rank_max": int|null,
#       "marker": "__RANK_FROM_DOC__",
#       "evidence": { "tuples_fixed": [...], "tuples_range": [...] }   # optional
#     }

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


# -------------------------
# Regexes
# -------------------------

PAREN_RE = re.compile(r"\((.{1,160}?)\)", re.DOTALL)
BAD_CHARS_RE = re.compile(r"[=\+\-\/\[\]\{\}<>\^]|torch\.|nn\.|::|->|\\|\"|\'")
HAS_COMMA_RE = re.compile(r",")
SHAPE_SECTION_HINT_RE = re.compile(r"(形状|输入|输出|Input|Output|Shape|Shapes)", re.IGNORECASE)
CANON_TOKEN_RE = re.compile(r"^[A-Z][A-Z0-9_]*$|^\*$|^\.\.\.$")

PARAM_TUPLE_TOKENS = {
    "kH", "kW", "kD",
    "sH", "sW", "sD",
    "dH", "dW", "dD",
    "pH", "pW", "pD",
    "oH", "oW", "oD",
    "iH", "iW", "iD",
}

TYPE_NOISE_TOKENS = {
    "bool", "optional", "int", "float", "str", "tensor", "none", "true", "false",
    "Optional", "Tensor", "Bool", "Int", "Float", "String"
}

TOKEN_ALIAS = {
    "minibatch": "N",
    "batch": "N",
    "batch_size": "N",
    "in_channels": "C",
    "channels": "C",
    "channel": "C",
    "out_channels": "C",
    "iH": "H",
    "iW": "W",
    "iD": "D",
}

RANK_MARKER = "__RANK_FROM_DOC__"


# -----------------------
# helpers
# -----------------------

def safe_name(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return "unknown_api"
    return (
        s.replace("::", "_")
         .replace(".", "_")
         .replace("/", "_")
         .replace(" ", "_")
         .replace("-", "_")
    )


def _normalize_inside(s: str) -> str:
    s = s.replace("，", ",")
    s = s.replace("\u3000", " ")
    s = re.sub(r"\s+", " ", s.strip())
    return s


def _split_tokens(inside: str) -> List[str]:
    inside = _normalize_inside(inside)
    parts = [p.strip() for p in inside.split(",")]
    return [p for p in parts if p]


def _canonicalize_token(tok: str) -> Optional[str]:
    if not tok:
        return None
    tok = tok.strip()
    if tok.isdigit():
        return None
    if tok.lower() in {x.lower() for x in TYPE_NOISE_TOKENS}:
        return None
    if tok in TOKEN_ALIAS:
        return TOKEN_ALIAS[tok]
    if CANON_TOKEN_RE.match(tok):
        return tok
    return None


def _classify_tuple(tokens: List[str]) -> str:
    if not tokens:
        return "reject"

    # param tuple reject (kernel/stride/pad/dilation...)
    for t in tokens:
        if t in ("kH", "kW", "kD", "sH", "sW", "sD", "pH", "pW", "pD", "dH", "dW", "dD", "oH", "oW", "oD"):
            return "param_tuple"

    # wildcard => range-like
    if any(t in ("*", "...") for t in tokens):
        return "shape_range"

    # conservative: require N/C for "shape_fixed"
    if not any(t in ("N", "C") or t.startswith("C_") for t in tokens):
        return "reject"

    return "shape_fixed"


def extract_shape_info(text: str, focus_sections: bool = True) -> Dict[str, Any]:
    if focus_sections:
        lines = text.splitlines()
        keep: List[str] = []
        window = 0
        for ln in lines:
            if SHAPE_SECTION_HINT_RE.search(ln):
                window = 80
            if window > 0:
                keep.append(ln)
                window -= 1
        if keep:
            text = "\n".join(keep)

    tuples_fixed: List[List[str]] = []
    tuples_range: List[List[str]] = []

    fixed_ranks: Set[int] = set()
    rank_any = False
    rank_min: Optional[int] = None
    rank_max: Optional[int] = None

    for m in PAREN_RE.finditer(text):
        inside_raw = m.group(1)
        inside = _normalize_inside(inside_raw)

        if BAD_CHARS_RE.search(inside):
            continue

        has_comma = bool(HAS_COMMA_RE.search(inside))
        if not has_comma:
            inside_stripped = inside.strip()
            if inside_stripped not in ("*", "..."):
                continue
            parts = [inside_stripped]
        else:
            parts = _split_tokens(inside)
            if not (1 <= len(parts) <= 10):
                continue

        canon: List[str] = []
        bad = False
        for tok in parts:
            ct = _canonicalize_token(tok)
            if ct is None:
                bad = True
                break
            canon.append(ct)
        if bad:
            continue

        kind = _classify_tuple(canon)
        if kind in ("param_tuple", "reject"):
            continue

        if kind == "shape_fixed":
            tuples_fixed.append(canon)
            r = len(canon)
            if 1 <= r <= 12:
                fixed_ranks.add(r)

        elif kind == "shape_range":
            tuples_range.append(canon)

            if canon == ["*"]:
                rank_any = True
                continue

            if "*" in canon or "..." in canon:
                idx = canon.index("*") if "*" in canon else canon.index("...")
                min_r = idx
                if min_r <= 0:
                    rank_any = True
                else:
                    rank_min = min_r if rank_min is None else min(rank_min, min_r)

    def _dedup(lsts: List[List[str]]) -> List[List[str]]:
        seen = set()
        out = []
        for x in lsts:
            k = tuple(x)
            if k not in seen:
                seen.add(k)
                out.append(x)
        return out

    tuples_fixed = _dedup(tuples_fixed)
    tuples_range = _dedup(tuples_range)

    return {
        "tuples_fixed": tuples_fixed,
        "tuples_range": tuples_range,
        "fixed_ranks": sorted(fixed_ranks),
        "rank_any": bool(rank_any),
        "rank_min": rank_min,
        "rank_max": rank_max,
    }


# -------------------------
# IO helpers
# -------------------------

def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return path.read_text(encoding="latin-1", errors="ignore")


def expand_paths(inputs: List[str]) -> List[Path]:
    out: List[Path] = []
    for s in inputs:
        p = Path(s).expanduser().resolve()
        if p.is_dir():
            out.extend(sorted(p.glob("*.txt")))
        elif p.is_file():
            out.append(p)
    return out


def load_mapping_json(path: Path) -> Dict[str, List[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("mapping json must be an object: {api_name: [doc_paths...]}")
    out: Dict[str, List[str]] = {}
    for k, v in data.items():
        if isinstance(v, str):
            out[str(k)] = [v]
        elif isinstance(v, list):
            out[str(k)] = [str(x) for x in v]
        else:
            raise ValueError(f"mapping value for {k} must be string or list")
    return out


def build_rank_record(api_name: str, rec: Dict[str, Any], with_evidence: bool) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "api_name": api_name,
        "rank_candidates": rec.get("fixed_ranks", []) or [],
        "rank_any": bool(rec.get("rank_any", False)),
        "rank_min": rec.get("rank_min", None),
        "rank_max": rec.get("rank_max", None),
        "marker": RANK_MARKER,
    }
    if with_evidence:
        record["evidence"] = {
            "tuples_fixed": rec.get("tuples_fixed", []) or [],
            "tuples_range": rec.get("tuples_range", []) or [],
        }
    return record


def merge_rank_record(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge per-api rank records:
      - rank_candidates: union
      - rank_any: OR
      - rank_min: min
      - rank_max: max
      - evidence: concat+dedup (optional)
    """
    out = dict(old)

    # union ranks
    s = set(int(x) for x in (out.get("rank_candidates") or []))
    s.update(int(x) for x in (new.get("rank_candidates") or []))
    out["rank_candidates"] = sorted(s)

    # rank_any
    out["rank_any"] = bool(out.get("rank_any", False) or new.get("rank_any", False))

    # min/max
    o_min = out.get("rank_min", None)
    n_min = new.get("rank_min", None)
    if n_min is not None:
        out["rank_min"] = n_min if o_min is None else min(int(o_min), int(n_min))

    o_max = out.get("rank_max", None)
    n_max = new.get("rank_max", None)
    if n_max is not None:
        out["rank_max"] = n_max if o_max is None else max(int(o_max), int(n_max))

    # marker/api_name keep
    out["api_name"] = out.get("api_name") or new.get("api_name")
    out["marker"] = out.get("marker") or new.get("marker") or RANK_MARKER

    # evidence merge if both present
    if isinstance(out.get("evidence"), dict) or isinstance(new.get("evidence"), dict):
        e_old = out.get("evidence") or {"tuples_fixed": [], "tuples_range": []}
        e_new = new.get("evidence") or {"tuples_fixed": [], "tuples_range": []}

        def _dedup_list_of_lists(x: List[List[Any]]) -> List[List[Any]]:
            seen = set()
            res = []
            for item in x:
                k = tuple(item)
                if k not in seen:
                    seen.add(k)
                    res.append(item)
            return res

        tf = (e_old.get("tuples_fixed") or []) + (e_new.get("tuples_fixed") or [])
        tr = (e_old.get("tuples_range") or []) + (e_new.get("tuples_range") or [])
        out["evidence"] = {
            "tuples_fixed": _dedup_list_of_lists(tf)[:200],
            "tuples_range": _dedup_list_of_lists(tr)[:200],
        }

    return out


def write_rank_file(out_dir: Path, api_name: str, record: Dict[str, Any], merge: bool) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"{safe_name(api_name)}.rank.json"
    if merge and fp.exists():
        try:
            old = json.loads(fp.read_text(encoding="utf-8"))
            if isinstance(old, dict):
                record = merge_rank_record(old, record)
        except Exception:
            # if broken file, overwrite
            pass
    fp.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api_name", help="single api_name, used with --doc_txt/--doc_dir")
    ap.add_argument("--doc_txt", action="append", default=[], help="doc txt file path (repeatable)")
    ap.add_argument("--doc_dir", action="append", default=[], help="doc txt directory path (repeatable, load *.txt)")
    ap.add_argument("--mapping_json", help="batch mode mapping: {api_name: [doc_paths...]}")
    ap.add_argument("--out_dir", required=True, help="output directory for per-api rank json files")
    ap.add_argument("--merge", action="store_true", help="merge into existing per-api rank file if exists")
    ap.add_argument("--no_focus", action="store_true", help="disable focusing shape sections")
    ap.add_argument("--with_evidence", action="store_true", help="include tuples_fixed/tuples_range as evidence in rank file")
    ap.add_argument("--debug", action="store_true", help="print extracted tuples and ranks")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    focus_sections = (not args.no_focus)

    if args.mapping_json:
        mapping = load_mapping_json(Path(args.mapping_json).resolve())
        for api, doc_paths in mapping.items():
            paths = expand_paths(doc_paths)
            merged_text = ""
            for p in paths:
                merged_text += "\n" + read_text_file(p)

            rec = extract_shape_info(merged_text, focus_sections=focus_sections)
            record = build_rank_record(api, rec, with_evidence=bool(args.with_evidence))
            fp = write_rank_file(out_dir, api, record, merge=bool(args.merge))

            if args.debug:
                print(f"[{api}] ranks={record['rank_candidates']} any={record['rank_any']} min={record['rank_min']} -> {fp}")
                if args.with_evidence:
                    for t in (record.get("evidence", {}).get("tuples_fixed") or [])[:30]:
                        print("  fixed:", t)
                    for t in (record.get("evidence", {}).get("tuples_range") or [])[:30]:
                        print("  range:", t)
    else:
        if not args.api_name:
            raise SystemExit("Need --api_name in non-mapping mode.")

        inputs: List[str] = []
        inputs.extend(args.doc_txt or [])
        inputs.extend(args.doc_dir or [])
        paths = expand_paths(inputs)
        if not paths:
            raise SystemExit("No doc files found. Provide --doc_txt or --doc_dir.")

        merged_text = ""
        for p in paths:
            merged_text += "\n" + read_text_file(p)

        rec = extract_shape_info(merged_text, focus_sections=focus_sections)
        record = build_rank_record(args.api_name, rec, with_evidence=bool(args.with_evidence))
        fp = write_rank_file(out_dir, args.api_name, record, merge=bool(args.merge))

        if args.debug:
            print(f"[{args.api_name}] ranks={record['rank_candidates']} any={record['rank_any']} min={record['rank_min']} -> {fp}")
            if args.with_evidence:
                for t in (record.get("evidence", {}).get("tuples_fixed") or [])[:50]:
                    print("  fixed:", t)
                for t in (record.get("evidence", {}).get("tuples_range") or [])[:50]:
                    print("  range:", t)

    print(f"[+] wrote rank files into {out_dir}")


if __name__ == "__main__":
    main()
