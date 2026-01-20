#!/usr/bin/env python3
# doc_rank_extractor.py
# Parse PyTorch doc txt files to extract shape tuples and infer rank candidates.
# Enhanced:
#   - Token whitelist with alias mapping
#   - Param-tuple blacklist (kernel/stride/pad dims)
#   - Support star/ellipsis: (N,C,*) / (N,C,...) / (*)
# Output:
#   multi_rank_index.json: api_name -> {fixed_ranks, rank_min, rank_max, rank_any}
# Backward compatibility: you can still read fixed_ranks as ranks list.

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple, Any


# -------------------------
# Regexes
# -------------------------

# Match parentheses content (allow newline inside, bounded)
PAREN_RE = re.compile(r"\((.{1,160}?)\)", re.DOTALL)

# Reject formula-like / code-like parentheses contents
BAD_CHARS_RE = re.compile(r"[=\+\-\/\[\]\{\}<>\^]|torch\.|nn\.|::|->|\\|\"|\'")

# Comma indicates tuple-like
HAS_COMMA_RE = re.compile(r",")

# Shape section hints (CN/EN)
SHAPE_SECTION_HINT_RE = re.compile(
    r"(形状|输入|输出|Input|Output|Shape|Shapes)", re.IGNORECASE
)

# Token whitelist:
# - Canonical dims: N,C,H,W,D,L,T,K, etc. and variants like C_in, H_k
# - Special: "*" or "..."
CANON_TOKEN_RE = re.compile(r"^[A-Z][A-Z0-9_]*$|^\*$|^\.\.\.$")

# Hard blacklist tokens that strongly indicate NOT input shape but param tuples
# (kernel/stride/pad/dilation dims)
PARAM_TUPLE_TOKENS = {
    "kH", "kW", "kD",
    "sH", "sW", "sD",
    "dH", "dW", "dD",
    "pH", "pW", "pD",
    "oH", "oW", "oD",
    "iH", "iW", "iD",  # NOTE: iH/iW can appear in input shape (rare); we handle via alias mapping below
}

# Reject obvious type-ish words (your dropout2d bool/optional noise)
TYPE_NOISE_TOKENS = {
    "bool", "optional", "int", "float", "str", "tensor", "none", "true", "false",
    "Optional", "Tensor", "Bool", "Int", "Float", "String"
}

# Alias mapping from doc words to canonical dim symbols
# Only include a small, safe set (avoid overfitting).
TOKEN_ALIAS = {
    "minibatch": "N",
    "batch": "N",
    "batch_size": "N",
    "in_channels": "C",
    "channels": "C",
    "channel": "C",
    "out_channels": "C",  # sometimes docs say out_channels but it's still a channel dim
    # Some docs use iH/iW for input height/width
    "iH": "H",
    "iW": "W",
    "iD": "D",
}


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
    """
    Return canonical token if it passes whitelist/alias rules,
    else None (meaning reject).
    """
    if not tok:
        return None

    # Normalize common punctuation/spaces
    tok = tok.strip()

    # Drop pure numbers
    if tok.isdigit():
        return None

    # Drop obvious noise type words
    if tok.lower() in {x.lower() for x in TYPE_NOISE_TOKENS}:
        return None

    # Alias mapping (case-sensitive for iH/iW)
    if tok in TOKEN_ALIAS:
        return TOKEN_ALIAS[tok]

    # Some docs may write 'N,C,H,W' without spaces; already split.
    # Accept canonical token patterns
    if CANON_TOKEN_RE.match(tok):
        # For "*" / "..." keep as is
        return tok

    # Otherwise reject
    return None


def _classify_tuple(tokens: List[str]) -> str:
    """
    Classify tuple into:
      - "shape_fixed": likely input shape with fixed rank (no * or ...)
      - "shape_range": contains * or ... (rank range / min-rank)
      - "param_tuple": kernel/stride/pad tuple (do not count as input ranks)
      - "reject": not a meaningful tuple
    """
    if not tokens:
        return "reject"

    # If any raw token is a param-tuple token (kH,kW,sH...), treat as param tuple
    # BUT allow iH/iW if mapped to H/W already. So check BEFORE canonicalization?
    # Here tokens are canonicalized already; so we check on original would be better.
    # We'll do a conservative check: if tuple contains k*/s*/p*/d*/o* patterns -> param_tuple
    for t in tokens:
        if t in ("kH", "kW", "kD", "sH", "sW", "sD", "pH", "pW", "pD", "dH", "dW", "dD", "oH", "oW", "oD"):
            return "param_tuple"

    # If any token is '*' or '...' => range-like
    if any(t in ("*", "...") for t in tokens):
        # "(*)" or "(...,)" etc.
        return "shape_range"

    # Fixed rank: must contain at least one typical batch/channel dim to avoid counting (H,W) etc.
    # This avoids max_pool2d picking up (kH,kW) etc (already filtered), and other random tuples.
    if not any(t in ("N", "C") or t.startswith("C_") for t in tokens):
        # Without N/C, it's often not an input shape tuple for NN ops.
        # Keep it reject to be safe.
        return "reject"

    return "shape_fixed"


def extract_shape_info(text: str, focus_sections: bool = True) -> Dict[str, Any]:
    """
    Extract:
      - tuples_fixed: list of canonical token lists without * / ...
      - tuples_range: list of canonical token lists containing * / ...
      - fixed_ranks: set[int]
      - rank_min/rank_max/rank_any: inferred from range tuples
    """
    # Focus on shape-related sections to reduce noise
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

        # Fast reject: formula/code-ish
        if BAD_CHARS_RE.search(inside):
            continue

        # Accept either:
        #   - comma tuples "(N, C, H, W)"
        #   - star-only "(*)"
        #   - ellipsis-only "(...)"
        # so: allow no comma only for "*" / "..."
        has_comma = bool(HAS_COMMA_RE.search(inside))
        if not has_comma:
            inside_stripped = inside.strip()
            if inside_stripped not in ("*", "..."):
                continue
            # Make it a one-token tuple
            parts = [inside_stripped]
        else:
            # Split tokens
            parts = _split_tokens(inside)
            if not (1 <= len(parts) <= 10):
                continue

        # Canonicalize tokens with whitelist/alias rules
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

        # Classify
        kind = _classify_tuple(canon)
        if kind == "param_tuple" or kind == "reject":
            continue

        if kind == "shape_fixed":
            tuples_fixed.append(canon)
            r = len(canon)
            if 1 <= r <= 12:
                fixed_ranks.add(r)

        elif kind == "shape_range":
            tuples_range.append(canon)

            # Infer range info
            # "(*)" => any rank (at least 1)
            if canon == ["*"]:
                rank_any = True
                continue

            # Patterns like (N, C, *) or (N, C, ...)
            # We treat '*' or '...' as "remaining dims arbitrary"
            # so min rank is number of tokens before '*'/'...'
            if "*" in canon or "..." in canon:
                idx = canon.index("*") if "*" in canon else canon.index("...")
                # If wildcard at position idx, minimum rank is idx (tokens before wildcard) OR idx+?:
                # Example: (N, C, *) => at least 2 dims, but practically means >=2
                # We'll set min_rank = idx  (count of concrete dims)
                min_r = idx
                if min_r <= 0:
                    rank_any = True
                else:
                    rank_min = min_r if rank_min is None else min(rank_min, min_r)
                # max rank unknown in docs, keep None

    # De-dup tuples for nicer debug
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


def merge_api_record(index: Dict[str, Any], api_name: str, rec: Dict[str, Any]) -> None:
    """
    Merge records:
      fixed_ranks: union
      rank_any: OR
      rank_min: min
      rank_max: max (if ever used)
    """
    old = index.get(api_name)
    if not isinstance(old, dict):
        old = {"fixed_ranks": [], "rank_any": False, "rank_min": None, "rank_max": None}

    # fixed_ranks
    s = set(int(x) for x in (old.get("fixed_ranks") or []))
    s.update(int(x) for x in (rec.get("fixed_ranks") or []))
    old["fixed_ranks"] = sorted(s)

    # rank_any
    old["rank_any"] = bool(old.get("rank_any", False) or rec.get("rank_any", False))

    # rank_min
    o_min = old.get("rank_min", None)
    r_min = rec.get("rank_min", None)
    if r_min is not None:
        old["rank_min"] = r_min if o_min is None else min(int(o_min), int(r_min))

    # rank_max (unused, kept for future)
    o_max = old.get("rank_max", None)
    r_max = rec.get("rank_max", None)
    if r_max is not None:
        old["rank_max"] = r_max if o_max is None else max(int(o_max), int(r_max))

    index[api_name] = old


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api_name", help="single api_name, used with --doc_txt/--doc_dir")
    ap.add_argument("--doc_txt", action="append", default=[], help="doc txt file path (repeatable)")
    ap.add_argument("--doc_dir", action="append", default=[], help="doc txt directory path (repeatable, load *.txt)")
    ap.add_argument("--mapping_json", help="batch mode mapping: {api_name: [doc_paths...]}")
    ap.add_argument("--out_json", default="multi_rank_index.json", help="output json path")
    ap.add_argument("--merge", action="store_true", help="merge into existing out_json if exists")
    ap.add_argument("--no_focus", action="store_true", help="disable focusing shape sections")
    ap.add_argument("--debug", action="store_true", help="print extracted tuples and ranks")
    args = ap.parse_args()

    out_path = Path(args.out_json).resolve()

    index: Dict[str, Any] = {}
    if args.merge and out_path.exists():
        try:
            index = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            index = {}

    focus_sections = (not args.no_focus)

    # Mode 1: mapping_json (batch)
    if args.mapping_json:
        mapping = load_mapping_json(Path(args.mapping_json).resolve())
        for api, doc_paths in mapping.items():
            paths = expand_paths(doc_paths)
            merged_text = ""
            for p in paths:
                merged_text += "\n" + read_text_file(p)

            rec = extract_shape_info(merged_text, focus_sections=focus_sections)
            merge_api_record(index, api, rec)

            if args.debug:
                print(f"[{api}] fixed_ranks={rec['fixed_ranks']} rank_any={rec['rank_any']} rank_min={rec['rank_min']}")
                for t in rec["tuples_fixed"][:30]:
                    print("  fixed:", t)
                for t in rec["tuples_range"][:30]:
                    print("  range:", t)

    else:
        # Mode 2: single api_name + docs
        if not args.api_name:
            raise SystemExit("Need --api_name in non-mapping mode.")
        inputs = []
        inputs.extend(args.doc_txt or [])
        inputs.extend(args.doc_dir or [])
        paths = expand_paths(inputs)
        if not paths:
            raise SystemExit("No doc files found. Provide --doc_txt or --doc_dir.")

        merged_text = ""
        for p in paths:
            merged_text += "\n" + read_text_file(p)

        rec = extract_shape_info(merged_text, focus_sections=focus_sections)
        merge_api_record(index, args.api_name, rec)

        if args.debug:
            print(f"[{args.api_name}] fixed_ranks={rec['fixed_ranks']} rank_any={rec['rank_any']} rank_min={rec['rank_min']}")
            for t in rec["tuples_fixed"][:50]:
                print("  fixed:", t)
            for t in rec["tuples_range"][:50]:
                print("  range:", t)

    out_path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[+] wrote {out_path} with {len(index)} api entries")


if __name__ == "__main__":
    main()
