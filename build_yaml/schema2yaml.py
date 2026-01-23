#!/usr/bin/env python3
# schema_json_to_yaml_skeleton.py
#
# Stage-B version:
#   - Read schema json -> yaml skeleton
#   - Read per-API rank hint json produced by Stage-A (doc_rank_extractor.py)
#   - DO NOT inject rank into params
#   - Always emit a top-level `rank_hints` block (structure stable)
#
# rank file path:
#   <rank_index_dir>/<safe_name(api_name)>.rank.json
#
# rank file schema (Stage-A):
#   {
#     "api_name": "...",
#     "rank_candidates": [2,3,4,5],
#     "rank_any": false,
#     "rank_min": null,
#     "rank_max": null,
#     "marker": "__RANK_FROM_DOC__",
#     ...
#   }

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# -----------------------
# Global defaults / knobs
# -----------------------
DEFAULT_TENSOR_DTYPES = ["float32", "float64"]
DEFAULT_TENSOR_DTYPES_OPT = ["float32", "float64"]

DEFAULT_INT_RANGE = [-1, 8]
DEFAULT_DIM_RANGE = [-4, 4]
DEFAULT_FLOAT_RANGE = [-1.0, 1.0]
DEFAULT_EPS_RANGE = [1e-12, 1e-1]
DEFAULT_PROB_RANGE = [0.0, 1.0]

DEFAULT_INT_LIST_LEN_RANGE = [1, 3]
DEFAULT_INT_LIST_RANGE = [0, 4]

# Stage-B markers
RANK_MISS_MARKER = "__RANK_TODO__"
RANK_FROM_DOC_MARKER = "__RANK_FROM_DOC__"


# -----------------------
# helpers
# -----------------------
def safe_name(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return "default"
    return (
        s.replace("::", "_")
        .replace(".", "_")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
    )


def parse_default(default_repr: Optional[str]) -> Any:
    if default_repr is None:
        return None
    try:
        return ast.literal_eval(default_repr)
    except Exception:
        return default_repr


def _name_lower(name: str) -> str:
    return (name or "").lower()


def _tensor_dtype_choices_for_param_name(arg_name: str, optional: bool) -> List[str]:
    n = _name_lower(arg_name)
    if any(tok in n for tok in ("indices", "index")) or (n.startswith("ind") or "_ind" in n):
        return ["int64"]
    if "mask" in n:
        return ["bool"]
    return DEFAULT_TENSOR_DTYPES_OPT if optional else DEFAULT_TENSOR_DTYPES


def _int_range_for_param_name(arg_name: str) -> List[int]:
    n = _name_lower(arg_name)
    if "dim" in n or "axis" in n:
        return DEFAULT_DIM_RANGE
    return DEFAULT_INT_RANGE


def _float_range_for_param_name(arg_name: str) -> List[float]:
    n = _name_lower(arg_name)
    if "eps" in n:
        return DEFAULT_EPS_RANGE
    if n == "p" or n.endswith("_p") or "prob" in n:
        return DEFAULT_PROB_RANGE
    return DEFAULT_FLOAT_RANGE


def infer_kind(arg_name: str, type_str: str, default_repr: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    t = (type_str or "").strip()
    default_val = parse_default(default_repr)
    n = _name_lower(arg_name)

    is_optional_tensor = ("Tensor" in t) and (("Optional[" in t) or t.endswith("?") or t.startswith("Tensor?"))
    is_tensor = ("Tensor" in t)

    if is_optional_tensor:
        spec = {
            "kind": "tensor_optional",
            "dtype_choices": _tensor_dtype_choices_for_param_name(arg_name, optional=True),
            "shape_spec": ["TODO_SHAPE"],
        }
        return "tensor_optional", spec

    if is_tensor:
        spec = {
            "kind": "tensor",
            "dtype_choices": _tensor_dtype_choices_for_param_name(arg_name, optional=False),
            "shape_spec": ["TODO_SHAPE"],
        }
        return "tensor", spec

    if t == "bool":
        return "bool", {"kind": "bool", "default": bool(default_val) if isinstance(default_val, bool) else None}

    if t in ("float", "double"):
        spec: Dict[str, Any] = {"kind": "float", "range": _float_range_for_param_name(arg_name)}
        if isinstance(default_val, (int, float)):
            spec["default"] = float(default_val)
        return "float", spec

    if t in ("int", "SymInt"):
        spec = {"kind": "int", "range": _int_range_for_param_name(arg_name)}
        if isinstance(default_val, int):
            spec["default"] = int(default_val)
            lo, hi = spec["range"]
            if default_val < lo:
                lo = int(default_val)
            if default_val > hi:
                hi = int(default_val)
            spec["range"] = [lo, hi]
        return "int", spec

    if t.endswith("[2]") and (t.startswith("SymInt") or t.startswith("int")):
        values: List[Any] = [1, 2, 3, [1, 1], [2, 2], [3, 3]]
        if isinstance(default_val, list) and len(default_val) == 2:
            if default_val not in values:
                values.insert(0, default_val)
        spec = {"kind": "int_or_tuple", "values": values}
        return "int_or_tuple", spec

    if t in ("List[int]", "int[]", "SymInt[]"):
        spec = {
            "kind": "int_list",
            "len_range": DEFAULT_INT_LIST_LEN_RANGE[:],
            "range": DEFAULT_INT_LIST_RANGE[:],
        }
        if isinstance(default_val, list):
            spec["default"] = default_val
            L = max(1, len(default_val))
            spec["len_range"] = [L, L]
        return "int_list", spec

    if t == "str":
        values: List[str] = []
        if isinstance(default_val, str):
            values.append(default_val)
        if n == "reduction":
            for x in ("none", "mean", "sum"):
                if x not in values:
                    values.append(x)
        if not values:
            values = [""]
        spec = {"kind": "enum", "values": values}
        return "enum", spec

    fallback_val = str(default_val) if default_val is not None else "TODO"
    return "enum", {"kind": "enum", "values": [fallback_val]}


def load_rank_hints(api_name: str, rank_index_dir: Optional[Path]) -> Dict[str, Any]:
    """
    Stage-B: load per-api rank file if exists, otherwise emit a stable placeholder.

    Output structure is stable and DOES NOT include evidence.
    """
    # default placeholder
    hints: Dict[str, Any] = {
        "marker": RANK_MISS_MARKER,
        "status": "missing",  # missing / unassigned
        "rank_candidates": [RANK_MISS_MARKER],
        "rank_any": None,
        "rank_min": None,
        "rank_max": None,
    }

    if not rank_index_dir:
        return hints

    fp = rank_index_dir / f"{safe_name(api_name)}.rank.json"
    if not fp.exists():
        return hints

    try:
        data = json.loads(fp.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return hints

    if not isinstance(data, dict):
        return hints

    # Stage-A schema uses rank_candidates
    ranks = data.get("rank_candidates")
    rank_any = data.get("rank_any", False)
    rank_min = data.get("rank_min", None)
    rank_max = data.get("rank_max", None)

    # normalize ranks
    norm_ranks: Optional[List[int]] = None
    if isinstance(ranks, list):
        tmp: List[int] = []
        for x in ranks:
            try:
                tmp.append(int(x))
            except Exception:
                continue
        # keep unique sorted
        norm_ranks = sorted(set(tmp))

    # marker from Stage-A file, but keep our stable style
    marker = data.get("marker") or RANK_FROM_DOC_MARKER

    hints = {
        "marker": marker,
        "status": "unassigned",
        "rank_candidates": norm_ranks if norm_ranks else [RANK_MISS_MARKER],
        "rank_any": bool(rank_any),
        "rank_min": rank_min,
        "rank_max": rank_max,
    }
    return hints


def build_yaml_for_overload(
    api_name: str,
    category: str,
    aten_name: str,
    overload_key: str,
    overload_schema: Dict[str, Any],
    rank_hints: Dict[str, Any],
) -> Dict[str, Any]:
    args = overload_schema.get("arguments", [])
    params: Dict[str, Any] = {}

    for a in args:
        name = a["name"]
        type_str = a["type"]
        default_repr = a.get("default")
        _, spec = infer_kind(name, type_str, default_repr)

        if a.get("has_default", False):
            spec["has_default"] = True
            spec["default_repr"] = default_repr

        params[name] = spec

    y = {
        "api_name": api_name,
        "category": category,
        "rank_hints": rank_hints,  # <-- Stage-B: API-level hint, no evidence, not in params
        "aten": {
            "aten_name": aten_name,
            "overload": overload_key if overload_key else "default",
            "schema_str": overload_schema.get("schema_str", ""),
        },
        "shape_vars": {},
        "params": params,
        "constraints": [],
    }
    return y


def convert_one_json(json_path: Path, out_dir: Path, rank_index_dir: Optional[Path]) -> List[Path]:
    data = json.loads(json_path.read_text(encoding="utf-8", errors="ignore"))
    api_name = data.get("api_name", "unknown.api")
    category = api_name.split(".")[-1]

    aten = data.get("aten") or {}
    aten_name = aten.get("aten_name", category)
    overloads = (aten.get("overloads") or {})

    # Stage-B: load rank hints from per-api file
    rank_hints = load_rank_hints(api_name, rank_index_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    produced: List[Path] = []

    for k, ov in overloads.items():
        overload_key = k if k is not None else ""
        y = build_yaml_for_overload(api_name, category, aten_name, overload_key, ov, rank_hints=rank_hints)

        fn = f"{safe_name(api_name)}__ov_{safe_name(overload_key)}.yaml"
        out_path = out_dir / fn
        out_path.write_text(
            yaml.safe_dump(y, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        produced.append(out_path)

    return produced


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema_json", required=True, help="single schema json or a directory of jsons")
    ap.add_argument("--out_dir", required=True, help="where to write yaml skeletons")
    # Stage-B: read per-api rank files output by Stage-A
    ap.add_argument("--rank_index_dir", default=None, help="directory containing <safe_name(api)>.rank.json (optional)")
    args = ap.parse_args()

    src = Path(args.schema_json).resolve()
    out_dir = Path(args.out_dir).resolve()
    rank_index_dir = Path(args.rank_index_dir).resolve() if args.rank_index_dir else None

    if src.is_dir():
        jsons = sorted(src.glob("*.json"))
        print(f"[+] found {len(jsons)} schema json files in {src}")
        for jp in jsons:
            outs = convert_one_json(jp, out_dir, rank_index_dir=rank_index_dir)
            for o in outs:
                print(f"[+] wrote {o}")
    else:
        outs = convert_one_json(src, out_dir, rank_index_dir=rank_index_dir)
        for o in outs:
            print(f"[+] wrote {o}")


if __name__ == "__main__":
    main()
