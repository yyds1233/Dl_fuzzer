#!/usr/bin/env python3
# schema_json_to_yaml_skeleton.py
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

# For scalar numeric params (conservative, broadly valid)
DEFAULT_INT_RANGE = [-1, 8]
DEFAULT_DIM_RANGE = [-4, 4]  # dims often allow negative indexing
DEFAULT_FLOAT_RANGE = [-1.0, 1.0]
DEFAULT_EPS_RANGE = [1e-12, 1e-1]
DEFAULT_PROB_RANGE = [0.0, 1.0]

# For generic int lists
DEFAULT_INT_LIST_LEN_RANGE = [1, 3]
DEFAULT_INT_LIST_RANGE = [0, 4]


# -----------------------
# helpers
# -----------------------
def safe_name(s: str) -> str:
    s = s.strip()
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
    """
    default_repr 是 repr() 出来的字符串，比如:
      "1", "[1, 1]", "'valid'", "None"
    尽量用 literal_eval 还原成 Python 对象。
    """
    if default_repr is None:
        return None
    try:
        return ast.literal_eval(default_repr)
    except Exception:
        # 兜底：原样返回字符串
        return default_repr


def _name_lower(name: str) -> str:
    return (name or "").lower()


def _tensor_dtype_choices_for_param_name(arg_name: str, optional: bool) -> List[str]:
    """
    Very light heuristics:
      - *index/indices/ind* -> int64
      - *mask* -> bool
      - otherwise -> float32/float64 (conservative default)
    """
    n = _name_lower(arg_name)

    # Keep these heuristics intentionally narrow to avoid misclassification.
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
    # eps is usually > 0
    if "eps" in n:
        return DEFAULT_EPS_RANGE
    # probability-like params
    if n == "p" or n.endswith("_p") or "prob" in n:
        return DEFAULT_PROB_RANGE
    return DEFAULT_FLOAT_RANGE


def infer_kind(arg_name: str, type_str: str, default_repr: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    """
    把 ATen schema 的 type_str 映射到 YAML 的 kind + 附加字段（skeleton）。
    目标：保守 + 高可执行率（减少 early error），后续再用 normalization/probe/LLM 精化。
    """
    t = type_str.strip()
    default_val = parse_default(default_repr)
    n = _name_lower(arg_name)

    # Detect optional Tensor (keep conservative)
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

    # bool
    if t == "bool":
        return "bool", {"kind": "bool", "default": bool(default_val) if isinstance(default_val, bool) else None}

    # float / double
    if t in ("float", "double"):
        spec: Dict[str, Any] = {"kind": "float", "range": _float_range_for_param_name(arg_name)}
        if isinstance(default_val, (int, float)):
            spec["default"] = float(default_val)
        return "float", spec

    # int / SymInt
    if t in ("int", "SymInt"):
        spec = {"kind": "int", "range": _int_range_for_param_name(arg_name)}
        if isinstance(default_val, int):
            spec["default"] = int(default_val)
            # Keep it conservative: make sure default falls in range
            lo, hi = spec["range"]
            if default_val < lo:
                lo = int(default_val)
            if default_val > hi:
                hi = int(default_val)
            spec["range"] = [lo, hi]
        return "int", spec

    # Fixed-size int tuple/list: SymInt[2] / int[2]
    # (Note: in some torch versions this might show as List[int]; normalization stage can fix it using schema_str.)
    if t.endswith("[2]") and (t.startswith("SymInt") or t.startswith("int")):
        # Use int_or_tuple; avoid including 0 (0 stride/dilation is often invalid)
        values: List[Any] = [1, 2, 3, [1, 1], [2, 2], [3, 3]]
        if isinstance(default_val, list) and len(default_val) == 2:
            if default_val not in values:
                values.insert(0, default_val)
        spec = {"kind": "int_or_tuple", "values": values}
        return "int_or_tuple", spec

    # List[int] / int[] / SymInt[]  -> int_list
    if t in ("List[int]", "int[]", "SymInt[]"):
        spec: Dict[str, Any] = {
            "kind": "int_list",
            "len_range": DEFAULT_INT_LIST_LEN_RANGE[:],
            "range": DEFAULT_INT_LIST_RANGE[:],
        }
        if isinstance(default_val, list):
            # If default exists, fix length to default length (more executable)
            spec["default"] = default_val
            L = max(1, len(default_val))
            spec["len_range"] = [L, L]
        return "int_list", spec

    # str -> enum
    if t == "str":
        values: List[str] = []
        if isinstance(default_val, str):
            values.append(default_val)

        # tiny, safe enhancement for common patterns
        if n == "reduction":
            for x in ("none", "mean", "sum"):
                if x not in values:
                    values.append(x)

        if not values:
            # keep minimal; empty string might still be invalid for many ops, but it's only a skeleton
            values = [""]

        spec = {"kind": "enum", "values": values}
        return "enum", spec

    # Fallback: enum placeholder (keep minimal)
    fallback_val = str(default_val) if default_val is not None else "TODO"
    return "enum", {"kind": "enum", "values": [fallback_val]}


def build_yaml_for_overload(
    api_name: str,
    category: str,
    aten_name: str,
    overload_key: str,
    overload_schema: Dict[str, Any],
) -> Dict[str, Any]:
    args = overload_schema.get("arguments", [])
    params: Dict[str, Any] = {}

    for a in args:
        name = a["name"]
        type_str = a["type"]
        default_repr = a.get("default")
        _, spec = infer_kind(name, type_str, default_repr)

        # 把 schema 默认值也保留一份，方便你后处理
        if a.get("has_default", False):
            spec["has_default"] = True
            spec["default_repr"] = default_repr

        params[name] = spec

    y = {
        "api_name": api_name,
        "category": category,
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


def convert_one_json(json_path: Path, out_dir: Path) -> List[Path]:
    data = json.loads(json_path.read_text(encoding="utf-8", errors="ignore"))
    api_name = data.get("api_name", "unknown.api")
    category = api_name.split(".")[-1]

    aten = data.get("aten") or {}
    aten_name = aten.get("aten_name", category)
    overloads = (aten.get("overloads") or {})

    out_dir.mkdir(parents=True, exist_ok=True)
    produced: List[Path] = []

    for k, ov in overloads.items():
        overload_key = k if k is not None else ""
        y = build_yaml_for_overload(api_name, category, aten_name, overload_key, ov)

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
    args = ap.parse_args()

    src = Path(args.schema_json).resolve()
    out_dir = Path(args.out_dir).resolve()

    if src.is_dir():
        jsons = sorted(src.glob("*.json"))
        print(f"[+] found {len(jsons)} schema json files in {src}")
        for jp in jsons:
            outs = convert_one_json(jp, out_dir)
            for o in outs:
                print(f"[+] wrote {o}")
    else:
        outs = convert_one_json(src, out_dir)
        for o in outs:
            print(f"[+] wrote {o}")


if __name__ == "__main__":
    main()
