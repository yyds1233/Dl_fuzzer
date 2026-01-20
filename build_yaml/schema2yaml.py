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

DEFAULT_INT_RANGE = [-1, 8]
DEFAULT_DIM_RANGE = [-4, 4]
DEFAULT_FLOAT_RANGE = [-1.0, 1.0]
DEFAULT_EPS_RANGE = [1e-12, 1e-1]
DEFAULT_PROB_RANGE = [0.0, 1.0]

DEFAULT_INT_LIST_LEN_RANGE = [1, 3]
DEFAULT_INT_LIST_RANGE = [0, 4]

RANK_MISS_MARKER = "__RANK_TODO__"



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
    t = type_str.strip()
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
        spec: Dict[str, Any] = {
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


def _apply_rank_info_to_tensor_param(
    spec: Dict[str, Any],
    param_name: str,
    rank_info: Optional[Dict[str, Any]],
) -> None:
    """
    永远输出 rank 结构：
      - 命中 rank_info：写真实值
      - 未命中：写特定标记（结构不变）
    默认仅对 input/self 生效，避免误约束 weight/bias 等。
    """
    if spec.get("kind") not in ("tensor", "tensor_optional"):
        return

    # 只对 input/self 加（你要全加的话，删掉这一段 if）
    if param_name not in ("input", "self"):
        return

    if rank_info:
        fixed_ranks = rank_info.get("fixed_ranks")
        rank_any = bool(rank_info.get("rank_any", False))
        rank_min = rank_info.get("rank_min", None)
        rank_max = rank_info.get("rank_max", None)

        spec["rank"] = {
            "rank_any": rank_any,
            "fixed_ranks": [int(x) for x in fixed_ranks] if fixed_ranks is not None else None,
            "rank_min": rank_min,
            "rank_max": rank_max,
        }
    else:
        # 未命中：输出特定标记，但保持 rank 结构完全一致
        spec["rank"] = {
            "rank_any": None,
            "fixed_ranks": [RANK_MISS_MARKER],
            "rank_min": None,
            "rank_max": None,
        }


def build_yaml_for_overload(
    api_name: str,
    category: str,
    aten_name: str,
    overload_key: str,
    overload_schema: Dict[str, Any],
    rank_info: Optional[Dict[str, Any]] = None,
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

        # === 新增：注入 rank 信息 ===
        _apply_rank_info_to_tensor_param(spec, name, rank_info)

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


def convert_one_json(json_path: Path, out_dir: Path, rank_index: Optional[Dict[str, Any]] = None) -> List[Path]:
    data = json.loads(json_path.read_text(encoding="utf-8", errors="ignore"))
    api_name = data.get("api_name", "unknown.api")
    category = api_name.split(".")[-1]

    aten = data.get("aten") or {}
    aten_name = aten.get("aten_name", category)
    overloads = (aten.get("overloads") or {})

    # === 新增：取当前 api 的 rank_info ===
    rank_info = (rank_index or {}).get(api_name)

    out_dir.mkdir(parents=True, exist_ok=True)
    produced: List[Path] = []

    for k, ov in overloads.items():
        overload_key = k if k is not None else ""
        y = build_yaml_for_overload(api_name, category, aten_name, overload_key, ov, rank_info=rank_info)

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
    # === 新增：rank index 参数（可选） ===
    ap.add_argument("--rank_index_json", default=None, help="rank index json file (optional)")
    args = ap.parse_args()

    src = Path(args.schema_json).resolve()
    out_dir = Path(args.out_dir).resolve()

    # === 新增：加载 rank index ===
    rank_index: Optional[Dict[str, Any]] = None
    if args.rank_index_json:
        rp = Path(args.rank_index_json).resolve()
        rank_index = json.loads(rp.read_text(encoding="utf-8", errors="ignore"))

    if src.is_dir():
        jsons = sorted(src.glob("*.json"))
        print(f"[+] found {len(jsons)} schema json files in {src}")
        for jp in jsons:
            outs = convert_one_json(jp, out_dir, rank_index=rank_index)
            for o in outs:
                print(f"[+] wrote {o}")
    else:
        outs = convert_one_json(src, out_dir, rank_index=rank_index)
        for o in outs:
            print(f"[+] wrote {o}")


if __name__ == "__main__":
    main()
