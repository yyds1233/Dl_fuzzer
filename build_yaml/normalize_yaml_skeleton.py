#!/usr/bin/env python3
# normalize_yaml_skeleton.py
#
# Stage-B->Stage-B' normalization:
#   - Keep YAML structure stable (especially rank_hints)
#   - Normalize params according to schema_str truth:
#       * has_default truth
#       * Optional/bool/dtype/device/layout kinds
#       * SymInt[k]/int[k] upgrading: int_list -> int_or_tuple (k=2/3 common)
#   - Apply universal, low-risk range fixes for common knobs
#   - Add generator version stamp for reproducibility/debugging
#
import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


# -----------------------
# Global markers
# -----------------------
RANK_MISS_MARKER = "__RANK_TODO__"
ENUM_TODO_MARKER = "__ENUM_TODO__"

GENERATOR_BLOCK = {
    "stage": "B-normalize",
    "version": "2026-01-21-v1",
}


# -----------------------
# schema_str parsing
# -----------------------
def _extract_args_section(schema_str: str) -> str:
    if not schema_str:
        return ""
    l = schema_str.find("(")
    if l < 0:
        return ""
    depth = 0
    for i in range(l, len(schema_str)):
        c = schema_str[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return schema_str[l + 1 : i]
    return ""


def _split_top_level_commas(s: str) -> List[str]:
    out: List[str] = []
    cur: List[str] = []
    depth_paren = 0
    depth_brack = 0
    for ch in s:
        if ch == "(":
            depth_paren += 1
        elif ch == ")":
            depth_paren = max(0, depth_paren - 1)
        elif ch == "[":
            depth_brack += 1
        elif ch == "]":
            depth_brack = max(0, depth_brack - 1)

        if ch == "," and depth_paren == 0 and depth_brack == 0:
            part = "".join(cur).strip()
            if part:
                out.append(part)
            cur = []
        else:
            cur.append(ch)

    tail = "".join(cur).strip()
    if tail:
        out.append(tail)
    return out


def _parse_one_arg(part: str) -> Optional[Tuple[str, str, bool, Optional[int], Optional[str]]]:
    p = part.strip()
    if not p:
        return None
    if p == "*":
        return None

    eq_pos = p.find("=")
    if eq_pos >= 0:
        left = p[:eq_pos].strip()
        default_str = p[eq_pos + 1 :].strip()
        has_default = True
    else:
        left = p
        default_str = None
        has_default = False

    m = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*$", left)
    if not m:
        return None
    name = m.group(1)
    type_str = left[: m.start(1)].strip()

    fixed_k = None
    m2 = re.search(r"\b(?:SymInt|int)\s*\[\s*(\d+)\s*\]", type_str)
    if m2:
        try:
            fixed_k = int(m2.group(1))
        except Exception:
            fixed_k = None

    return name, type_str, has_default, fixed_k, default_str


def parse_schema_str(schema_str: str) -> Dict[str, Dict[str, Any]]:
    args_section = _extract_args_section(schema_str)
    parts = _split_top_level_commas(args_section)
    info: Dict[str, Dict[str, Any]] = {}

    for part in parts:
        parsed = _parse_one_arg(part)
        if not parsed:
            continue
        name, type_str, has_default, fixed_k, default_str = parsed
        info[name] = {
            "has_default": has_default,
            "fixed_arity": fixed_k,
            "type_str": type_str,
            "default_str": default_str,
        }
    return info


# -----------------------
# helpers
# -----------------------
def _ensure_int(x: Any) -> Optional[int]:
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    return None


def _ensure_float(x: Any) -> Optional[float]:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    return None


def _normalize_rank_hints(data: Dict[str, Any]) -> bool:
    """
    Ensure top-level rank_hints exists and has stable structure.
    No evidence fields.
    """
    changed = False
    rh = data.get("rank_hints")

    # default stable block
    default = {
        "marker": RANK_MISS_MARKER,
        "status": "missing",  # missing/unassigned/assigned
        "rank_candidates": [RANK_MISS_MARKER],
        "rank_any": None,
        "rank_min": None,
        "rank_max": None,
    }

    if not isinstance(rh, dict):
        data["rank_hints"] = default
        return True

    # marker
    if "marker" not in rh or not isinstance(rh.get("marker"), str) or not rh.get("marker"):
        rh["marker"] = default["marker"]
        changed = True

    # status normalize
    st = rh.get("status")
    if st not in ("missing", "unassigned", "assigned"):
        # map some common variants
        if st in ("miss", "none", None):
            rh["status"] = "missing"
        else:
            rh["status"] = "unassigned"
        changed = True

    # rank_candidates
    rc = rh.get("rank_candidates")
    if not isinstance(rc, list) or len(rc) == 0:
        rh["rank_candidates"] = default["rank_candidates"]
        changed = True
    else:
        # normalize ints when possible; keep markers if present
        norm: List[Any] = []
        for x in rc:
            if isinstance(x, int) and not isinstance(x, bool):
                norm.append(int(x))
            elif isinstance(x, str):
                norm.append(x)
            else:
                # drop unknown
                continue
        if not norm:
            rh["rank_candidates"] = default["rank_candidates"]
            changed = True
        else:
            # unique & sorted for ints, keep strings at end
            ints = sorted({v for v in norm if isinstance(v, int)})
            strs = [v for v in norm if isinstance(v, str)]
            rh["rank_candidates"] = ints + strs if ints else strs
            changed = True

    for k in ("rank_any", "rank_min", "rank_max"):
        if k not in rh:
            rh[k] = default[k]
            changed = True

    return changed


def _apply_has_default_truth(param_spec: Dict[str, Any], truth_has_default: bool, truth_default_str: Optional[str]) -> bool:
    changed = False

    if not truth_has_default:
        for k in ("has_default", "default_repr", "default"):
            if k in param_spec:
                del param_spec[k]
                changed = True
        return changed

    if param_spec.get("has_default") is not True:
        param_spec["has_default"] = True
        changed = True

    if "default_repr" not in param_spec and truth_default_str is not None:
        param_spec["default_repr"] = truth_default_str
        changed = True

    return changed


def _bump_min_range(spec: Dict[str, Any], min_val: int) -> bool:
    r = spec.get("range")
    if not (isinstance(r, list) and len(r) == 2):
        return False
    lo_i = _ensure_int(r[0])
    hi_i = _ensure_int(r[1])
    if lo_i is None or hi_i is None:
        return False
    new_lo = max(lo_i, min_val)
    if new_lo == lo_i:
        return False
    spec["range"] = [new_lo, hi_i]
    return True


def _clamp_float_range(spec: Dict[str, Any], lo: float, hi: float) -> bool:
    r = spec.get("range")
    if not (isinstance(r, list) and len(r) == 2):
        return False
    a = _ensure_float(r[0])
    b = _ensure_float(r[1])
    if a is None or b is None:
        return False
    new_a = max(a, lo)
    new_b = min(b, hi)
    if new_a == a and new_b == b:
        return False
    # keep order sane
    if new_a > new_b:
        new_a, new_b = lo, hi
    spec["range"] = [new_a, new_b]
    return True


def _filter_int_or_tuple_values(spec: Dict[str, Any], min_val: int, fixed_k: Optional[int]) -> bool:
    vals = spec.get("values")
    if not isinstance(vals, list):
        return False

    changed = False
    new_vals: List[Any] = []
    for v in vals:
        if isinstance(v, bool):
            continue
        if isinstance(v, int):
            if v >= min_val:
                new_vals.append(v)
            else:
                changed = True
        elif isinstance(v, list):
            if fixed_k is not None and len(v) != fixed_k:
                changed = True
                continue
            ok = True
            for e in v:
                if not isinstance(e, int) or isinstance(e, bool) or e < min_val:
                    ok = False
                    break
            if ok:
                new_vals.append(v)
            else:
                changed = True
        else:
            new_vals.append(v)

    # ensure some base candidates
    if min_val >= 1 and 1 not in new_vals:
        new_vals.insert(0, 1)
        changed = True
    if min_val <= 0 and 0 not in new_vals:
        new_vals.insert(0, 0)
        changed = True

    if fixed_k is not None:
        base = [max(min_val, 1)] * fixed_k
        if base not in new_vals:
            new_vals.append(base)
            changed = True

    # dedup (preserve order)
    seen = set()
    deduped = []
    for x in new_vals:
        key = ("i", x) if isinstance(x, int) else ("l", tuple(x)) if isinstance(x, list) else ("o", str(x))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(x)

    spec["values"] = deduped
    return changed


def _force_fixed_arity_on_int_list(spec: Dict[str, Any], k: int) -> bool:
    if spec.get("kind") != "int_list":
        return False
    changed = False
    if spec.get("len_range") != [k, k]:
        spec["len_range"] = [k, k]
        changed = True
    if "default" in spec and isinstance(spec["default"], list) and len(spec["default"]) != k:
        del spec["default"]
        changed = True
    return changed

def _ensure_enum_values(spec: Dict[str, Any], required: List[str]) -> bool:
    """
    Ensure enum 'values' contains all required strings.
    Returns whether changed.
    """
    if spec.get("kind") != "enum":
        return False
    vals = spec.get("values")
    if not isinstance(vals, list):
        return False

    changed = False
    existing = set()
    for v in vals:
        if isinstance(v, str):
            existing.add(v)

    for r in required:
        if r not in existing:
            vals.append(r)
            existing.add(r)
            changed = True

    spec["values"] = vals
    return changed



def _upgrade_int_list_to_int_or_tuple(spec: Dict[str, Any], k: int, min_val: int, default_list: Optional[List[int]]) -> bool:
    """
    Upgrade kind:int_list -> kind:int_or_tuple with candidate values.
    This better matches many ATen args that are effectively "int or tuple".
    """
    if spec.get("kind") != "int_list":
        return False

    changed = False
    spec["kind"] = "int_or_tuple"
    changed = True

    # build candidates
    values: List[Any] = []

    # add scalar candidates
    base_scalars = [max(min_val, 1), max(min_val, 2), max(min_val, 3)]
    for s in base_scalars:
        if s not in values:
            values.append(s)

    # add tuple candidates
    for t in ([max(min_val, 1)] * k, [max(min_val, 2)] * k, [max(min_val, 3)] * k):
        if t not in values:
            values.append(t)

    if default_list and len(default_list) == k and default_list not in values:
        values.insert(0, default_list)

    # drop int_list specific fields
    for drop in ("len_range", "range", "default"):
        if drop in spec:
            # keep default as candidate inserted above
            del spec[drop]
            changed = True

    spec["values"] = values
    return changed


def _force_bool_kind(spec: Dict[str, Any], default_str: Optional[str]) -> bool:
    if spec.get("kind") == "bool":
        # ensure default if we can parse it
        if default_str in ("True", "False") and spec.get("default") is None:
            spec["default"] = (default_str == "True")
            return True
        return False

    # convert from enum/bad kinds
    spec.clear()
    spec["kind"] = "bool"
    spec["default"] = (default_str == "True") if default_str in ("True", "False") else None
    return True


def _normalize_enum_todo(spec: Dict[str, Any]) -> bool:
    """
    Ensure enum placeholder is consistent for unknown enums like dtype/layout/device.
    """
    if spec.get("kind") != "enum":
        return False
    vals = spec.get("values")
    if not isinstance(vals, list) or not vals:
        spec["values"] = [ENUM_TODO_MARKER]
        return True
    # avoid misleading singletons like ["False"] for Optional[bool] (handled elsewhere),
    # or ["TODO"] inconsistent marker.
    if len(vals) == 1 and isinstance(vals[0], str) and vals[0] in ("TODO", ""):
        spec["values"] = [ENUM_TODO_MARKER]
        return True
    return False


def normalize_one_yaml(data: Dict[str, Any], fail_on_parse_error: bool = False) -> Tuple[Dict[str, Any], Dict[str, int]]:
    stats = {
        "files_changed": 0,
        "params_touched": 0,
        "default_fixed": 0,
        "range_fixed": 0,
        "fixed_arity_fixed": 0,
        "kind_fixed": 0,
        "rank_hints_fixed": 0,
        "generator_fixed": 0,
    }

    changed_any = False

    # ensure generator stamp
    if not isinstance(data.get("generator"), dict):
        data["generator"] = GENERATOR_BLOCK.copy()
        stats["generator_fixed"] += 1
        changed_any = True

    # ensure rank_hints stable
    if _normalize_rank_hints(data):
        stats["rank_hints_fixed"] += 1
        changed_any = True

    aten = (data.get("aten") or {})
    schema_str = aten.get("schema_str") or ""
    try:
        schema_info = parse_schema_str(schema_str)
    except Exception:
        if fail_on_parse_error:
            raise
        schema_info = {}

    params = data.get("params")
    if not isinstance(params, dict):
        if changed_any:
            stats["files_changed"] = 1
        return data, stats

    for name, spec in params.items():
        if not isinstance(spec, dict):
            continue
        touched = False

        # Use schema_str truth when available
        srec = schema_info.get(name)
        type_str = (srec.get("type_str") if isinstance(srec, dict) else "") or ""
        truth_default_str = (srec.get("default_str") if isinstance(srec, dict) else None)
        fixed_k = (srec.get("fixed_arity") if isinstance(srec, dict) else None)

        # A: has_default truth
        if isinstance(srec, dict):
            truth_has_default = bool(srec.get("has_default", False))
            if _apply_has_default_truth(spec, truth_has_default, truth_default_str):
                stats["default_fixed"] += 1
                touched = True

        # B: kind fix from schema type (bool & optional bool)
        # Recognize bool in schema_str fragments
        if re.search(r"\bbool\??\b", type_str):
            if _force_bool_kind(spec, truth_default_str):
                stats["kind_fixed"] += 1
                touched = True

        # C: enum placeholders for dtype/layout/device/etc
        # Examples in schema_str:
        #   ScalarType? dtype=None, Layout? layout=None, Device? device=None
        if any(tok in type_str for tok in ("ScalarType", "Layout", "Device")) or name in ("dtype", "layout", "device"):
            # keep enum kind, but standardize values marker
            if spec.get("kind") != "enum":
                # don't nuke other fields like has_default/default_repr
                keep_has_default = spec.get("has_default")
                keep_default_repr = spec.get("default_repr")
                spec.clear()
                spec["kind"] = "enum"
                spec["values"] = [ENUM_TODO_MARKER]
                if keep_has_default is True:
                    spec["has_default"] = True
                if keep_default_repr is not None:
                    spec["default_repr"] = keep_default_repr
                stats["kind_fixed"] += 1
                touched = True
            else:
                if _normalize_enum_todo(spec):
                    stats["kind_fixed"] += 1
                    touched = True

        # D: fixed arity [k] from schema_str
        if isinstance(fixed_k, int) and fixed_k > 0:
            # for int_list keep fixed arity
            if _force_fixed_arity_on_int_list(spec, fixed_k):
                stats["fixed_arity_fixed"] += 1
                touched = True

            # upgrade int_list -> int_or_tuple for k=2/3 common knobs
            if spec.get("kind") == "int_list" and fixed_k in (2, 3):
                # choose min_val based on name
                min_val = 0
                if name in ("stride", "dilation", "kernel_size"):
                    min_val = 1
                if name in ("padding", "output_padding"):
                    min_val = 0
                default_list = None
                if isinstance(truth_default_str, str) and truth_default_str.startswith("[") and truth_default_str.endswith("]"):
                    # best-effort parse list like [1, 1]
                    try:
                        default_list = [int(x.strip()) for x in truth_default_str.strip("[]").split(",") if x.strip()]
                    except Exception:
                        default_list = None

                if _upgrade_int_list_to_int_or_tuple(spec, fixed_k, min_val, default_list):
                    stats["kind_fixed"] += 1
                    touched = True

            # if int_or_tuple, enforce list length == k
            if spec.get("kind") == "int_or_tuple":
                vals = spec.get("values")
                if isinstance(vals, list):
                    new_vals = []
                    local_changed = False
                    for v in vals:
                        if isinstance(v, list) and len(v) != fixed_k:
                            local_changed = True
                            continue
                        new_vals.append(v)
                    if local_changed:
                        spec["values"] = new_vals
                        stats["fixed_arity_fixed"] += 1
                        touched = True

        # E: universal range fixes by name (low-risk, good for fuzz)
        if name in ("groups", "stride", "dilation", "kernel_size"):
            if spec.get("kind") in ("int", "int_list") and _bump_min_range(spec, 1):
                stats["range_fixed"] += 1
                touched = True
            if spec.get("kind") == "int_or_tuple":
                if _filter_int_or_tuple_values(spec, 1, fixed_k if isinstance(fixed_k, int) else None):
                    stats["range_fixed"] += 1
                    touched = True

        if name in ("padding", "output_padding"):
            if spec.get("kind") in ("int", "int_list") and _bump_min_range(spec, 0):
                stats["range_fixed"] += 1
                touched = True
            if spec.get("kind") == "int_or_tuple":
                if _filter_int_or_tuple_values(spec, 0, fixed_k if isinstance(fixed_k, int) else None):
                    stats["range_fixed"] += 1
                    touched = True
        
                # F: enum padding: if values contains "valid", ensure it also contains "same"
        if name == "padding" and spec.get("kind") == "enum":
            vals = spec.get("values")
            if isinstance(vals, list):
                # case/quote tolerant match for "valid"
                has_valid = any(
                    isinstance(v, str) and v.strip().strip('"').strip("'").lower() == "valid"
                    for v in vals
                )
                if has_valid:
                    # add "same" if missing
                    has_same = any(
                        isinstance(v, str) and v.strip().strip('"').strip("'").lower() == "same"
                        for v in vals
                    )
                    if not has_same:
                        vals.append("same")
                        spec["values"] = vals
                        stats["kind_fixed"] += 1  # 或者你愿意新加 enum_fixed 统计也行
                        touched = True

        # probability-like float params: clamp to [0,1]
        if spec.get("kind") == "float" and name in ("p", "prob", "dropout"):
            if _clamp_float_range(spec, 0.0, 1.0):
                stats["range_fixed"] += 1
                touched = True

        # eps-like float params: bump low to 1e-12
        if spec.get("kind") == "float" and "eps" in name:
            r = spec.get("range")
            if isinstance(r, list) and len(r) == 2:
                lo = _ensure_float(r[0])
                hi = _ensure_float(r[1])
                if lo is not None and hi is not None and lo <= 0:
                    spec["range"] = [1e-12, hi]
                    stats["range_fixed"] += 1
                    touched = True

        if touched:
            stats["params_touched"] += 1
            changed_any = True

    if changed_any:
        stats["files_changed"] = 1
    return data, stats


# -----------------------
# CLI
# -----------------------
def iter_yaml_files(src: Path) -> List[Path]:
    if src.is_dir():
        return sorted(list(src.glob("*.yaml")) + list(src.glob("*.yml")))
    return [src]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="a yaml file or a directory containing yaml skeletons")
    ap.add_argument("--out_dir", default="", help="output dir; if empty, overwrite in-place")
    ap.add_argument("--dry_run", action="store_true", help="do not write files, only print stats")
    ap.add_argument("--fail_on_parse_error", action="store_true", help="fail if schema_str parsing errors")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    files = iter_yaml_files(src)
    if not files:
        raise SystemExit(f"No yaml files found under: {src}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    total = {
        "files": 0,
        "files_changed": 0,
        "params_touched": 0,
        "default_fixed": 0,
        "range_fixed": 0,
        "fixed_arity_fixed": 0,
        "kind_fixed": 0,
        "rank_hints_fixed": 0,
        "generator_fixed": 0,
    }

    for yp in files:
        raw = yp.read_text(encoding="utf-8", errors="ignore")
        data = yaml.safe_load(raw)
        if not isinstance(data, dict):
            print(f"[!] skip non-dict yaml: {yp}")
            continue

        new_data, stats = normalize_one_yaml(data, fail_on_parse_error=args.fail_on_parse_error)

        total["files"] += 1
        for k in total.keys():
            if k == "files":
                continue
            if k in stats:
                total[k] += stats[k]

        if args.dry_run:
            print(
                f"[DRY] {yp.name}: changed={bool(stats['files_changed'])}, "
                f"touched={stats['params_touched']}, "
                f"default_fixed={stats['default_fixed']}, range_fixed={stats['range_fixed']}, "
                f"fixed_arity_fixed={stats['fixed_arity_fixed']}, kind_fixed={stats['kind_fixed']}, "
                f"rank_hints_fixed={stats['rank_hints_fixed']}, generator_fixed={stats['generator_fixed']}"
            )
            continue

        out_path = (out_dir / yp.name) if out_dir is not None else yp
        out_path.write_text(
            yaml.safe_dump(new_data, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        if stats["files_changed"]:
            print(f"[+] normalized: {yp} -> {out_path}")
        else:
            print(f"[=] unchanged:  {yp} -> {out_path}")

    print("\n=== Summary ===")
    for k, v in total.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
