#!/usr/bin/env python3
# normalize_yaml_skeleton.py
import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


# -----------------------
# schema_str parsing
# -----------------------
def _extract_args_section(schema_str: str) -> str:
    """
    Extract the substring inside the first (...) of schema_str.
    Example:
      'aten::conv2d(Tensor input, Tensor weight, Tensor? bias=None, SymInt[2] stride=[1,1]) -> Tensor'
      -> 'Tensor input, Tensor weight, Tensor? bias=None, SymInt[2] stride=[1,1]'
    """
    if not schema_str:
        return ""
    l = schema_str.find("(")
    if l < 0:
        return ""
    # find the matching ')'
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
    """
    Split by commas, but ignore commas inside [] or ().
    """
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
    """
    Parse one argument fragment like:
      'Tensor? bias=None'
      'SymInt[2] stride=[1,1]'
      'Tensor(a!) self'
      '*, SymInt dim'  (here '*' is a standalone token)
    Returns:
      (name, type_str, has_default, fixed_arity_k, default_str)
    """
    p = part.strip()
    if not p:
        return None
    if p == "*":  # kw-only separator
        return None

    # Split default
    eq_pos = p.find("=")
    if eq_pos >= 0:
        left = p[:eq_pos].strip()
        default_str = p[eq_pos + 1 :].strip()
        has_default = True
    else:
        left = p
        default_str = None
        has_default = False

    # name = last identifier token in left
    m = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*$", left)
    if not m:
        return None
    name = m.group(1)
    type_str = left[: m.start(1)].strip()

    # fixed arity detection in type_str: SymInt[2], int[3], SymInt?[2] (rare)
    fixed_k = None
    m2 = re.search(r"\b(?:SymInt|int)\s*\[\s*(\d+)\s*\]", type_str)
    if m2:
        try:
            fixed_k = int(m2.group(1))
        except Exception:
            fixed_k = None

    return name, type_str, has_default, fixed_k, default_str


def parse_schema_str(schema_str: str) -> Dict[str, Dict[str, Any]]:
    """
    Returns mapping:
      arg_name -> {
        'has_default': bool (truth from schema_str),
        'fixed_arity': Optional[int],
        'type_str': str,
        'default_str': Optional[str],
      }
    """
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


def _ensure_int(x: Any) -> Optional[int]:
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    return None


# -----------------------
# normalization rules
# -----------------------
def _apply_has_default_truth(param_spec: Dict[str, Any], truth_has_default: bool, truth_default_str: Optional[str]) -> bool:
    """
    Update YAML param spec in-place:
      - If truth_has_default is False: remove has_default/default_repr/default if present.
      - If truth_has_default is True: set has_default=True; if default_repr missing, store best-effort default_repr from schema_str.
    Returns whether changed.
    """
    changed = False

    if not truth_has_default:
        # remove noisy fields
        for k in ("has_default", "default_repr", "default"):
            if k in param_spec:
                del param_spec[k]
                changed = True
        return changed

    # truth has default
    if param_spec.get("has_default") is not True:
        param_spec["has_default"] = True
        changed = True

    # preserve existing default_repr if it exists; otherwise add a best-effort one
    if "default_repr" not in param_spec and truth_default_str is not None:
        # Note: this is not always python repr(), but it's still useful for later stages
        param_spec["default_repr"] = truth_default_str
        changed = True

    return changed


def _bump_min_range(spec: Dict[str, Any], min_val: int) -> bool:
    """
    If spec has 'range': [lo, hi], bump lo to max(lo, min_val).
    For int_list also bumps 'range' if present.
    """
    if "range" not in spec:
        return False
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


def _filter_int_or_tuple_values(spec: Dict[str, Any], min_val: int, fixed_k: Optional[int]) -> bool:
    """
    For kind == int_or_tuple, filter invalid scalar/list candidates.
    - scalar candidate must be >= min_val
    - list candidate elements must be >= min_val
    - if fixed_k is not None: list candidates must have len == fixed_k
    Also ensure basic candidates exist.
    """
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
            # unknown type, keep (but it's rare); to be safe, keep it
            new_vals.append(v)

    # Ensure basic candidates
    if min_val >= 1:
        if 1 not in new_vals:
            new_vals.insert(0, 1)
            changed = True
    else:
        if 0 not in new_vals:
            new_vals.insert(0, 0)
            changed = True

    if fixed_k is not None:
        base1 = [max(min_val, 1)] * fixed_k
        if base1 not in new_vals:
            new_vals.append(base1)
            changed = True

    spec["values"] = new_vals
    return changed


def _force_fixed_arity_on_int_list(spec: Dict[str, Any], k: int) -> bool:
    """
    If kind == int_list, force len_range = [k,k].
    If default exists but length mismatched, drop default.
    """
    if spec.get("kind") != "int_list":
        return False

    changed = False
    lr = spec.get("len_range")
    if lr != [k, k]:
        spec["len_range"] = [k, k]
        changed = True

    if "default" in spec and isinstance(spec["default"], list):
        if len(spec["default"]) != k:
            del spec["default"]
            changed = True

    return changed


def normalize_one_yaml(data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    Normalize one YAML dict and return (data, stats)
    """
    stats = {
        "files_changed": 0,
        "params_touched": 0,
        "default_fixed": 0,
        "range_fixed": 0,
        "fixed_arity_fixed": 0,
    }

    aten = (data.get("aten") or {})
    schema_str = aten.get("schema_str") or ""
    schema_info = parse_schema_str(schema_str)

    params = data.get("params")
    if not isinstance(params, dict):
        return data, stats

    changed_any = False

    for name, spec in params.items():
        if not isinstance(spec, dict):
            continue

        touched = False

        # Rule A: has_default truth from schema_str (name=)
        if name in schema_info:
            truth = bool(schema_info[name]["has_default"])
            truth_default_str = schema_info[name].get("default_str")
            if _apply_has_default_truth(spec, truth, truth_default_str):
                stats["default_fixed"] += 1
                touched = True

            # Rule C: fixed arity [k] from schema_str for int_list
            k = schema_info[name].get("fixed_arity")
            if isinstance(k, int) and k > 0:
                if _force_fixed_arity_on_int_list(spec, k):
                    stats["fixed_arity_fixed"] += 1
                    touched = True

                # also help int_or_tuple: filter values / ensure k-lists
                if spec.get("kind") == "int_or_tuple":
                    # choose a conservative min_val for tuple lists: don't enforce here unless name is in known set below
                    pass

        # Rule B: range lower bound bumps for obvious params
        # groups -> >=1
        if name == "groups":
            if spec.get("kind") in ("int", "int_list") and _bump_min_range(spec, 1):
                stats["range_fixed"] += 1
                touched = True
            if spec.get("kind") == "int_or_tuple":
                fk = schema_info.get(name, {}).get("fixed_arity")
                if _filter_int_or_tuple_values(spec, 1, fk if isinstance(fk, int) else None):
                    stats["range_fixed"] += 1
                    touched = True

        # stride / dilation -> >=1
        if name in ("stride", "dilation"):
            if spec.get("kind") in ("int", "int_list") and _bump_min_range(spec, 1):
                stats["range_fixed"] += 1
                touched = True
            if spec.get("kind") == "int_or_tuple":
                fk = schema_info.get(name, {}).get("fixed_arity")
                if _filter_int_or_tuple_values(spec, 1, fk if isinstance(fk, int) else None):
                    stats["range_fixed"] += 1
                    touched = True

        # padding -> >=0
        if name == "padding":
            if spec.get("kind") in ("int", "int_list") and _bump_min_range(spec, 0):
                stats["range_fixed"] += 1
                touched = True
            if spec.get("kind") == "int_or_tuple":
                fk = schema_info.get(name, {}).get("fixed_arity")
                if _filter_int_or_tuple_values(spec, 0, fk if isinstance(fk, int) else None):
                    stats["range_fixed"] += 1
                    touched = True

        # If schema says fixed arity and spec is int_or_tuple, enforce list candidates length == k (and keep scalars)
        if name in schema_info and spec.get("kind") == "int_or_tuple":
            k = schema_info[name].get("fixed_arity")
            if isinstance(k, int) and k > 0:
                # don't bump min_val here unless name is one of the known "non-negative" knobs above;
                # just enforce length by filtering list entries with wrong length.
                vals = spec.get("values")
                if isinstance(vals, list):
                    new_vals = []
                    local_changed = False
                    for v in vals:
                        if isinstance(v, list) and len(v) != k:
                            local_changed = True
                            continue
                        new_vals.append(v)
                    if local_changed:
                        spec["values"] = new_vals
                        stats["fixed_arity_fixed"] += 1
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
    }

    for yp in files:
        raw = yp.read_text(encoding="utf-8", errors="ignore")
        data = yaml.safe_load(raw)
        if not isinstance(data, dict):
            print(f"[!] skip non-dict yaml: {yp}")
            continue

        new_data, stats = normalize_one_yaml(data)

        total["files"] += 1
        total["files_changed"] += stats["files_changed"]
        total["params_touched"] += stats["params_touched"]
        total["default_fixed"] += stats["default_fixed"]
        total["range_fixed"] += stats["range_fixed"]
        total["fixed_arity_fixed"] += stats["fixed_arity_fixed"]

        if args.dry_run:
            print(f"[DRY] {yp.name}: changed={bool(stats['files_changed'])}, touched={stats['params_touched']}, "
                  f"default_fixed={stats['default_fixed']}, range_fixed={stats['range_fixed']}, "
                  f"fixed_arity_fixed={stats['fixed_arity_fixed']}")
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
    print(f"files: {total['files']}")
    print(f"files_changed: {total['files_changed']}")
    print(f"params_touched: {total['params_touched']}")
    print(f"default_fixed: {total['default_fixed']}")
    print(f"range_fixed: {total['range_fixed']}")
    print(f"fixed_arity_fixed: {total['fixed_arity_fixed']}")


if __name__ == "__main__":
    main()
