#!/usr/bin/env python3
# llm_patch_yaml.py
#
# Stage C (new paradigm):
#   - Give LLM the doc text + current YAML skeleton
#   - Ask LLM to output a COMPLETE candidate YAML
#   - Parse candidate YAML
#   - Extract ONLY allowed Stage-C fields from candidate:
#       * shape_vars
#       * params.*.shape_spec
#       * params.*.shape_spec_by_rank
#   - Normalize them
#   - Merge them back into the original YAML
#
# This avoids relying on fragile patch formats.

import os
import re
import copy
import json
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set

import yaml
from openai import OpenAI, BadRequestError
from llm_prompts import YAML_PATCH_SYSTEM_PROMPT


# ----------------------------
# 1) helpers
# ----------------------------
def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")


def safe_truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-max_chars // 2 :]
    return head + "\n\n[...TRUNCATED...]\n\n" + tail


def load_yaml_obj(yaml_path: Path) -> Any:
    return yaml.safe_load(yaml_path.read_text(encoding="utf-8", errors="ignore"))


def dump_yaml_obj(obj: Any) -> str:
    return yaml.safe_dump(
        obj,
        sort_keys=False,
        allow_unicode=True,
        width=1000,
        default_flow_style=False,
    )


def safe_name(s: Any, max_len: int = 120) -> str:
    if s is None:
        return "null"
    s = str(s).strip()
    if not s:
        return "empty"
    s = s.replace("::", "_").replace("/", "_").replace("\\", "_")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("._-")
    if not s:
        s = "empty"
    return s[:max_len]


def is_tensor_kind(kind: Any) -> bool:
    return isinstance(kind, str) and kind.startswith("tensor")


def infer_primary_param_from_base(base_yaml: Dict[str, Any]) -> Optional[str]:
    params = base_yaml.get("params")
    if not isinstance(params, dict):
        return None

    if "input" in params and isinstance(params["input"], dict) and is_tensor_kind(params["input"].get("kind")):
        return "input"

    if "self" in params and isinstance(params["self"], dict) and is_tensor_kind(params["self"].get("kind")):
        return "self"

    for pname, p in params.items():
        if isinstance(p, dict) and is_tensor_kind(p.get("kind")):
            return pname

    return None


# def get_expected_ranks(base_yaml: Dict[str, Any]) -> List[int]:
#     rank_hints = base_yaml.get("rank_hints") or {}
#     out: List[int] = []
#     for x in rank_hints.get("rank_candidates") or []:
#         try:
#             out.append(int(x))
#         except Exception:
#             pass
#     return sorted(set(out))


def default_range_for_dim(name: str) -> List[int]:
    n = str(name).strip()

    if n == "N":
        return [1, 8]
    if n in ("C", "C_in", "C_out", "C_per_group", "groups"):
        return [1, 64]
    if n in ("H", "W", "L"):
        return [1, 128]
    if n == "D":
        return [1, 32]
    if n in ("KH", "KW", "KD"):
        return [1, 11]
    if n in ("M", "K", "P", "Q", "R", "S", "T", "U", "V"):
        return [1, 128]

    return [1, 64]


# ----------------------------
# 2) YAML extraction / parsing
# ----------------------------
_FENCED_YAML_RE = re.compile(
    r"```(?:yaml|yml)?\s*(.*?)```",
    re.IGNORECASE | re.DOTALL,
)


def extract_yaml_block(text: str) -> str:
    """
    Extract candidate YAML from LLM output.

    Priority:
      1) fenced ```yaml ... ```
      2) fenced ``` ... ```
      3) substring starting from first line that looks like YAML top-level key
      4) whole text
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("empty model output")

    m = _FENCED_YAML_RE.search(text)
    if m:
        block = m.group(1).strip()
        if block:
            return block

    lines = text.splitlines()
    start_idx = None
    for i, ln in enumerate(lines):
        if re.match(r"^[A-Za-z_][A-Za-z0-9_-]*\s*:\s*", ln):
            start_idx = i
            break

    if start_idx is not None:
        block = "\n".join(lines[start_idx:]).strip()
        if block:
            return block

    return text


def parse_candidate_yaml(text: str) -> Dict[str, Any]:
    """
    Parse candidate YAML text into a mapping/dict.
    """
    block = extract_yaml_block(text)
    obj = yaml.safe_load(block)
    if not isinstance(obj, dict):
        raise ValueError("candidate YAML is not a mapping/dict")
    return obj


# ----------------------------
# 3) normalization
# ----------------------------
_ALLOWED_BUILTINS = {
    "isinstance", "all", "any", "len", "tuple",
    "min", "max", "abs",
}

def normalize_rank_candidates(v: Any) -> List[int]:
    out: List[int] = []

    if not isinstance(v, (list, tuple, set)):
        return out

    for x in v:
        try:
            r = int(x)
        except Exception:
            continue
        if r >= 0:
            out.append(r)

    return sorted(set(out))


def get_expected_ranks(yaml_obj: Dict[str, Any]) -> List[int]:
    if not isinstance(yaml_obj, dict):
        return []

    rank_hints = yaml_obj.get("rank_hints") or {}
    if not isinstance(rank_hints, dict):
        return []

    return normalize_rank_candidates(rank_hints.get("rank_candidates"))


def collect_explicit_rank_candidates_from_shape_spec(spec: Any) -> List[int]:
    """
    只从显式 finite shape candidates 里收集 rank，不处理 variadic "...".

    例如:
      [K] -> [1]
      [[K], [M, K]] -> [1, 2]
      ["[K]", "[M, K]"] -> [1, 2]
      [[K], ["...", M, K]] -> [1]
    """
    out: Set[int] = set()

    single = normalize_shape_spec_list(spec)
    if single is not None:
        out.add(len(single))
        return sorted(out)

    if isinstance(spec, list):
        for item in spec:
            one = normalize_shape_spec_list(item)
            if one is not None:
                out.add(len(one))

    return sorted(out)


def contains_variadic_shape_candidate(spec: Any) -> bool:
    """
    判断 spec 里是否包含 variadic 形式，例如:
      ["...", M, K]
      "[..., M, K]"
      [[K], ["...", M, K]]
    """
    if normalize_variadic_shape_spec_list(spec) is not None:
        return True

    if isinstance(spec, list):
        for item in spec:
            if normalize_variadic_shape_spec_list(item) is not None:
                return True

    return False


def infer_effective_ranks(
    base: Dict[str, Any],
    candidate: Dict[str, Any],
    primary_param: Optional[str],
) -> List[int]:
    """
    rank 来源优先级：
      1) candidate.rank_hints.rank_candidates
      2) candidate 中 primary_param 的 shape_spec_by_rank keys
      3) candidate 中 primary_param 的显式 finite shape_spec 候选长度
      4) candidate 其他 param 的 shape_spec_by_rank keys
      5) base.rank_hints.rank_candidates

    不再使用硬编码 fallback [1,2,3,4] / [2,3,4,5]。
    """
    cand_ranks: Set[int] = set()

    # 1) candidate.rank_hints.rank_candidates
    cand_rank_hints = candidate.get("rank_hints")
    if isinstance(cand_rank_hints, dict):
        cand_ranks.update(normalize_rank_candidates(cand_rank_hints.get("rank_candidates")))

    cand_params = candidate.get("params")
    if isinstance(cand_params, dict):
        # 2) primary_param.shape_spec_by_rank
        if isinstance(primary_param, str):
            cp = cand_params.get(primary_param)
            if isinstance(cp, dict):
                sbr = normalize_shape_spec_by_rank(cp.get("shape_spec_by_rank"))
                cand_ranks.update(int(k) for k in sbr.keys())

                # 3) primary_param 的显式 finite shape_spec
                cand_ranks.update(collect_explicit_rank_candidates_from_shape_spec(cp.get("shape_spec")))

        # 4) 其他 param 的 shape_spec_by_rank
        for pname, cp in cand_params.items():
            if pname == primary_param or not isinstance(cp, dict):
                continue
            sbr = normalize_shape_spec_by_rank(cp.get("shape_spec_by_rank"))
            cand_ranks.update(int(k) for k in sbr.keys())

    if cand_ranks:
        return sorted(cand_ranks)

    # 5) fallback to base rank_hints
    return get_expected_ranks(base)


def expand_variadic_shape_spec_for_common_ranks(
    spec: List[str],
    expected_ranks: List[int],
) -> Dict[str, List[str]]:
    """
    把单个 variadic spec 展开成有限 rank->spec.

    例子：
      ["...", M, K] + [1,2,3,4]
        -> {
             "2": [M, K],
             "3": [B1, M, K],
             "4": [B1, B2, M, K]
           }

      [N, C, "..."] + [2,3,4,5]
        -> {
             "2": [N, C],
             "3": [N, C, L],
             "4": [N, C, H, W],
             "5": [N, C, D, H, W]
           }

    不再在 expected_ranks 为空时私自兜底。
    """
    if not isinstance(spec, list) or spec.count("...") != 1:
        return {}

    ranks = sorted(set(int(x) for x in expected_ranks if isinstance(x, int) or str(x).isdigit()))
    if not ranks:
        return {}

    ell_idx = spec.index("...")
    prefix = spec[:ell_idx]
    suffix = spec[ell_idx + 1:]

    out: Dict[str, List[str]] = {}

    for r in ranks:
        extra = r - len(prefix) - len(suffix)
        if extra < 0:
            continue

        # 1) 省略号在最前面：通常是 batch dims，例如 [..., M, K]
        if ell_idx == 0:
            middle = [f"B{i+1}" for i in range(extra)]

        # 2) 省略号在最后面：通常是 spatial dims，例如 [N, C, ...]
        elif ell_idx == len(spec) - 1:
            if extra == 0:
                middle = []
            elif extra == 1:
                middle = ["L"]
            elif extra == 2:
                middle = ["H", "W"]
            elif extra == 3:
                middle = ["D", "H", "W"]
            else:
                middle = [f"X{i+1}" for i in range(extra)]

        # 3) 省略号在中间：保守使用通用 X1/X2/...
        else:
            middle = [f"X{i+1}" for i in range(extra)]

        full = prefix + middle + suffix
        if len(full) == r and all(is_plain_var_name(x) for x in full):
            out[str(r)] = full

    return dict(sorted(out.items(), key=lambda kv: int(kv[0])))


def normalize_shape_spec_variants(
    raw_ss: Any,
    expected_ranks: List[int],
) -> Dict[str, List[str]]:
    """
    把 shape_spec 的多种写法统一归一成 rank->spec。

    支持：
      [K]
      "[K]"
      ["[K]"]
      ["...", M, K]
      "[..., M, K]"
      [[K], ["...", M, K]]
      ["[K]", "[..., M, K]"]
      [[M, K], [B1, M, K], [B1, B2, M, K]]

    返回：
      {"1": [K], "2": [M, K], ...}
    """
    out: Dict[str, List[str]] = {}

    # A) 单个 finite spec
    single = normalize_shape_spec_list(raw_ss)
    if single is not None:
        out[str(len(single))] = single
        return out

    # B) 单个 variadic spec
    variadic = normalize_variadic_shape_spec_list(raw_ss)
    if variadic is not None:
        return expand_variadic_shape_spec_for_common_ranks(variadic, expected_ranks)

    # C) 候选列表，例如 [[K], ["...", M, K]] / ["[K]", "[..., M, K]"]
    if isinstance(raw_ss, list) and raw_ss:
        saw_any = False

        for item in raw_ss:
            finite_item = normalize_shape_spec_list(item)
            if finite_item is not None:
                out[str(len(finite_item))] = finite_item
                saw_any = True
                continue

            variadic_item = normalize_variadic_shape_spec_list(item)
            if variadic_item is not None:
                expanded = expand_variadic_shape_spec_for_common_ranks(variadic_item, expected_ranks)
                if not expanded:
                    return {}
                out.update(expanded)
                saw_any = True
                continue

            return {}

        if saw_any:
            return dict(sorted(out.items(), key=lambda kv: int(kv[0])))

    return {}

def is_plain_var_name(x: Any) -> bool:
    if not isinstance(x, str):
        return False
    x = x.strip()
    if not x:
        return False
    if x == "TODO_SHAPE":
        return False
    if any(ch in x for ch in ("+", "-", "*", "/", "%", "(", ")", " ", "[", "]", ".", ",")):
        return False
    return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", x))

def parse_bracket_shape_string(s: str) -> Optional[List[str]]:
    """
    Parse strings like:
      "[C]" -> ["C"]
      "[N, C, ...]" -> ["N", "C", "..."]
    """
    if not isinstance(s, str):
        return None

    s = s.strip()
    if not (s.startswith("[") and s.endswith("]")):
        return None

    inner = s[1:-1].strip()
    if not inner:
        return None

    parts = [x.strip() for x in inner.split(",")]
    if not parts:
        return None

    out: List[str] = []
    for p in parts:
        if not p:
            return None
        if p == "...":
            out.append("...")
        elif is_plain_var_name(p):
            out.append(p)
        else:
            return None

    return out


def normalize_variadic_shape_spec_list(spec: Any) -> Optional[List[str]]:
    """
    Accept variadic forms like:
      ["[N, C, ...]"] -> ["N", "C", "..."]
      ["N", "C", "..."] -> ["N", "C", "..."]
    """
    if isinstance(spec, str):
        parsed = parse_bracket_shape_string(spec)
        if parsed and "..." in parsed:
            return parsed

    if isinstance(spec, list) and len(spec) == 1 and isinstance(spec[0], str):
        parsed = parse_bracket_shape_string(spec[0])
        if parsed and "..." in parsed:
            return parsed

    if isinstance(spec, list) and spec and all(isinstance(x, str) for x in spec):
        cleaned = [x.strip() for x in spec]
        if "..." in cleaned and all((x == "..." or is_plain_var_name(x)) for x in cleaned):
            return cleaned

    return None


def expand_variadic_shape_spec_for_common_ranks(
    spec: List[str],
    expected_ranks: List[int],
) -> Dict[str, List[str]]:
    """
    Expand:
      ["N", "C", "..."]
    into:
      {
        "2": ["N", "C"],
        "3": ["N", "C", "L"],
        "4": ["N", "C", "H", "W"],
        "5": ["N", "C", "D", "H", "W"],
      }

    If expected_ranks is empty, caller should supply a fallback like [2,3,4,5].
    """
    if not isinstance(spec, list) or spec.count("...") != 1:
        return {}

    ell_idx = spec.index("...")
    prefix = spec[:ell_idx]
    suffix = spec[ell_idx + 1:]

    out: Dict[str, List[str]] = {}
    for r in sorted(set(int(x) for x in expected_ranks if isinstance(x, int) or str(x).isdigit())):
        extra = r - len(prefix) - len(suffix)
        if extra < 0:
            continue

        if extra == 0:
            middle: List[str] = []
        elif extra == 1:
            middle = ["L"]
        elif extra == 2:
            middle = ["H", "W"]
        elif extra == 3:
            middle = ["D", "H", "W"]
        else:
            middle = [f"X{i+1}" for i in range(extra)]

        full = prefix + middle + suffix
        if len(full) == r and all(is_plain_var_name(x) for x in full):
            out[str(r)] = full

    return dict(sorted(out.items(), key=lambda kv: int(kv[0])))

def normalize_shape_var_entry(v: Any) -> Optional[List[int]]:
    """
    Accept:
      - [lo, hi]
      - {min: 1, max: 8}
      - {type: int, min: 1, max: 8}
      - int -> [1, int]  (rare fallback)
    """
    if isinstance(v, (list, tuple)) and len(v) == 2:
        try:
            lo = int(v[0])
            hi = int(v[1])
            if lo < 1:
                lo = 1
            if hi < lo:
                hi = lo
            return [lo, hi]
        except Exception:
            return None

    if isinstance(v, dict):
        lo = v.get("min", 1)
        hi = v.get("max", None)
        try:
            lo = int(lo)
        except Exception:
            lo = 1
        if hi is None:
            hi = max(lo, 8)
        try:
            hi = int(hi)
        except Exception:
            hi = max(lo, 8)
        if lo < 1:
            lo = 1
        if hi < lo:
            hi = lo
        return [lo, hi]

    if isinstance(v, int):
        lo = 1
        hi = max(1, int(v))
        return [lo, hi]

    return None


# def normalize_shape_vars_dict(shape_vars: Any) -> Dict[str, List[int]]:
#     out: Dict[str, List[int]] = {}
#     if not isinstance(shape_vars, dict):
#         return out

#     for k, v in shape_vars.items():
#         if not isinstance(k, str) or not k.strip():
#             continue
#         norm = normalize_shape_var_entry(v)
#         if norm is not None:
#             out[k.strip()] = norm

#     return out
def normalize_shape_vars_dict(shape_vars: Any) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    if not isinstance(shape_vars, dict):
        return out

    for k, v in shape_vars.items():
        if not isinstance(k, str) or not k.strip():
            continue

        key = k.strip()

        # support:
        #   N: int
        #   C: integer
        #   D: SymInt
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("int", "integer", "symint"):
                out[key] = default_range_for_dim(key)
                continue

        norm = normalize_shape_var_entry(v)
        if norm is not None:
            out[key] = norm

    return out


# def normalize_shape_spec_list(spec: Any) -> Optional[List[str]]:
#     """
#     Normalize a single shape spec into flat list[str].

#     Accept:
#       - ["N", "C", "H", "W"]
#       - [["C"]] -> ["C"]
#     Reject:
#       - multi-rank lists like [["N","C"], ["N","C","L"]]
#       - objects / descriptions / invalid tokens
#       - TODO_SHAPE
#     """
#     if isinstance(spec, list) and spec and all(isinstance(x, str) for x in spec):
#         cleaned = [x.strip() for x in spec if is_plain_var_name(x)]
#         if len(cleaned) == len(spec) and cleaned:
#             return cleaned
#         return None

#     if (
#         isinstance(spec, list)
#         and len(spec) == 1
#         and isinstance(spec[0], list)
#         and all(isinstance(x, str) for x in spec[0])
#     ):
#         cleaned = [x.strip() for x in spec[0] if is_plain_var_name(x)]
#         if len(cleaned) == len(spec[0]) and cleaned:
#             return cleaned
#         return None

#     return None
def normalize_shape_spec_list(spec: Any) -> Optional[List[str]]:
    """
    Normalize a single shape spec into flat list[str].

    Accept:
      - ["N", "C", "H", "W"]
      - [["C"]] -> ["C"]
      - "[C]" -> ["C"]
      - ["[C]"] -> ["C"]

    Reject:
      - multi-rank lists like [["N","C"], ["N","C","L"]]
      - variadic forms like ["N","C","..."]  (handled elsewhere)
      - objects / descriptions / invalid tokens
      - TODO_SHAPE
    """
    if isinstance(spec, str):
        parsed = parse_bracket_shape_string(spec)
        if parsed and "..." not in parsed:
            return parsed

    if isinstance(spec, list) and len(spec) == 1 and isinstance(spec[0], str):
        parsed = parse_bracket_shape_string(spec[0])
        if parsed and "..." not in parsed:
            return parsed

    if isinstance(spec, list) and spec and all(isinstance(x, str) for x in spec):
        cleaned = [x.strip() for x in spec]
        if "..." in cleaned:
            return None
        if all(is_plain_var_name(x) for x in cleaned):
            return cleaned
        return None

    if (
        isinstance(spec, list)
        and len(spec) == 1
        and isinstance(spec[0], list)
        and all(isinstance(x, str) for x in spec[0])
    ):
        cleaned = [x.strip() for x in spec[0]]
        if "..." in cleaned:
            return None
        if all(is_plain_var_name(x) for x in cleaned):
            return cleaned
        return None

    return None

def normalize_shape_spec_by_rank(shape_spec_by_rank: Any) -> Dict[str, List[str]]:
    """
    Normalize rank->shape_spec mapping.

    Accept:
      {
        "2": ["N","C"],
        "4": ["N","C","H","W"]
      }
    """
    out: Dict[str, List[str]] = {}
    if not isinstance(shape_spec_by_rank, dict):
        return out

    for rk, spec in shape_spec_by_rank.items():
        try:
            r = int(rk)
        except Exception:
            continue

        norm = normalize_shape_spec_list(spec)
        if norm is None:
            continue
        if len(norm) != r:
            continue

        out[str(r)] = norm

    return dict(sorted(out.items(), key=lambda kv: int(kv[0])))


def convert_multirank_shape_spec_list(spec: Any) -> Dict[str, List[str]]:
    """
    Convert shape_spec like:
      - [[N,C], [N,C,L], [N,C,H,W], [N,C,D,H,W]]
    into:
      {"2":[N,C], "3":[N,C,L], ...}
    """
    out: Dict[str, List[str]] = {}
    if not isinstance(spec, list):
        return out

    for item in spec:
        if not isinstance(item, list):
            return {}
        norm = normalize_shape_spec_list(item)
        if norm is None:
            return {}
        out[str(len(norm))] = norm

    return dict(sorted(out.items(), key=lambda kv: int(kv[0])))


def collect_vars_from_specs(
    param_overlay: Dict[str, Dict[str, Any]],
) -> Set[str]:
    vars_used: Set[str] = set()

    for upd in param_overlay.values():
        if not isinstance(upd, dict):
            continue

        ss = upd.get("shape_spec")
        if isinstance(ss, list):
            for x in ss:
                if isinstance(x, str):
                    vars_used.add(x)

        sbr = upd.get("shape_spec_by_rank")
        if isinstance(sbr, dict):
            for spec in sbr.values():
                if isinstance(spec, list):
                    for x in spec:
                        if isinstance(x, str):
                            vars_used.add(x)

    return vars_used


# ----------------------------
# 4) Stage-C extraction / merge
# ----------------------------
def extract_stagec_fields(candidate: Dict[str, Any], base: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract ONLY Stage-C allowed fields from candidate YAML.

    Returns overlay:
      {
        "primary_param": str|None,
        "rank_hints": {
          "rank_candidates": [..]    # optional
        },
        "shape_vars": {VAR:[lo,hi], ...},
        "params": {
          pname: {
            "shape_spec": [...],               # optional
            "shape_spec_by_rank": {..}         # optional
          }
        },
        "warnings": [...],
      }
    """
    overlay: Dict[str, Any] = {
        "primary_param": None,
        "rank_hints": {},
        "shape_vars": {},
        "params": {},
        "warnings": [],
    }

    base_params = base.get("params")
    cand_params = candidate.get("params")

    if not isinstance(base_params, dict):
        overlay["warnings"].append("base YAML missing params")
        return overlay

    if not isinstance(cand_params, dict):
        overlay["warnings"].append("candidate YAML missing params")
        cand_params = {}

    # root shape_vars
    overlay["shape_vars"] = normalize_shape_vars_dict(candidate.get("shape_vars"))

    # root rank_hints.rank_candidates
    cand_rank_hints = candidate.get("rank_hints")
    if isinstance(cand_rank_hints, dict):
        cand_rank_candidates = normalize_rank_candidates(cand_rank_hints.get("rank_candidates"))
        if cand_rank_candidates:
            overlay["rank_hints"]["rank_candidates"] = cand_rank_candidates

    # infer primary param:
    # 1) explicit candidate rank_assignment.primary_param
    # 2) param with shape_spec_by_rank
    # 3) param with multi-rank / variadic shape_spec signal
    # 4) fallback to base inference
    rank_assignment = candidate.get("rank_assignment")
    if isinstance(rank_assignment, dict):
        pp = rank_assignment.get("primary_param")
        if isinstance(pp, str) and pp in base_params:
            overlay["primary_param"] = pp

    if overlay["primary_param"] is None:
        for pname, cp in cand_params.items():
            if pname not in base_params or not isinstance(cp, dict):
                continue
            sbr = normalize_shape_spec_by_rank(cp.get("shape_spec_by_rank"))
            if sbr:
                overlay["primary_param"] = pname
                break

    if overlay["primary_param"] is None:
        for pname, cp in cand_params.items():
            if pname not in base_params or not isinstance(cp, dict):
                continue
            raw_ss = cp.get("shape_spec")
            explicit_ranks = collect_explicit_rank_candidates_from_shape_spec(raw_ss)
            if len(explicit_ranks) >= 2 or contains_variadic_shape_candidate(raw_ss):
                overlay["primary_param"] = pname
                break

    if overlay["primary_param"] is None:
        overlay["primary_param"] = infer_primary_param_from_base(base)

    primary_param = overlay["primary_param"]
    effective_ranks = infer_effective_ranks(base, candidate, primary_param)
    if effective_ranks:
        overlay["rank_hints"]["rank_candidates"] = effective_ranks

    # extract per-param shape fields
    for pname, bp in base_params.items():
        if not isinstance(bp, dict):
            continue

        cp = cand_params.get(pname)
        if not isinstance(cp, dict):
            continue

        upd: Dict[str, Any] = {}

        # explicit shape_spec_by_rank from candidate
        explicit_sbr = normalize_shape_spec_by_rank(cp.get("shape_spec_by_rank"))

        # shape_spec -> normalize into rank variants
        raw_ss = cp.get("shape_spec")
        derived_sbr = normalize_shape_spec_variants(raw_ss, effective_ranks)

        merged_sbr: Dict[str, List[str]] = {}
        if explicit_sbr:
            merged_sbr.update(explicit_sbr)

        # 只有当 raw_ss 真的是多 rank 结果时，才合并成 shape_spec_by_rank
        if len(derived_sbr) >= 2:
            merged_sbr.update(derived_sbr)

        # 优先保留 shape_spec_by_rank
        if merged_sbr:
            merged_sbr = dict(sorted(merged_sbr.items(), key=lambda kv: int(kv[0])))
            upd["shape_spec_by_rank"] = merged_sbr

            try:
                min_rank = min(int(k) for k in merged_sbr.keys())
                upd["shape_spec"] = list(merged_sbr[str(min_rank)])
            except Exception:
                pass
        else:
            # 如果只有一个 finite spec，就保留成 shape_spec
            single_ss = normalize_shape_spec_list(raw_ss)
            if single_ss:
                upd["shape_spec"] = single_ss
            elif len(derived_sbr) == 1:
                only_rank = next(iter(sorted(derived_sbr.keys(), key=int)))
                upd["shape_spec"] = list(derived_sbr[only_rank])
            elif contains_variadic_shape_candidate(raw_ss) and not effective_ranks:
                overlay["warnings"].append(
                    f"param {pname} has variadic shape_spec but no finite rank_candidates were provided"
                )

        if upd:
            overlay["params"][pname] = upd

    # 如果 primary_param 只有单一 shape_spec 且 effective_ranks 恰好只有一个 rank，
    # 升级成 shape_spec_by_rank，保证一致性
    if (
        isinstance(primary_param, str)
        and primary_param in overlay["params"]
        and "shape_spec_by_rank" not in overlay["params"][primary_param]
        and "shape_spec" in overlay["params"][primary_param]
        and len(effective_ranks) == 1
    ):
        ss = overlay["params"][primary_param]["shape_spec"]
        if isinstance(ss, list) and len(ss) == effective_ranks[0]:
            overlay["params"][primary_param]["shape_spec_by_rank"] = {
                str(effective_ranks[0]): list(ss)
            }

    # auto-fill missing shape_vars from used vars
    used_vars = collect_vars_from_specs(overlay["params"])
    for var in sorted(used_vars):
        if var not in overlay["shape_vars"]:
            overlay["shape_vars"][var] = default_range_for_dim(var)

    return overlay


def validate_stagec_overlay(overlay: Dict[str, Any], base: Dict[str, Any]) -> List[str]:
    errs: List[str] = []

    base_params = base.get("params")
    if not isinstance(base_params, dict):
        errs.append("base YAML missing params dict")
        return errs

    primary_param = overlay.get("primary_param")
    if primary_param is not None:
        if not isinstance(primary_param, str) or primary_param not in base_params:
            errs.append(f"overlay.primary_param invalid: {primary_param!r}")

    # rank_hints.rank_candidates validation
    overlay_rank_hints = overlay.get("rank_hints") or {}
    if not isinstance(overlay_rank_hints, dict):
        errs.append("overlay.rank_hints must be dict")
        overlay_rank_hints = {}

    expected_ranks = normalize_rank_candidates(overlay_rank_hints.get("rank_candidates"))
    if not expected_ranks:
        expected_ranks = get_expected_ranks(base)

    # shape_vars validation
    for k, v in (overlay.get("shape_vars") or {}).items():
        if not isinstance(k, str) or not k:
            errs.append(f"shape_vars key invalid: {k!r}")
            continue
        if not (isinstance(v, list) and len(v) == 2 and all(isinstance(x, int) for x in v)):
            errs.append(f"shape_vars[{k}] must be [lo,hi] ints, got {v!r}")
            continue
        lo, hi = v
        if lo < 1 or hi < lo:
            errs.append(f"shape_vars[{k}] invalid range: {v!r}")

    allowed_vars = set((overlay.get("shape_vars") or {}).keys()) | set((base.get("shape_vars") or {}).keys())

    params_overlay = overlay.get("params") or {}
    if not isinstance(params_overlay, dict):
        errs.append("overlay.params must be dict")
        return errs

    for pname, upd in params_overlay.items():
        if pname not in base_params:
            errs.append(f"overlay.params has unknown param: {pname}")
            continue
        if not isinstance(upd, dict):
            errs.append(f"overlay.params[{pname}] must be dict")
            continue

        ss = upd.get("shape_spec")
        if ss is not None:
            if not (isinstance(ss, list) and all(isinstance(x, str) for x in ss)):
                errs.append(f"overlay.params[{pname}].shape_spec must be list[str]")
            else:
                for x in ss:
                    if not is_plain_var_name(x):
                        errs.append(f"overlay.params[{pname}].shape_spec has invalid token: {x!r}")
                    elif x not in allowed_vars:
                        errs.append(f"overlay.params[{pname}].shape_spec references missing var: {x!r}")

        sbr = upd.get("shape_spec_by_rank")
        if sbr is not None:
            if not isinstance(sbr, dict):
                errs.append(f"overlay.params[{pname}].shape_spec_by_rank must be dict")
            else:
                for rk, spec in sbr.items():
                    try:
                        r = int(rk)
                    except Exception:
                        errs.append(f"overlay.params[{pname}].shape_spec_by_rank has non-int key: {rk!r}")
                        continue

                    if not (isinstance(spec, list) and all(isinstance(x, str) for x in spec)):
                        errs.append(f"overlay.params[{pname}].shape_spec_by_rank[{rk}] must be list[str]")
                        continue

                    if len(spec) != r:
                        errs.append(
                            f"overlay.params[{pname}].shape_spec_by_rank[{rk}] length={len(spec)} != rank={r}"
                        )

                    for x in spec:
                        if not is_plain_var_name(x):
                            errs.append(
                                f"overlay.params[{pname}].shape_spec_by_rank[{rk}] invalid token: {x!r}"
                            )
                        elif x not in allowed_vars:
                            errs.append(
                                f"overlay.params[{pname}].shape_spec_by_rank[{rk}] references missing var: {x!r}"
                            )

    # primary_param 与 expected_ranks 的一致性校验
    if expected_ranks and isinstance(primary_param, str):
        upd = params_overlay.get(primary_param) or {}
        got_sbr = upd.get("shape_spec_by_rank")
        got_ss = upd.get("shape_spec")

        if len(expected_ranks) >= 2:
            if not isinstance(got_sbr, dict):
                errs.append(
                    f"primary_param={primary_param} must have shape_spec_by_rank because expected ranks are {expected_ranks}"
                )
            else:
                got_ranks = sorted(int(k) for k in got_sbr.keys())
                missing = [r for r in expected_ranks if r not in got_ranks]
                if missing:
                    errs.append(
                        f"missing shape_spec_by_rank entries for primary_param={primary_param}: {missing}"
                    )
        else:
            only_rank = expected_ranks[0]
            ok = False

            if isinstance(got_sbr, dict) and str(only_rank) in got_sbr:
                ok = True
            elif isinstance(got_ss, list) and len(got_ss) == only_rank:
                ok = True

            if not ok:
                errs.append(
                    f"primary_param={primary_param} must provide shape_spec or shape_spec_by_rank for rank {only_rank}"
                )

    return errs

def merge_stagec_overlay(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge extracted Stage-C overlay back into original YAML.
    """
    out = copy.deepcopy(base)

    # merge rank_hints.rank_candidates
    out_rank_hints = out.get("rank_hints")
    if not isinstance(out_rank_hints, dict):
        out_rank_hints = {}

    overlay_rank_hints = overlay.get("rank_hints") or {}
    if isinstance(overlay_rank_hints, dict):
        rank_candidates = normalize_rank_candidates(overlay_rank_hints.get("rank_candidates"))
        if rank_candidates:
            out_rank_hints["rank_candidates"] = rank_candidates

    out["rank_hints"] = out_rank_hints

    # merge shape_vars
    out_shape_vars = out.get("shape_vars")
    if not isinstance(out_shape_vars, dict):
        out_shape_vars = {}
    out_shape_vars.update(overlay.get("shape_vars") or {})
    out["shape_vars"] = out_shape_vars

    primary_param = overlay.get("primary_param")
    params = out.get("params")
    if not isinstance(params, dict):
        raise RuntimeError("base YAML missing params dict")

    params_overlay = overlay.get("params") or {}

    for pname, upd in params_overlay.items():
        if pname not in params:
            continue

        p = params.get(pname)
        if not isinstance(p, dict) or not isinstance(upd, dict):
            continue

        sbr = upd.get("shape_spec_by_rank")
        if isinstance(sbr, dict) and sbr:
            sbr = dict(sorted(sbr.items(), key=lambda kv: int(kv[0])))
            p["shape_spec_by_rank"] = sbr

            try:
                min_rank = min(int(k) for k in sbr.keys())
                p["shape_spec"] = list(sbr[str(min_rank)])
            except Exception:
                pass
            continue

        ss = upd.get("shape_spec")
        if isinstance(ss, list) and ss:
            base_ss = p.get("shape_spec")
            has_todo = isinstance(base_ss, list) and any(x == "TODO_SHAPE" for x in base_ss)

            # primary param 一定更新；非 primary 保守更新 TODO / 缺失场景
            if (
                pname == primary_param
                or has_todo
                or "shape_spec" not in p
                or not isinstance(base_ss, list)
            ):
                p["shape_spec"] = list(ss)

    return out


# ----------------------------
# 5) LLM call
# ----------------------------
def call_llm_for_candidate_yaml(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


# ----------------------------
# 6) main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc_txt", required=True)
    ap.add_argument("--yaml_in", required=True)
    ap.add_argument("--yaml_out_dir", required=True)

    ap.add_argument("--model", default="gpt-5-codex")
    ap.add_argument("--max_doc_chars", type=int, default=80000)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=4000)
    ap.add_argument("--max_retries", type=int, default=2)
    ap.add_argument(
        "--fail_on_invalid",
        action="store_true",
        help="if set, stop when extraction/validation fails; otherwise write best-effort with warnings",
    )
    ap.add_argument(
        "--keep_variant_constraints_by_rank",
        action="store_true",
        help="kept for compatibility; Stage C does not write constraints",
    )

    args = ap.parse_args()

    doc_path = Path(args.doc_txt).resolve()
    yaml_in_path = Path(args.yaml_in).resolve()
    out_dir = Path(args.yaml_out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    doc_text = safe_truncate(read_text(doc_path), args.max_doc_chars)
    yaml_obj = load_yaml_obj(yaml_in_path)
    yaml_text = read_text(yaml_in_path)

    if not isinstance(yaml_obj, dict):
        raise RuntimeError(f"Input YAML must be a mapping/dict: {yaml_in_path}")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.gpt.ge/v1/")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY env var")

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers={"x-foo": "true"},
    )

    system_prompt = YAML_PATCH_SYSTEM_PROMPT

    base_user_prompt = (
        "=== OFFICIAL DOCUMENTATION (TXT) ===\n"
        f"{doc_text}\n\n"
        "=== CURRENT YAML SKELETON ===\n"
        f"{yaml_text}\n\n"
        "Output ONE COMPLETE YAML document only.\n"
        "Do not output JSON.\n"
        "Do not output markdown fences unless necessary.\n"
        "Do not explain anything.\n"
        "Stage C only:\n"
        "- Fill/complete shape_vars\n"
        "- Fill/complete params.*.shape_spec or params.*.shape_spec_by_rank\n"
        "- Fill/complete rank_hints.rank_candidates when needed\n"
        "- Do NOT add semantic constraints\n"
        "- Leave constraints empty or unchanged\n"
    )

    last_errors: List[str] = []
    last_raw: str = ""
    last_yaml_block: str = ""
    last_overlay: Optional[Dict[str, Any]] = None
    final_merged: Optional[Dict[str, Any]] = None
    final_primary_param: Optional[str] = None
    final_warnings: List[str] = []
    user_prompt = base_user_prompt

    for attempt in range(args.max_retries + 1):
        try:
            last_raw = call_llm_for_candidate_yaml(
                client=client,
                model=args.model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        except BadRequestError:
            print("=== BadRequestError ===")
            traceback.print_exc()
            raise

        try:
            last_yaml_block = extract_yaml_block(last_raw)
            candidate_yaml = parse_candidate_yaml(last_raw)
        except Exception as e:
            last_errors = [f"candidate YAML parse failed: {e}"]
            if attempt < args.max_retries:
                user_prompt = (
                    base_user_prompt
                    + "\n\nYour previous output was not valid YAML.\n"
                      "Output ONE YAML document only. No explanations.\n"
                )
                continue
            else:
                break

        overlay = extract_stagec_fields(candidate_yaml, yaml_obj)
        last_overlay = overlay
        last_errors = validate_stagec_overlay(overlay, yaml_obj)

        if not last_errors:
            final_merged = merge_stagec_overlay(yaml_obj, overlay)
            final_primary_param = overlay.get("primary_param")
            final_warnings = list(overlay.get("warnings") or [])
            break

        if attempt < args.max_retries:
            user_prompt = (
                base_user_prompt
                + "\n\nYour previous YAML was parsed, but the extracted Stage-C fields were invalid.\n"
                  "Fix the YAML and output ONE YAML document only.\n"
                  "Validation errors:\n"
                + "\n".join(f"- {e}" for e in last_errors)
                + "\n\nPrevious extracted overlay:\n"
                + yaml.safe_dump(last_overlay, sort_keys=False, allow_unicode=True)
            )
        else:
            # best-effort merge if possible
            try:
                final_merged = merge_stagec_overlay(yaml_obj, overlay)
                final_primary_param = overlay.get("primary_param")
                final_warnings = list(overlay.get("warnings") or [])
            except Exception:
                final_merged = copy.deepcopy(yaml_obj)
                final_primary_param = None
                final_warnings = list(overlay.get("warnings") or [])

    if final_merged is None:
        final_merged = copy.deepcopy(yaml_obj)
        final_primary_param = infer_primary_param_from_base(yaml_obj)

    api_name = yaml_obj.get("api_name", "unknown.api")
    aten = (yaml_obj.get("aten") or {}) if isinstance(yaml_obj.get("aten"), dict) else {}
    overload = aten.get("overload", "default")
    primary_param_for_name = final_primary_param or "unknown"

    # summary
    params = final_merged.get("params") or {}
    summary_shape_spec_by_rank_keys: List[str] = []
    if isinstance(params, dict) and isinstance(final_primary_param, str) and final_primary_param in params:
        p = params.get(final_primary_param)
        if isinstance(p, dict) and isinstance(p.get("shape_spec_by_rank"), dict):
            summary_shape_spec_by_rank_keys = sorted(
                list(p["shape_spec_by_rank"].keys()),
                key=lambda x: int(x),
            )

    summary = {
        "primary_param": final_primary_param,
        "shape_spec_by_rank_keys": summary_shape_spec_by_rank_keys,
        "shape_vars_keys": sorted(list((final_merged.get("shape_vars") or {}).keys()))
        if isinstance(final_merged.get("shape_vars"), dict) else [],
    }

    out_name = f"{safe_name(api_name)}__ov_{safe_name(overload)}__{safe_name(primary_param_for_name)}__MULTIRANK.yaml"
    out_path = out_dir / out_name
    out_path.write_text(dump_yaml_obj(final_merged), encoding="utf-8")

    meta = {
        "model": args.model,
        "doc_txt": str(doc_path),
        "yaml_in": str(yaml_in_path),
        "yaml_out": str(out_path),
        "primary_param": final_primary_param,
        "summary": summary,
        "warnings": final_warnings,
        "validation_errors": last_errors,
        "extracted_overlay": last_overlay,
        "raw_model_output_snippet": (last_raw[:8000] if last_raw else ""),
        "raw_yaml_block_snippet": (last_yaml_block[:8000] if last_yaml_block else ""),
    }
    (out_path.with_suffix(out_path.suffix + ".meta.json")).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[+] wrote Stage-C merged yaml: {out_path}")
    if last_errors:
        msg = "[!] Stage-C extraction/validation errors (not fully fixed):\n" + "\n".join(f"   - {e}" for e in last_errors)
        if args.fail_on_invalid:
            raise SystemExit(msg)
        else:
            print(msg)


if __name__ == "__main__":
    main()