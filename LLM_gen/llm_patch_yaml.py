#!/usr/bin/env python3
# llm_patch_yaml.py
import os
import re
import argparse
import json
import traceback
import ast
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
    """
    Convert arbitrary string to a filesystem-safe name.
    """
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


# ----------------------------
# 2) JSON patch parsing
# ----------------------------
def extract_json_object(text: str) -> str:
    text = (text or "").strip()
    if not text:
        raise ValueError("empty model output")

    if text.startswith("{") and text.endswith("}"):
        return text

    l = text.find("{")
    r = text.rfind("}")
    if l != -1 and r != -1 and r > l:
        return text[l: r + 1]

    raise ValueError("cannot find JSON object in model output")


def parse_patch(raw_text: str) -> Dict[str, Any]:
    js = extract_json_object(raw_text)
    return json.loads(js)


def normalize_multi_patch(patch_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stage-C multi-variant patch normalization.
    Expected keys:
      - rank_assignment: {primary_param, confidence, notes}
      - variants: [{rank, shape_vars, shape_spec_fixes, constraints}, ...]
      - shared_constraints: [...]
      - changes, warnings
    """
    out: Dict[str, Any] = {}
    out["rank_assignment"] = patch_dict.get("rank_assignment") or {}
    out["variants"] = patch_dict.get("variants") or []
    out["shared_constraints"] = patch_dict.get("shared_constraints") or []
    out["changes"] = patch_dict.get("changes") or []
    out["warnings"] = patch_dict.get("warnings") or []

    if not isinstance(out["rank_assignment"], dict):
        out["rank_assignment"] = {}
    if not isinstance(out["variants"], list):
        out["variants"] = []
    if not isinstance(out["shared_constraints"], list):
        out["shared_constraints"] = []
    if not isinstance(out["changes"], list):
        out["changes"] = []
    if not isinstance(out["warnings"], list):
        out["warnings"] = []

    def _clean_constraints(lst: Any) -> List[str]:
        cleaned: List[str] = []
        if not isinstance(lst, list):
            return cleaned
        for item in lst:
            if isinstance(item, str) and item.strip():
                cleaned.append(item.strip())
            elif isinstance(item, dict):
                for k in ("expr", "constraint"):
                    v = item.get(k)
                    if isinstance(v, str) and v.strip():
                        cleaned.append(v.strip())
                        break
        return cleaned

    out["shared_constraints"] = _clean_constraints(out["shared_constraints"])

    cleaned_variants: List[Dict[str, Any]] = []
    for v in out["variants"]:
        if not isinstance(v, dict):
            continue
        vv = dict(v)

        r = vv.get("rank", None)
        if not (r is None or isinstance(r, int)):
            vv["rank"] = None

        if not isinstance(vv.get("shape_vars"), dict):
            vv["shape_vars"] = {}

        if not isinstance(vv.get("shape_spec_fixes"), dict):
            vv["shape_spec_fixes"] = {}

        vv["constraints"] = _clean_constraints(vv.get("constraints") or [])
        cleaned_variants.append(vv)

    out["variants"] = cleaned_variants
    return out


# ----------------------------
# 3) validation utilities
# ----------------------------
_ALLOWED_BUILTINS = {
    "isinstance", "all", "any", "len", "tuple",
    "min", "max", "abs",
}


def validate_shape_vars(shape_vars: Dict[str, Any]) -> List[str]:
    errs = []
    for k, v in shape_vars.items():
        if not isinstance(k, str) or not k:
            errs.append(f"shape_vars key invalid: {k!r}")
            continue
        if not (isinstance(v, (list, tuple)) and len(v) == 2):
            errs.append(f"shape_vars[{k}] must be [lo,hi], got: {v!r}")
            continue
        lo, hi = v[0], v[1]
        if not (isinstance(lo, int) and isinstance(hi, int)):
            errs.append(f"shape_vars[{k}] lo/hi must be int, got: {v!r}")
            continue
        if lo < 1 or hi < 1 or lo > hi:
            errs.append(f"shape_vars[{k}] must satisfy 1<=lo<=hi, got: {v!r}")
    return errs


def validate_no_expressions_in_shape_spec(shape_spec: List[str]) -> Optional[str]:
    for item in shape_spec:
        if not isinstance(item, str):
            return f"shape_spec item must be str, got {item!r}"
        if any(ch in item for ch in ("+", "-", "*", "/", "%", "(", ")", " ", "[", "]", ".")):
            return f"shape_spec contains expression-like token: {item!r}"
    return None


def validate_constraints_eval_safety(constraints: List[Any]) -> List[str]:
    errs = []
    for i, c in enumerate(constraints):
        if not isinstance(c, str):
            errs.append(f"constraints[{i}] must be a string")
            continue
        if ";" in c or "\n" in c:
            errs.append(f"constraints[{i}] contains ';' or newline, not allowed: {c!r}")
        if "import " in c or "lambda" in c or "def " in c or "class " in c:
            errs.append(f"constraints[{i}] contains forbidden keyword: {c!r}")
    return errs


def extract_names_ast(expr: str) -> Set[str]:
    """
    AST-based name extraction:
    - Collect only ast.Name identifiers.
    - DO NOT treat attribute names (ndim/shape) as free variables.
    """
    try:
        node = ast.parse(expr, mode="eval")
    except Exception:
        return set()

    names: Set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            names.add(n.id)

    keywords = {"None", "True", "False"}
    return {x for x in names if x not in keywords}


def validate_constraint_names_defined(constraints: List[Any], allowed_names: Set[str]) -> List[str]:
    errs = []
    for i, c in enumerate(constraints):
        if not isinstance(c, str):
            errs.append(f"constraints[{i}] must be a string, got {type(c).__name__}: {c!r}")
            continue

        names = extract_names_ast(c)
        names = {n for n in names if n not in _ALLOWED_BUILTINS}
        unknown = sorted([n for n in names if n not in allowed_names])
        if unknown:
            errs.append(f"constraints[{i}] references undefined names: {unknown} | expr={c!r}")
    return errs


def normalize_shape_vars_for_write(shape_vars: Dict[str, Any]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for k, v in shape_vars.items():
        if isinstance(v, (list, tuple)) and len(v) == 2:
            out[str(k)] = [int(v[0]), int(v[1])]
    return out


def validate_multi_patch(
    patch: Dict[str, Any],
    base_yaml: Dict[str, Any],
    base_allowed_names: Set[str],
) -> List[str]:
    """
    Validate Stage-C patch structure + per-variant correctness.
    NOTE: This validator is still "variants-shaped".
    """
    errs: List[str] = []

    params = base_yaml.get("params") if isinstance(base_yaml, dict) else None
    if not isinstance(params, dict):
        errs.append("base YAML missing params dict")
        return errs

    base_shape_vars = base_yaml.get("shape_vars")
    if not isinstance(base_shape_vars, dict):
        base_shape_vars = {}

    rank_assignment = patch.get("rank_assignment") or {}
    if not isinstance(rank_assignment, dict):
        errs.append("rank_assignment must be an object")
        rank_assignment = {}

    primary_param = rank_assignment.get("primary_param", None)
    if primary_param is not None and not isinstance(primary_param, str):
        errs.append("rank_assignment.primary_param must be string or null")
        primary_param = None

    variants = patch.get("variants") or []
    if not isinstance(variants, list) or not variants:
        errs.append("variants must be a non-empty list")
        return errs

    shared_constraints = patch.get("shared_constraints") or []
    if not isinstance(shared_constraints, list):
        errs.append("shared_constraints must be a list[str]")
        shared_constraints = []

    errs += validate_constraints_eval_safety(shared_constraints)

    shared_allowed = set(base_allowed_names) | set(base_shape_vars.keys())
    errs += validate_constraint_names_defined(shared_constraints, shared_allowed)

    def _param_has_todo_shape(pname: str) -> bool:
        p = params.get(pname)
        if not isinstance(p, dict):
            return False
        ss = p.get("shape_spec")
        return isinstance(ss, list) and any(x == "TODO_SHAPE" for x in ss)

    for i, v in enumerate(variants):
        if not isinstance(v, dict):
            errs.append(f"variants[{i}] must be object")
            continue

        rank = v.get("rank", None)
        if not (rank is None or isinstance(rank, int)):
            errs.append(f"variants[{i}].rank must be int or null")

        sv = v.get("shape_vars") or {}
        if not isinstance(sv, dict):
            errs.append(f"variants[{i}].shape_vars must be object")
            sv = {}

        errs += [f"variants[{i}].{e}" for e in validate_shape_vars(sv)]

        v_constraints = v.get("constraints") or []
        if not isinstance(v_constraints, list):
            errs.append(f"variants[{i}].constraints must be list[str]")
            v_constraints = []

        combined_constraints = list(shared_constraints) + list(v_constraints)
        errs += [f"variants[{i}].{e}" for e in validate_constraints_eval_safety(combined_constraints)]

        allowed_now = set(base_allowed_names) | set(base_shape_vars.keys()) | set(sv.keys())
        errs += [f"variants[{i}].{e}" for e in validate_constraint_names_defined(combined_constraints, allowed_now)]

        fixes = v.get("shape_spec_fixes") or {}
        if not isinstance(fixes, dict):
            errs.append(f"variants[{i}].shape_spec_fixes must be object")
            fixes = {}

        for pname, spec_list in fixes.items():
            if pname not in params:
                errs.append(f"variants[{i}].shape_spec_fixes has unknown param: {pname}")
                continue

            # We still allow providing fixes even when base yaml has no TODO_SHAPE,
            # because Stage-C output can be used to build shape_spec_by_rank later.
            if not isinstance(spec_list, list) or not all(isinstance(x, str) for x in spec_list):
                errs.append(f"variants[{i}].shape_spec_fixes[{pname}] must be list[str]")
                continue

            err = validate_no_expressions_in_shape_spec(spec_list)
            if err:
                errs.append(f"variants[{i}].shape_spec_fixes[{pname}] invalid: {err}")
                continue

            missing = [x for x in spec_list if x not in sv and x not in base_shape_vars]
            if missing:
                errs.append(f"variants[{i}].shape_spec_fixes[{pname}] references vars not in shape_vars: {missing}")

            if isinstance(rank, int) and primary_param and pname == primary_param:
                if len(spec_list) != rank:
                    errs.append(
                        f"variants[{i}] primary_param={primary_param} rank={rank} "
                        f"but shape_spec_fixes[{pname}] length={len(spec_list)}"
                    )

    return errs


# ----------------------------
# 4) build ONE yaml from variants
# ----------------------------
def build_single_yaml_from_stagec(
    base_yaml: Dict[str, Any],
    stagec_patch: Dict[str, Any],
    keep_variant_constraints_by_rank: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Convert Stage-C multi-variant result into ONE YAML object.

    Writes:
      - shape_vars: union over variants (+ keep existing)
      - constraints: keep minimal -> shared_constraints only (dedup)
      - params[primary_param].shape_spec_by_rank: { "2":[...], "3":[...], ... }
      - optionally params[primary_param].constraints_by_rank: { "2":[...], ... } (if keep_variant_constraints_by_rank)
      - optionally set params[primary_param].shape_spec as a fallback (min rank) for compatibility
      - For non-primary params: apply shape_spec from first available variant as a single shape_spec (optional)

    Returns: (merged_yaml, summary_meta)
    """
    out = dict(base_yaml)

    params = out.get("params")
    if not isinstance(params, dict):
        raise RuntimeError("base YAML missing params dict")

    rank_assignment = stagec_patch.get("rank_assignment") or {}
    primary_param = rank_assignment.get("primary_param") or None

    variants = stagec_patch.get("variants") or []
    shared_constraints = stagec_patch.get("shared_constraints") or []
    if not isinstance(shared_constraints, list):
        shared_constraints = []

    # 1) union shape_vars across variants
    union_sv: Dict[str, Any] = {}
    for v in variants:
        if isinstance(v, dict) and isinstance(v.get("shape_vars"), dict):
            union_sv.update(v.get("shape_vars") or {})

    # merge into existing shape_vars
    old_sv = out.get("shape_vars")
    if not isinstance(old_sv, dict):
        old_sv = {}
    merged_sv = dict(old_sv)
    merged_sv.update(normalize_shape_vars_for_write(union_sv))
    out["shape_vars"] = merged_sv

    # 2) constraints: keep minimal -> shared only (dedup)
    def _dedup_keep_order(lst: List[str]) -> List[str]:
        seen = set()
        res = []
        for x in lst:
            if not isinstance(x, str):
                continue
            s = x.strip()
            if not s or s in seen:
                continue
            seen.add(s)
            res.append(s)
        return res

    out["constraints"] = _dedup_keep_order(list(out.get("constraints") or []) + list(shared_constraints))

    # 3) primary_param shape_spec_by_rank
    shape_spec_by_rank: Dict[str, List[str]] = {}
    constraints_by_rank: Dict[str, List[str]] = {}

    if primary_param and primary_param in params:
        for v in variants:
            if not isinstance(v, dict):
                continue
            r = v.get("rank", None)
            if not isinstance(r, int):
                continue
            fixes = v.get("shape_spec_fixes") or {}
            if not isinstance(fixes, dict):
                continue
            spec = fixes.get(primary_param)
            if isinstance(spec, list) and all(isinstance(x, str) for x in spec):
                shape_spec_by_rank[str(r)] = list(spec)

                if keep_variant_constraints_by_rank:
                    vcs = v.get("constraints") or []
                    if isinstance(vcs, list):
                        constraints_by_rank[str(r)] = _dedup_keep_order([c for c in vcs if isinstance(c, str)])

        # write shape_spec_by_rank
        p = params.get(primary_param)
        if isinstance(p, dict) and shape_spec_by_rank:
            p["shape_spec_by_rank"] = shape_spec_by_rank

            # optional: store per-rank constraints (so runtime can add rank-specific checks)
            if keep_variant_constraints_by_rank and constraints_by_rank:
                p["constraints_by_rank"] = constraints_by_rank

            # fallback: choose smallest rank spec as params[primary].shape_spec (compat)
            try:
                min_rank = min(int(k) for k in shape_spec_by_rank.keys())
                p["shape_spec"] = list(shape_spec_by_rank[str(min_rank)])
            except Exception:
                pass

    # 4) non-primary param shape_spec: pick first available spec from variants (rank-independent usually)
    # (Optional, keep conservative: only set if base param shape_spec contains TODO_SHAPE)
    for v in variants:
        if not isinstance(v, dict):
            continue
        fixes = v.get("shape_spec_fixes") or {}
        if not isinstance(fixes, dict):
            continue
        for pname, spec in fixes.items():
            if pname == primary_param:
                continue
            if pname not in params:
                continue
            p = params.get(pname)
            if not isinstance(p, dict):
                continue
            base_ss = p.get("shape_spec")
            has_todo = isinstance(base_ss, list) and any(x == "TODO_SHAPE" for x in base_ss)
            if not has_todo:
                continue
            if isinstance(spec, list) and all(isinstance(x, str) for x in spec):
                p["shape_spec"] = list(spec)
        # continue scanning all variants; cheap enough

    summary = {
        "primary_param": primary_param,
        "shape_spec_by_rank_keys": sorted(shape_spec_by_rank.keys(), key=lambda x: int(x)) if shape_spec_by_rank else [],
        "union_shape_vars_keys": sorted(list(union_sv.keys())),
        "kept_shared_constraints": len(shared_constraints),
        "kept_variant_constraints_by_rank": keep_variant_constraints_by_rank,
    }
    return out, summary


# ----------------------------
# 5) LLM call
# ----------------------------
def call_llm_for_patch(
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

    ap.add_argument("--model", default="gpt-4o-2024-08-06")
    ap.add_argument("--max_doc_chars", type=int, default=80000)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_tokens", type=int, default=2000)
    ap.add_argument("--max_retries", type=int, default=2)
    ap.add_argument(
        "--fail_on_invalid",
        action="store_true",
        help="if set, stop when patch validation fails; otherwise write best-effort with warnings",
    )
    ap.add_argument(
        "--keep_variant_constraints_by_rank",
        action="store_true",
        help="store per-rank constraints under params[primary].constraints_by_rank (default: off)",
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

    # Allowed names for constraints name checking:
    allowed_names = set(_ALLOWED_BUILTINS)
    params = yaml_obj.get("params")
    if isinstance(params, dict):
        allowed_names |= set(params.keys())
    allowed_names |= {"padding_tuple", "stride_tuple", "dilation_tuple", "kernel_size_tuple"}

    system_prompt = YAML_PATCH_SYSTEM_PROMPT

    base_user_prompt = (
        "=== OFFICIAL DOCUMENTATION (TXT) ===\n"
        f"{doc_text}\n\n"
        "=== YAML SKELETON (INPUT YAML; DO NOT REWRITE IT) ===\n"
        f"{yaml_text}\n\n"
        "Return ONLY JSON (no YAML, no markdown, no extra text).\n"
        "IMPORTANT: Stage C. Keep constraints minimal and only high-confidence.\n"
    )

    last_errors: List[str] = []
    last_raw: str = ""
    final_patch: Optional[Dict[str, Any]] = None
    user_prompt = base_user_prompt

    # -----------------------
    # retry loop
    # -----------------------
    for attempt in range(args.max_retries + 1):
        try:
            last_raw = call_llm_for_patch(
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

        # (1) parse json
        try:
            patch_dict = parse_patch(last_raw)
        except Exception as e:
            last_errors = [f"JSON parse failed: {e}"]
            if attempt < args.max_retries:
                user_prompt = (
                    base_user_prompt
                    + "\n\nYour previous output was NOT valid JSON.\n"
                      "Please output ONLY a valid JSON object.\n"
                )
                continue

            final_patch = {
                "rank_assignment": {"primary_param": None, "confidence": 0.0, "notes": ["json_parse_failed"]},
                "variants": [{"rank": None, "shape_vars": {}, "shape_spec_fixes": {}, "constraints": []}],
                "shared_constraints": [],
                "changes": [],
                "warnings": [f"Failed to parse JSON patch: {e}"],
            }
            break

        # (2) normalize stage-c patch
        patch = normalize_multi_patch(patch_dict)

        # (3) validate stage-c patch
        last_errors = validate_multi_patch(patch, yaml_obj, allowed_names)

        if not last_errors:
            final_patch = patch
            break

        # retry with feedback
        if attempt < args.max_retries:
            user_prompt = (
                base_user_prompt
                + "\n\nYour previous JSON FAILED validation.\n"
                  "Fix it and output ONLY JSON again.\n"
                  "Validation errors:\n"
                + "\n".join(f"- {e}" for e in last_errors)
                + "\n\nPrevious JSON (for reference):\n"
                + (extract_json_object(last_raw) if last_raw else "")
                + "\n"
            )
        else:
            final_patch = patch  # keep last even if invalid

    if final_patch is None:
        raise RuntimeError("LLM call failed (no patch produced)")

    # -----------------------
    # build ONE yaml (multi-rank) and write
    # -----------------------
    api_name = yaml_obj.get("api_name", "unknown.api")
    aten = (yaml_obj.get("aten") or {}) if isinstance(yaml_obj.get("aten"), dict) else {}
    overload = aten.get("overload", "default")

    rank_assignment = final_patch.get("rank_assignment") or {}
    primary_param = rank_assignment.get("primary_param") or "unknown"

    merged_yaml, summary = build_single_yaml_from_stagec(
        base_yaml=yaml_obj,
        stagec_patch=final_patch,
        keep_variant_constraints_by_rank=bool(args.keep_variant_constraints_by_rank),
    )

    out_name = f"{safe_name(api_name)}__ov_{safe_name(overload)}__{safe_name(primary_param)}__MULTIRANK.yaml"
    out_path = out_dir / out_name
    out_path.write_text(dump_yaml_obj(merged_yaml), encoding="utf-8")

    meta = {
        "model": args.model,
        "doc_txt": str(doc_path),
        "yaml_in": str(yaml_in_path),
        "yaml_out": str(out_path),
        "rank_assignment": rank_assignment,
        "num_variants_in_json": len(final_patch.get("variants") or []),
        "summary": summary,
        "warnings": final_patch.get("warnings", []),
        "validation_errors": last_errors,
        "raw_model_output_snippet": (last_raw[:8000] if last_raw else ""),
    }
    (out_path.with_suffix(out_path.suffix + ".meta.json")).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[+] wrote single multi-rank yaml: {out_path}")
    if last_errors:
        msg = "[!] Stage-C patch validation errors (not fully fixed):\n" + "\n".join(f"   - {e}" for e in last_errors)
        if args.fail_on_invalid:
            raise SystemExit(msg)
        else:
            print(msg)


if __name__ == "__main__":
    main()
