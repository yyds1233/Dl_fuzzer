#!/usr/bin/env python3
# patch_constraints.py
import os
import argparse
import json
import traceback
import ast
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml
from openai import OpenAI, BadRequestError

from llm_prompts import YAML_CONSTRAINT_SYSTEM_PROMPT


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


# ----------------------------
# 2) JSON parsing
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
    return json.loads(extract_json_object(raw_text))


def normalize_constraint_list(lst: Any) -> List[str]:
    cleaned: List[str] = []
    if not isinstance(lst, list):
        return cleaned

    for item in lst:
        if isinstance(item, str) and item.strip():
            cleaned.append(item.strip())
        elif isinstance(item, dict):
            # tolerate {"expr": "..."} / {"constraint": "..."}
            for k in ("expr", "constraint"):
                v = item.get(k)
                if isinstance(v, str) and v.strip():
                    cleaned.append(v.strip())
                    break
    return cleaned


def normalize_constraint_patch(p: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["constraints_append"] = p.get("constraints_append") or []
    out["constraints_remove"] = p.get("constraints_remove") or []
    out["changes"] = p.get("changes") or []
    out["warnings"] = p.get("warnings") or []
    out["confidence"] = p.get("confidence", 0.5)

    if not isinstance(out["constraints_append"], list):
        out["constraints_append"] = []
    if not isinstance(out["constraints_remove"], list):
        out["constraints_remove"] = []
    if not isinstance(out["changes"], list):
        out["changes"] = []
    if not isinstance(out["warnings"], list):
        out["warnings"] = []

    try:
        out["confidence"] = float(out["confidence"])
    except Exception:
        out["confidence"] = 0.5

    out["constraints_append"] = normalize_constraint_list(out["constraints_append"])
    out["constraints_remove"] = normalize_constraint_list(out["constraints_remove"])
    return out


def try_convert_legacy_fullspec_to_constraint_patch(
    obj: Dict[str, Any],
    base_yaml: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Convert accidental full-spec JSON output into Stage-D patch format.

    Example bad output:
      {
        "api_name": "...",
        "constraints": ["input.ndim == 4", ...],
        ...
      }

    We convert by diffing against existing YAML constraints.
    """
    if not isinstance(obj, dict):
        return None

    if "constraints_append" in obj or "constraints_remove" in obj:
        return None

    if "constraints" not in obj:
        return None

    old_constraints: List[str] = []
    for x in (base_yaml.get("constraints") or []):
        if isinstance(x, str) and x.strip():
            old_constraints.append(x.strip())

    new_constraints = normalize_constraint_list(obj.get("constraints"))

    to_append = [c for c in new_constraints if c not in old_constraints]
    to_remove = [c for c in old_constraints if c not in new_constraints]

    return {
        "constraints_append": to_append,
        "constraints_remove": to_remove,
        "changes": ["converted legacy full-spec output into Stage-D patch"],
        "warnings": ["model returned full-spec JSON instead of Stage-D patch"],
        "confidence": 0.5,
    }


# ----------------------------
# 3) validation
# ----------------------------
_ALLOWED_BUILTINS = {"isinstance", "all", "any", "len", "tuple", "min", "max", "abs"}


def validate_constraints_eval_safety(constraints: List[Any]) -> List[str]:
    errs: List[str] = []
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
    try:
        node = ast.parse(expr, mode="eval")
    except Exception:
        return set()

    names: Set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            names.add(n.id)

    return {x for x in names if x not in {"None", "True", "False"}}


def validate_constraint_names_defined(constraints: List[str], allowed_names: Set[str]) -> List[str]:
    errs: List[str] = []
    for i, c in enumerate(constraints):
        names = extract_names_ast(c)
        names = {n for n in names if n not in _ALLOWED_BUILTINS}
        unknown = sorted([n for n in names if n not in allowed_names])
        if unknown:
            errs.append(f"constraints[{i}] references undefined names: {unknown} | expr={c!r}")
    return errs


def _build_allowed_names(yaml_obj: Dict[str, Any]) -> Set[str]:
    allowed = set(_ALLOWED_BUILTINS)

    params = yaml_obj.get("params")
    if isinstance(params, dict):
        allowed |= set(params.keys())

    shape_vars = yaml_obj.get("shape_vars")
    if isinstance(shape_vars, dict):
        allowed |= set(shape_vars.keys())

    allowed |= {"padding_tuple", "stride_tuple", "dilation_tuple", "kernel_size_tuple"}
    return allowed


def _is_constraint_valid(c: str, allowed_names: Set[str], check_names: bool = True) -> bool:
    if validate_constraints_eval_safety([c]):
        return False
    if check_names and validate_constraint_names_defined([c], allowed_names):
        return False
    return True


def _filter_valid_constraints(
    constraints: List[str],
    allowed_names: Set[str],
    check_names: bool = True,
) -> Tuple[List[str], List[str]]:
    kept: List[str] = []
    dropped: List[str] = []
    for c in constraints:
        if _is_constraint_valid(c, allowed_names, check_names=check_names):
            kept.append(c)
        else:
            dropped.append(c)
    return kept, dropped


def _infer_primary_param_and_ranks(yaml_obj: Dict[str, Any]) -> Tuple[Optional[str], List[int]]:
    """
    Prefer params[*].shape_spec_by_rank, then fall back to rank_hints.
    """
    params = yaml_obj.get("params") or {}
    if isinstance(params, dict):
        for pname, p in params.items():
            if not isinstance(p, dict):
                continue
            by_rank = p.get("shape_spec_by_rank")
            if isinstance(by_rank, dict) and by_rank:
                ranks: List[int] = []
                for k in by_rank.keys():
                    try:
                        ranks.append(int(k))
                    except Exception:
                        pass
                return pname, sorted(set(ranks))

    rank_hints = yaml_obj.get("rank_hints") or {}
    rank_candidates = rank_hints.get("rank_candidates") or []
    ranks: List[int] = []
    if isinstance(rank_candidates, list):
        for x in rank_candidates:
            try:
                ranks.append(int(x))
            except Exception:
                pass

    primary_param: Optional[str] = None
    if isinstance(params, dict):
        if "input" in params:
            primary_param = "input"
        elif "self" in params:
            primary_param = "self"
        else:
            for pname, p in params.items():
                if isinstance(p, dict) and str(p.get("kind", "")).startswith("tensor"):
                    primary_param = pname
                    break

    return primary_param, sorted(set(ranks))


def _build_rank_constraint(primary_param: Optional[str], ranks: List[int]) -> Optional[str]:
    if not primary_param or not ranks:
        return None

    ranks = sorted(set(int(r) for r in ranks))
    if len(ranks) == 1:
        return f"{primary_param}.ndim == {ranks[0]}"
    inner = ", ".join(str(r) for r in ranks)
    return f"{primary_param}.ndim in ({inner})"


def _describe_shape_context(yaml_obj: Dict[str, Any]) -> str:
    """
    Build a concise textual summary of shape-related fields for the prompt.
    """
    params = yaml_obj.get("params") or {}
    if not isinstance(params, dict):
        return ""

    lines: List[str] = []
    for pname, p in params.items():
        if not isinstance(p, dict):
            continue

        if "shape_spec_by_rank" in p and isinstance(p["shape_spec_by_rank"], dict):
            lines.append(
                f"{pname}.shape_spec_by_rank = "
                f"{json.dumps(p['shape_spec_by_rank'], ensure_ascii=False)}"
            )
        elif "shape_spec" in p:
            lines.append(
                f"{pname}.shape_spec = "
                f"{json.dumps(p['shape_spec'], ensure_ascii=False)}"
            )

    return "\n".join(lines)


# ----------------------------
# 4) apply patch
# ----------------------------
def merge_constraints(existing: List[Any], to_remove: List[str], to_append: List[str]) -> List[str]:
    old: List[str] = []
    for x in existing or []:
        if isinstance(x, str) and x.strip():
            old.append(x.strip())

    remove_set = set([s.strip() for s in to_remove if s.strip()])
    kept = [c for c in old if c not in remove_set]

    seen = set(kept)
    for c in to_append:
        c = c.strip()
        if c and c not in seen:
            kept.append(c)
            seen.add(c)
    return kept


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
    ap.add_argument("--yaml_out", required=True)

    ap.add_argument("--model", default="gpt-4o-2024-08-06")
    ap.add_argument("--max_doc_chars", type=int, default=80000)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_tokens", type=int, default=1200)
    ap.add_argument("--max_retries", type=int, default=2)
    ap.add_argument("--fail_on_invalid", action="store_true")

    args = ap.parse_args()

    doc_path = Path(args.doc_txt).resolve()
    yaml_in_path = Path(args.yaml_in).resolve()
    yaml_out_path = Path(args.yaml_out).resolve()
    meta_out_path = yaml_out_path.with_suffix(yaml_out_path.suffix + ".meta.json")
    yaml_out_path.parent.mkdir(parents=True, exist_ok=True)

    doc_text = safe_truncate(read_text(doc_path), args.max_doc_chars)
    yaml_obj = load_yaml_obj(yaml_in_path)
    yaml_text = read_text(yaml_in_path)

    if not isinstance(yaml_obj, dict):
        raise RuntimeError(f"Input YAML must be a mapping/dict: {yaml_in_path}")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.gpt.ge/v1/")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY env var")

    client = OpenAI(api_key=api_key, base_url=base_url, default_headers={"x-foo": "true"})

    allowed_names = _build_allowed_names(yaml_obj)
    primary_param, available_ranks = _infer_primary_param_and_ranks(yaml_obj)
    shape_context_text = _describe_shape_context(yaml_obj)

    system_prompt = YAML_CONSTRAINT_SYSTEM_PROMPT
    base_user_prompt = (
        "=== OFFICIAL DOCUMENTATION (TXT) ===\n"
        f"{doc_text}\n\n"
        "=== INPUT YAML (DO NOT REWRITE IT) ===\n"
        f"{yaml_text}\n\n"
        "=== PRIMARY PARAM / RANK SUMMARY ===\n"
        f"primary_param = {primary_param}\n"
        f"available_ranks = {available_ranks}\n\n"
        "=== SHAPE CONTEXT ===\n"
        f"{shape_context_text}\n\n"
        "=== ALLOWED SYMBOLIC NAMES IN CONSTRAINTS ===\n"
        f"{sorted(allowed_names)}\n\n"
        "Return ONLY a JSON object patch following the required schema.\n"
        "Stage D: constraints patch ONLY. Minimal, high-confidence.\n"
        "IMPORTANT: Do not reference any names outside the allowed set above.\n"
    )

    last_errors: List[str] = []
    last_raw: str = ""
    final_patch: Optional[Dict[str, Any]] = None
    user_prompt = base_user_prompt

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

        try:
            patch_dict = parse_patch(last_raw)
            legacy_patch = try_convert_legacy_fullspec_to_constraint_patch(patch_dict, yaml_obj)
            if legacy_patch is not None:
                patch_dict = legacy_patch
        except Exception as e:
            last_errors = [f"JSON parse failed: {e}"]
            if attempt < args.max_retries:
                user_prompt = base_user_prompt + "\nYour previous output was NOT valid JSON. Output ONLY JSON.\n"
                continue
            final_patch = {
                "constraints_append": [],
                "constraints_remove": [],
                "changes": [],
                "warnings": [f"Failed to parse JSON patch: {e}"],
                "confidence": 0.0,
            }
            break

        patch = normalize_constraint_patch(patch_dict)

        last_errors = []
        last_errors += validate_constraints_eval_safety(patch["constraints_append"])
        last_errors += validate_constraints_eval_safety(patch["constraints_remove"])
        last_errors += validate_constraint_names_defined(patch["constraints_append"], allowed_names)

        if not last_errors:
            final_patch = patch
            break

        if attempt < args.max_retries:
            user_prompt = (
                base_user_prompt
                + "\nYour previous JSON FAILED validation. Fix it and output ONLY JSON.\n"
                + "Validation errors:\n"
                + "\n".join(f"- {e}" for e in last_errors)
                + "\n\nPrevious JSON (for reference):\n"
                + (extract_json_object(last_raw) if last_raw else "")
                + "\n"
            )
        else:
            final_patch = patch

    if final_patch is None:
        raise RuntimeError("LLM call failed (no patch produced)")

    # final sanitization: drop any still-invalid constraints
    final_patch = normalize_constraint_patch(final_patch)

    valid_append, dropped_append = _filter_valid_constraints(
        final_patch.get("constraints_append") or [],
        allowed_names,
        check_names=True,
    )
    valid_remove, dropped_remove = _filter_valid_constraints(
        final_patch.get("constraints_remove") or [],
        allowed_names,
        check_names=False,
    )

    final_patch["constraints_append"] = valid_append
    final_patch["constraints_remove"] = valid_remove

    if dropped_append or dropped_remove:
        final_patch.setdefault("warnings", []).append(
            f"dropped invalid constraints: append={len(dropped_append)}, remove={len(dropped_remove)}"
        )

    # deterministic fallback: at least add rank constraint if nothing survived
    if not final_patch.get("constraints_append"):
        rank_constraint = _build_rank_constraint(primary_param, available_ranks)
        if rank_constraint:
            final_patch["constraints_append"] = [rank_constraint]
            final_patch.setdefault("changes", []).append("added deterministic fallback rank constraint")
            final_patch.setdefault("warnings", []).append(
                "LLM produced no valid constraints; used fallback rank constraint"
            )

    out_yaml = dict(yaml_obj)
    old_constraints = out_yaml.get("constraints") or []
    out_yaml["constraints"] = merge_constraints(
        existing=old_constraints,
        to_remove=final_patch.get("constraints_remove") or [],
        to_append=final_patch.get("constraints_append") or [],
    )

    yaml_out_path.write_text(dump_yaml_obj(out_yaml), encoding="utf-8")

    meta = {
        "model": args.model,
        "doc_txt": str(doc_path),
        "yaml_in": str(yaml_in_path),
        "yaml_out": str(yaml_out_path),
        "primary_param": primary_param,
        "available_ranks": available_ranks,
        "confidence": float(final_patch.get("confidence", 0.5)),
        "changes": final_patch.get("changes", []),
        "warnings": final_patch.get("warnings", []),
        "validation_errors": last_errors,
        "raw_model_output_snippet": (last_raw[:8000] if last_raw else ""),
    }
    meta_out_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[+] wrote: {yaml_out_path}")
    print(f"[+] wrote: {meta_out_path}")

    if last_errors:
        msg = "[!] Stage-D patch validation errors (not fully fixed):\n" + "\n".join(f"   - {e}" for e in last_errors)
        if args.fail_on_invalid:
            raise SystemExit(msg)
        print(msg)


if __name__ == "__main__":
    main()