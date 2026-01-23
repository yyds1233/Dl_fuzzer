#!/usr/bin/env python3
# patch_constraints.py
import os
import re
import argparse
import json
import traceback
import ast
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

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
        return text[l : r + 1]

    raise ValueError("cannot find JSON object in model output")


def parse_patch(raw_text: str) -> Dict[str, Any]:
    return json.loads(extract_json_object(raw_text))


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

    def _clean_list(lst: List[Any]) -> List[str]:
        cleaned: List[str] = []
        for item in lst:
            if isinstance(item, str) and item.strip():
                cleaned.append(item.strip())
            elif isinstance(item, dict):
                # 容错：有些模型会用 {"expr": "..."} / {"constraint": "..."}
                for k in ("expr", "constraint"):
                    v = item.get(k)
                    if isinstance(v, str) and v.strip():
                        cleaned.append(v.strip())
                        break
        return cleaned

    out["constraints_append"] = _clean_list(out["constraints_append"])
    out["constraints_remove"] = _clean_list(out["constraints_remove"])
    return out


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

    # append with de-dup, keep order
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

    # allowed names: builtins + params + shape_vars + helper tuples
    allowed_names = set(_ALLOWED_BUILTINS)
    params = yaml_obj.get("params")
    if isinstance(params, dict):
        allowed_names |= set(params.keys())
    shape_vars = yaml_obj.get("shape_vars")
    if isinstance(shape_vars, dict):
        allowed_names |= set(shape_vars.keys())
    allowed_names |= {"padding_tuple", "stride_tuple", "dilation_tuple", "kernel_size_tuple"}

    system_prompt = YAML_CONSTRAINT_SYSTEM_PROMPT
    base_user_prompt = (
        "=== OFFICIAL DOCUMENTATION (TXT) ===\n"
        f"{doc_text}\n\n"
        "=== INPUT YAML (DO NOT REWRITE IT) ===\n"
        f"{yaml_text}\n\n"
        "Return ONLY a JSON object patch following the required schema.\n"
        "Stage D: constraints patch ONLY. Minimal, high-confidence.\n"
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

        # validate
        last_errors = []
        last_errors += validate_constraints_eval_safety(patch["constraints_append"])
        last_errors += validate_constraints_eval_safety(patch["constraints_remove"])  # remove也要求是安全字符串
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
            final_patch = patch  # best-effort

    if final_patch is None:
        raise RuntimeError("LLM call failed (no patch produced)")

    # apply to yaml
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
