#!/usr/bin/env python3
# llm_patch_yaml.py
import os
import re
import argparse
import json
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
    # keep stable, readable output
    return yaml.safe_dump(
        obj,
        sort_keys=False,
        allow_unicode=True,
        width=1000,
        default_flow_style=False,
    )


# ----------------------------
# 2) JSON patch parsing
# ----------------------------
def extract_json_object(text: str) -> str:
    """
    网关/模型有时会在 JSON 前后加解释文字，这里做一次“尽量提取第一个 {...} 块”的兜底。
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("empty model output")

    # 直接就是 JSON
    if text.startswith("{") and text.endswith("}"):
        return text

    # 尝试提取第一个大括号对象（贪婪到最后一个 }）
    l = text.find("{")
    r = text.rfind("}")
    if l != -1 and r != -1 and r > l:
        return text[l : r + 1]

    raise ValueError("cannot find JSON object in model output")


def parse_patch(raw_text: str) -> Dict[str, Any]:
    js = extract_json_object(raw_text)
    return json.loads(js)


def normalize_patch(patch_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    统一 patch 格式，确保字段存在且类型合理（不做深度强制，只做最关键的 normalization）
    Expected patch keys:
      - shape_vars: dict[str, [lo, hi]]
      - constraints: list[str]
      - shape_spec_fixes: dict[str, list[str]]  (optional)
      - changes: list[str] (optional)
      - confidence: float (optional)
      - warnings: list[str] (optional)
    """
    out: Dict[str, Any] = {}

    out["shape_vars"] = patch_dict.get("shape_vars") or {}
    out["constraints"] = patch_dict.get("constraints") or []
    out["shape_spec_fixes"] = patch_dict.get("shape_spec_fixes") or {}

    out["changes"] = patch_dict.get("changes") or []
    out["warnings"] = patch_dict.get("warnings") or []
    out["confidence"] = patch_dict.get("confidence", 0.5)

    # 类型兜底
    if not isinstance(out["shape_vars"], dict):
        out["shape_vars"] = {}
    if not isinstance(out["constraints"], list):
        out["constraints"] = []
    if not isinstance(out["shape_spec_fixes"], dict):
        out["shape_spec_fixes"] = {}
    if not isinstance(out["changes"], list):
        out["changes"] = []
    if not isinstance(out["warnings"], list):
        out["warnings"] = []
    try:
        out["confidence"] = float(out["confidence"])
    except Exception:
        out["confidence"] = 0.5

    # ✅ constraints: force list[str] with best-effort extraction
    cleaned_constraints: List[str] = []
    dropped: List[Tuple[int, str]] = []

    for i, item in enumerate(out["constraints"]):
        if isinstance(item, str):
            cleaned_constraints.append(item)
            continue

        # Best-effort: sometimes the model returns {"expr": "..."} or {"constraint": "..."}
        if isinstance(item, dict):
            for key in ("expr", "constraint"):
                v = item.get(key)
                if isinstance(v, str) and v.strip():
                    cleaned_constraints.append(v)
                    break
            else:
                dropped.append((i, f"dict({list(item.keys())})"))
            continue

        dropped.append((i, type(item).__name__))

    if dropped:
        out["warnings"].append(f"Dropped non-string constraints: {dropped}")

    out["constraints"] = cleaned_constraints

    return out


# ----------------------------
# 3) validation utilities
# ----------------------------
_ALLOWED_BUILTINS = {
    "isinstance", "all", "any", "len", "tuple",
    "min", "max", "abs",
}

_NAME_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")


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
        if any(ch in item for ch in ("+", "-", "*", "/", "%", "(", ")", " ", "[", "]")):
            return f"shape_spec contains expression-like token: {item!r}"
    return None


def extract_names(expr: str) -> Set[str]:
    names = set(_NAME_RE.findall(expr))
    keywords = {"and", "or", "not", "in", "is", "None", "True", "False"}
    return {n for n in names if n not in keywords}


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


# def validate_constraint_names_defined(constraints: List[str], allowed_names: Set[str]) -> List[str]:
#     errs = []
#     for i, c in enumerate(constraints):
#         names = extract_names(c)
#         names = {n for n in names if n not in _ALLOWED_BUILTINS}
#         unknown = sorted([n for n in names if n not in allowed_names])
#         if unknown:
#             errs.append(f"constraints[{i}] references undefined names: {unknown} | expr={c!r}")
#     return errs
def validate_constraint_names_defined(constraints: List[Any], allowed_names: Set[str]) -> List[str]:
    errs = []
    for i, c in enumerate(constraints):
        if not isinstance(c, str):
            errs.append(f"constraints[{i}] must be a string, got {type(c).__name__}: {c!r}")
            continue

        names = extract_names(c)
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


# ----------------------------
# 4) apply patch (keep everything else untouched)
# ----------------------------
def apply_patch_to_yaml_obj(yaml_obj: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(yaml_obj)

    if "shape_vars" not in out or out["shape_vars"] is None:
        out["shape_vars"] = {}
    if "constraints" not in out or out["constraints"] is None:
        out["constraints"] = []

    out["shape_vars"] = normalize_shape_vars_for_write(patch.get("shape_vars") or {})
    out["constraints"] = list(patch.get("constraints") or [])

    # optional: fix TODO_SHAPE only
    fixes = patch.get("shape_spec_fixes") or {}
    if fixes and "params" in out and isinstance(out["params"], dict):
        for pname, new_spec in fixes.items():
            if pname not in out["params"]:
                continue
            p = out["params"][pname]
            if not isinstance(p, dict):
                continue
            if "shape_spec" not in p:
                continue

            old = p["shape_spec"]
            if not (isinstance(old, list) and all(isinstance(x, str) for x in old)):
                continue
            if any(x == "TODO_SHAPE" for x in old):
                p["shape_spec"] = list(new_spec)

    return out


# ----------------------------
# 5) LLM call (chat.completions) + retry with feedback
# ----------------------------
def call_llm_for_patch(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """
    使用 chat.completions（最兼容），返回 assistant 的原始文本。
    """
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
    ap.add_argument("--max_tokens", type=int, default=1500, help="chat.completions max_tokens")
    ap.add_argument("--max_retries", type=int, default=2, help="retry when patch validation fails")

    args = ap.parse_args()

    doc_path = Path(args.doc_txt).resolve()
    yaml_in_path = Path(args.yaml_in).resolve()
    yaml_out_path = Path(args.yaml_out).resolve()
    meta_out_path = yaml_out_path.with_suffix(yaml_out_path.suffix + ".meta.json")

    doc_text = safe_truncate(read_text(doc_path), args.max_doc_chars)
    yaml_obj = load_yaml_obj(yaml_in_path)
    yaml_text = read_text(yaml_in_path)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.gpt.ge/v1/")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY env var")

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers={"x-foo": "true"},
    )

    # Allowed names set for constraint name checking
    allowed_names = set()
    if isinstance(yaml_obj, dict) and isinstance(yaml_obj.get("params"), dict):
        allowed_names |= set(yaml_obj["params"].keys())
    allowed_names |= {"padding_tuple", "stride_tuple", "dilation_tuple", "kernel_size_tuple"}

    system_prompt = YAML_PATCH_SYSTEM_PROMPT

    base_user_prompt = (
        "=== OFFICIAL DOCUMENTATION (TXT) ===\n"
        f"{doc_text}\n\n"
        "=== YAML SKELETON (INPUT YAML; DO NOT REWRITE IT) ===\n"
        f"{yaml_text}\n\n"
        "Return ONLY a JSON object patch with these keys:\n"
        "shape_vars, constraints, shape_spec_fixes (optional), changes (optional), confidence (optional), warnings (optional).\n"
        "Do NOT output YAML. Do NOT add extra commentary outside JSON.\n"
    )

    last_errors: List[str] = []
    final_patch: Optional[Dict[str, Any]] = None
    last_raw: str = ""

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

        # parse + normalize
        try:
            patch_dict = parse_patch(last_raw)
        except Exception as e:
            last_errors = [f"JSON parse failed: {e}"]
            if attempt < args.max_retries:
                user_prompt = (
                    base_user_prompt
                    + "\n\n"
                    + "Your previous output was NOT valid JSON.\n"
                    + "Please output ONLY valid JSON object (no extra text).\n"
                )
                continue
            final_patch = {
                "shape_vars": {},
                "constraints": [],
                "shape_spec_fixes": {},
                "changes": [],
                "confidence": 0.0,
                "warnings": [f"Failed to parse JSON patch: {e}"],
            }
            break

        patch = normalize_patch(patch_dict)

        # ---- validate patch ----
        last_errors = []
        last_errors += validate_shape_vars(patch["shape_vars"])
        last_errors += validate_constraints_eval_safety(patch["constraints"])

        allowed_now = allowed_names | set(patch["shape_vars"].keys())
        last_errors += validate_constraint_names_defined(patch["constraints"], allowed_now)

        # validate TODO_SHAPE fixes
        fixes = patch.get("shape_spec_fixes") or {}
        if fixes:
            for pname, spec in fixes.items():
                if not isinstance(spec, list) or not all(isinstance(x, str) for x in spec):
                    last_errors.append(f"shape_spec_fixes[{pname}] must be list[str]")
                    continue
                err = validate_no_expressions_in_shape_spec(spec)
                if err:
                    last_errors.append(f"shape_spec_fixes[{pname}] invalid: {err}")
                missing = [x for x in spec if x not in patch["shape_vars"]]
                if missing:
                    last_errors.append(
                        f"shape_spec_fixes[{pname}] references vars not in shape_vars: {missing}"
                    )

        if not last_errors:
            final_patch = patch
            break

        # retry with feedback
        if attempt < args.max_retries:
            user_prompt = (
                base_user_prompt
                + "\n\n"
                + "Your previous JSON patch FAILED validation.\n"
                + "Fix the patch and output ONLY JSON again.\n"
                + "Validation errors:\n"
                + "\n".join(f"- {e}" for e in last_errors)
                + "\n\n"
                + "Previous patch (for reference):\n"
                + (extract_json_object(last_raw) if last_raw else "")
                + "\n"
            )
        else:
            final_patch = patch  # keep last patch even if invalid (we will record errors)

    if final_patch is None:
        raise RuntimeError("LLM call failed (no patch produced)")

    merged = apply_patch_to_yaml_obj(yaml_obj, final_patch)

    yaml_out_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_out_path.write_text(dump_yaml_obj(merged), encoding="utf-8")

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
        print("[!] patch validation errors (not fixed):")
        for e in last_errors:
            print("   -", e)
    if meta["warnings"]:
        print("[!] warnings:")
        for w in meta["warnings"]:
            print("   -", w)


if __name__ == "__main__":
    main()
