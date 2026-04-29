#!/usr/bin/env python3
"""
LLM-driven PyTorch harness generator for Dl_fuzzer.

Reads:
  1) one or more API YAML files (same PyTorch API, possibly different overloads)
  2) one API txt doc file
  3) one Atheris README / notes file

Then calls an OpenAI-compatible LLM endpoint to generate a standalone,
executable Atheris fuzz harness (.py), extracts Python code from the model
response, validates it, optionally asks the model to repair it once, and
writes the final harness to disk.

Key PyTorch-specific behavior:
- Supports multiple YAML files for the same API.
- Treats YAMLs as high-value but imperfect overload/constraint hints.
- Builds one prompt that includes all matching YAMLs, the API txt, and the
  Atheris notes.

Examples:
  # Explicitly pass multiple YAMLs for the same API
  python llm_harness_codegen_torch.py \
    --yaml build_yaml/torch.select_scatter.default.yaml \
    --yaml build_yaml/torch.select_scatter.out.yaml \
    --api-txt api_txt/torch.select_scatter.txt \
    --atheris-doc docs/atheris_readme.txt \
    --out fuzz_output/llm.torch.select_scatter.py \
    --model gpt-5.4

  # Or collect all YAMLs in a directory whose api_name == torch.select_scatter
  python llm_harness_codegen_torch.py \
    --yaml-dir build_yaml \
    --api-name torch.select_scatter \
    --api-txt api_txt/torch.select_scatter.txt \
    --atheris-doc docs/atheris_readme.txt \
    --out fuzz_output/llm.torch.select_scatter.py \
    --model gpt-5.4

Environment variables:
  OPENAI_API_KEY      API key for OpenAI-compatible endpoint.
  OPENAI_BASE_URL     Optional base url for OpenAI-compatible endpoint.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import yaml

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: openai\n"
        "Install with: pip install openai pyyaml"
    ) from exc


DEFAULT_SYSTEM_PROMPT = """
You are an expert Python fuzzing engineer specializing in coverage-guided fuzzing for PyTorch APIs with Atheris.

Your job is to generate one high-quality, executable Atheris harness for the target PyTorch API.

Priority order:
1. The harness must be executable and syntactically valid Python 3.
2. The harness must actually invoke the target PyTorch API from the provided materials.
3. The harness should maximize reachable API parameter-space coverage and downstream code coverage.
4. The harness should use the YAMLs aggressively when they are plausible, but must not follow any YAML blindly if it appears inconsistent with the API text, PyTorch semantics, or basic executability.

Interpretation rules:
- Treat the YAMLs as high-value but potentially imperfect intermediate representations.
- There may be multiple YAML files for the same API corresponding to different overloads or normalization stages.
- Prefer the YAMLs when they appear reasonable and internally consistent.
- If the YAMLs seem incomplete, noisy, partially conflicting, or clearly inconsistent with the API text or likely PyTorch calling semantics, use your best judgment and rely more on the API text and normal PyTorch usage patterns.
- Never intentionally produce a harness that is obviously broken just to obey a suspicious YAML detail.
- When uncertain, favor a harness that is executable, exercises the real API, and explores diverse valid argument combinations.

Coverage goals:
- Prefer harness logic that can explore broad combinations of dtypes, ranks, shapes, optional arguments, overload modes, keyword arguments, and boundary values over time.
- Do not collapse the harness into one narrow “mostly valid” input pattern.
- Bias toward semantically valid inputs, but preserve meaningful diversity so the fuzzer can reach as much code as possible.
- Where compatible parameters are required, enforce compatibility.
- Where the API admits multiple valid modes, branches, or overload choices, expose those modes to fuzzing rather than hard-coding one mode.
- Prefer cheap/small tensors by default, but still allow controlled variation across ranks, dimensions, shapes, and values so different execution paths remain reachable.

Hard requirements:
1. Output exactly one Python code block and nothing else.
2. The harness must be a complete standalone .py file.
3. Import `atheris` first.
4. Do NOT use `with atheris.instrument_imports():`.
5. Do NOT call `atheris.instrument_all()`.
6. Add `@atheris.instrument_func` immediately before `TestOneInput`.
7. Define `TestOneInput(data: bytes)`.
8. Call `atheris.Setup(sys.argv, TestOneInput)` before `atheris.Fuzz()`.
9. Use `atheris.FuzzedDataProvider(data)` to decode bytes into arguments.
10. The harness must actually call the target PyTorch API named in the provided files.
11. Prefer generating PyTorch values that are valid often enough to reach deep code, while still exploring diverse argument combinations.
12. Use exactly the following exception-swallowing policy inside `TestOneInput`:

except (
    RuntimeError,
    TypeError,
    ValueError,
    IndexError,
    AssertionError,
    NotImplementedError,
):
    return
except Exception:
    return

13. Avoid placeholder code, TODOs, pseudo code, markdown, and explanations.
14. Do not depend on any local project helper modules unless they are explicitly included in the provided files.
15. Use deterministic helper functions inside the same file when needed.
16. The output must be directly runnable with Python after dependencies are installed.

Quality bar:
- The best answer is not the shortest harness.
- The best answer is the harness most likely to run successfully and cover a wide range of the target API’s parameter space.
""".strip()


PROMPT_TEMPLATE = """
Generate one executable Python Atheris harness for the PyTorch API described below.

Primary goal:
- Produce a coverage-guided fuzz harness that is executable, actually reaches the target API, and explores as much of the API’s valid parameter space as reasonably possible.

Reliability and source-priority rules:
- The YAMLs are intermediate representations and may be imperfect.
- Use the YAMLs as the primary structural hints for parameters, dtype relations, ranks, shapes, allowed values, constraints, and overload variants when they appear plausible.
- However, do NOT treat the YAMLs as infallible.
- If the YAMLs appear clearly inconsistent, overly noisy, incomplete, or incompatible with the API txt / likely PyTorch semantics / basic executability, then partially or fully override them using the API txt and your best judgment.
- If forced to choose, prefer:
  (a) an executable harness that correctly calls the target API and explores meaningful inputs
  over
  (b) strict obedience to a suspicious YAML detail.

Output contract:
- Return exactly one fenced Python code block.
- No prose before or after the code block.
- The file must be directly runnable with `python harness.py` after dependencies are installed.

Target API:
- API name: {api_name}

Multi-YAML guidance:
- Multiple YAML files may describe the same API with different overloads, views, or normalization stages.
- Reconcile them as best as possible.
- It is acceptable to let the harness branch between overload-compatible call shapes if that improves coverage while remaining executable.

Harness design objectives:
- Use the YAMLs as much as reasonably possible.
- Try to cover the full parameter space, not just one narrow valid corner.
- Maximize opportunities for code coverage by exposing:
  - different valid dtypes
  - different valid ranks
  - different compatible shapes
  - optional arguments when applicable
  - overload/value alternatives
  - boundary-sized tensors and representative edge-case values
- Prefer structured decoding from `atheris.FuzzedDataProvider(data)` rather than ad hoc randomness.
- Enforce parameter compatibility when the API requires shared dtype/shape/rank relationships.
- If attributes have allowed values, expose multiple allowed choices to fuzzing.
- If multiple argument construction strategies are valid, prefer the one that gives better reachable coverage while staying reasonably executable.
- Prefer small tensor sizes to keep execution cheap, but do not make shapes so trivial that most branches become unreachable.
- Include helper functions in the same file to decode booleans, ints, floats, strings, shapes, dtypes, tensors, lists, and optional values when useful.
- Avoid noisy logging/printing.
- Keep the harness self-contained.

Validity strategy:
- The harness should be validity-biased, not validity-only.
- Generate inputs that are often valid enough to execute meaningful code paths.
- Still preserve diversity across parameter combinations so the fuzzer can mutate toward deeper coverage.
- Use the required exception-swallowing policy exactly.

Reasoning policy for conflicting information:
- When YAMLs and API text agree, follow them closely.
- When YAMLs are missing detail, infer reasonable PyTorch-compatible behavior.
- When YAMLs conflict with the API text or appear obviously wrong, downweight the YAMLs and generate the best executable coverage-oriented harness you can.
- Do not mention this reasoning in the final output; only emit the Python file.

Implementation hints:
- Ensure the target API is actually called on the fuzzed arguments.
- If useful, vary overload mode / keyword shape via small branches driven by fuzz input.
- Reuse decoded fuzz choices to coordinate related parameters.
- Prefer helper abstractions that make the parameter-space exploration broader and cleaner.
- When a parameter space is very large, choose a compact but diverse representation that still lets mutations reach many combinations.

==== YAML SUMMARY BEGIN ====
{yaml_summary}
==== YAML SUMMARY END ====

==== ALL API YAMLs BEGIN ====
{all_yaml_text}
==== ALL API YAMLs END ====

==== API TXT BEGIN ====
{api_txt_text}
==== API TXT END ====

==== ATHERIS DOC BEGIN ====
{atheris_doc_text}
==== ATHERIS DOC END ====
""".strip()


@dataclass
class YamlInput:
    path: Path
    api_name: str
    data: dict
    text: str


@dataclass
class GenConfig:
    yaml_paths: List[Path]
    yaml_dir: Optional[Path]
    api_name: Optional[str]
    api_txt_path: Path
    atheris_doc_path: Path
    out_path: Path
    model: str
    api_key: Optional[str]
    base_url: Optional[str]
    api_mode: str
    temperature: float
    max_output_tokens: int
    max_yaml_chars_each: int
    max_api_txt_chars: int
    max_atheris_chars: int
    repair_attempts: int
    save_raw_response: bool
    save_prompt: bool


class HarnessGenerationError(RuntimeError):
    pass


def parse_args() -> GenConfig:
    parser = argparse.ArgumentParser(
        description="Generate an Atheris fuzz harness for a PyTorch API from one or more YAMLs + API txt + Atheris docs using an LLM."
    )
    parser.add_argument(
        "--yaml",
        dest="yaml_paths",
        action="append",
        default=[],
        help="Path to one API YAML file. Repeat this flag to pass multiple YAMLs for the same API.",
    )
    parser.add_argument(
        "--yaml-dir",
        dest="yaml_dir",
        default=None,
        help="Directory to recursively scan for YAML files. Use with --api-name to select all YAMLs whose api_name matches.",
    )
    parser.add_argument(
        "--api-name",
        dest="api_name",
        default=None,
        help="Target API name, e.g. torch.select_scatter. Required when using --yaml-dir unless a single API can be inferred.",
    )
    parser.add_argument("--api-txt", dest="api_txt_path", required=True, help="Path to API txt file")
    parser.add_argument(
        "--atheris-doc",
        dest="atheris_doc_path",
        required=True,
        help="Path to Atheris README/notes file",
    )
    parser.add_argument("--out", dest="out_path", required=True, help="Output harness .py path")
    parser.add_argument("--model", default="gpt-5.4", help="LLM model name")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY"),
        help="API key (default: OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL"),
        help="OpenAI-compatible base URL (default: OPENAI_BASE_URL)",
    )
    parser.add_argument(
        "--api-mode",
        choices=["responses", "chat", "auto"],
        default="auto",
        help="LLM API mode. auto = try responses first, then chat.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=120000)
    parser.add_argument("--max-yaml-chars-each", type=int, default=250000)
    parser.add_argument("--max-api-txt-chars", type=int, default=1200000)
    parser.add_argument("--max-atheris-chars", type=int, default=1200000)
    parser.add_argument(
        "--repair-attempts",
        type=int,
        default=1,
        help="How many repair rounds to attempt after static validation fails.",
    )
    parser.add_argument(
        "--save-raw-response",
        action="store_true",
        help="Also save raw model response next to output file",
    )
    parser.add_argument(
        "--save-prompt",
        action="store_true",
        help="Also save final prompt next to output file",
    )

    ns = parser.parse_args()
    return GenConfig(
        yaml_paths=[Path(x) for x in ns.yaml_paths],
        yaml_dir=Path(ns.yaml_dir) if ns.yaml_dir else None,
        api_name=ns.api_name,
        api_txt_path=Path(ns.api_txt_path),
        atheris_doc_path=Path(ns.atheris_doc_path),
        out_path=Path(ns.out_path),
        model=ns.model,
        api_key=ns.api_key,
        base_url=ns.base_url,
        api_mode=ns.api_mode,
        temperature=ns.temperature,
        max_output_tokens=ns.max_output_tokens,
        max_yaml_chars_each=ns.max_yaml_chars_each,
        max_api_txt_chars=ns.max_api_txt_chars,
        max_atheris_chars=ns.max_atheris_chars,
        repair_attempts=ns.repair_attempts,
        save_raw_response=ns.save_raw_response,
        save_prompt=ns.save_prompt,
    )


def read_text(path: Path, max_chars: int) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-(max_chars - len(head)) :]
    omitted = len(text) - len(head) - len(tail)
    return f"{head}\n\n[... TRUNCATED {omitted} CHARS ...]\n\n{tail}"


def load_one_yaml(path: Path, max_chars_each: int) -> YamlInput:
    text = read_text(path, max_chars_each)
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise HarnessGenerationError(f"YAML root must be a mapping/object: {path}")
    api_name = str(data.get("api_name") or "UNKNOWN_API")
    return YamlInput(path=path, api_name=api_name, data=data, text=text)


def collect_yaml_candidates(cfg: GenConfig) -> List[YamlInput]:
    yaml_inputs: List[YamlInput] = []

    for p in cfg.yaml_paths:
        yaml_inputs.append(load_one_yaml(p, cfg.max_yaml_chars_each))

    if cfg.yaml_dir:
        if not cfg.yaml_dir.is_dir():
            raise FileNotFoundError(f"YAML directory not found: {cfg.yaml_dir}")
        for p in sorted(cfg.yaml_dir.rglob("*.yml")) + sorted(cfg.yaml_dir.rglob("*.yaml")):
            try:
                yi = load_one_yaml(p, cfg.max_yaml_chars_each)
            except Exception:
                continue
            if cfg.api_name and yi.api_name != cfg.api_name:
                continue
            yaml_inputs.append(yi)

    # de-duplicate by resolved path while preserving order
    dedup: List[YamlInput] = []
    seen = set()
    for yi in yaml_inputs:
        key = str(yi.path.resolve())
        if key in seen:
            continue
        seen.add(key)
        dedup.append(yi)

    if not dedup:
        raise HarnessGenerationError("No YAML inputs found. Provide --yaml and/or --yaml-dir with --api-name.")

    # Infer/validate api_name
    api_names = {yi.api_name for yi in dedup if yi.api_name and yi.api_name != "UNKNOWN_API"}
    if cfg.api_name:
        dedup = [yi for yi in dedup if yi.api_name in {cfg.api_name, "UNKNOWN_API"}]
        if not dedup:
            raise HarnessGenerationError(f"No YAMLs matched api_name={cfg.api_name!r}")
        api_names = {yi.api_name for yi in dedup if yi.api_name and yi.api_name != "UNKNOWN_API"}
        if api_names and api_names != {cfg.api_name}:
            raise HarnessGenerationError(
                f"YAML api_name mismatch under requested api_name={cfg.api_name!r}: found {sorted(api_names)}"
            )
    else:
        if len(api_names) > 1:
            raise HarnessGenerationError(
                f"Multiple api_name values found across YAMLs: {sorted(api_names)}. Pass --api-name to disambiguate."
            )

    return dedup


def choose_api_name(cfg: GenConfig, yaml_inputs: Sequence[YamlInput]) -> str:
    if cfg.api_name:
        return cfg.api_name
    for yi in yaml_inputs:
        if yi.api_name and yi.api_name != "UNKNOWN_API":
            return yi.api_name
    return "UNKNOWN_API"


def _shorten(value, limit: int = 180) -> str:
    s = repr(value)
    return s if len(s) <= limit else s[: limit - 3] + "..."


def build_yaml_summary(api_name: str, yaml_inputs: Sequence[YamlInput]) -> str:
    lines: List[str] = []
    lines.append(f"Target API: {api_name}")
    lines.append(f"Number of YAML files: {len(yaml_inputs)}")

    for idx, yi in enumerate(yaml_inputs, start=1):
        d = yi.data
        lines.append(f"\n[{idx}] source_yaml={yi.path}")

        category = d.get("category")
        if category is not None:
            lines.append(f"  category: {_shorten(category)}")

        rank_hints = d.get("rank_hints")
        if isinstance(rank_hints, dict):
            lines.append(
                "  rank_hints: "
                f"status={_shorten(rank_hints.get('status'))}, "
                f"candidates={_shorten(rank_hints.get('rank_candidates'))}, "
                f"rank_any={_shorten(rank_hints.get('rank_any'))}"
            )

        aten = d.get("aten")
        if isinstance(aten, dict):
            lines.append(
                "  aten: "
                f"aten_name={_shorten(aten.get('aten_name'))}, "
                f"overload={_shorten(aten.get('overload'))}, "
                f"schema_str={_shorten(aten.get('schema_str'))}"
            )

        shape_vars = d.get("shape_vars")
        if isinstance(shape_vars, dict):
            lines.append(f"  shape_vars_keys: {list(shape_vars.keys())}")

        params = d.get("params")
        if isinstance(params, dict):
            lines.append(f"  params: {list(params.keys())}")
            for pname, pinfo in list(params.items())[:12]:
                if not isinstance(pinfo, dict):
                    continue
                bits = [f"kind={pinfo.get('kind')!r}"]
                if "dtype_choices" in pinfo:
                    bits.append(f"dtype_choices={_shorten(pinfo.get('dtype_choices'))}")
                if "shape_spec" in pinfo:
                    bits.append(f"shape_spec={_shorten(pinfo.get('shape_spec'))}")
                if "range" in pinfo:
                    bits.append(f"range={_shorten(pinfo.get('range'))}")
                lines.append(f"    - {pname}: " + ", ".join(bits))

        constraints = d.get("constraints")
        if isinstance(constraints, list):
            lines.append(f"  constraints_count: {len(constraints)}")
            for c in constraints[:12]:
                lines.append(f"    - {_shorten(c)}")

        generator = d.get("generator")
        if isinstance(generator, dict):
            lines.append(f"  generator: stage={_shorten(generator.get('stage'))}, version={_shorten(generator.get('version'))}")

    return "\n".join(lines)


def build_all_yaml_text(yaml_inputs: Sequence[YamlInput]) -> str:
    blocks: List[str] = []
    for idx, yi in enumerate(yaml_inputs, start=1):
        blocks.append(f"---- YAML #{idx} PATH: {yi.path} ----\n{yi.text}")
    return "\n\n".join(blocks)


def build_prompt(api_name: str, yaml_summary: str, all_yaml_text: str, api_txt_text: str, atheris_doc_text: str) -> str:
    return PROMPT_TEMPLATE.format(
        api_name=api_name,
        yaml_summary=yaml_summary,
        all_yaml_text=all_yaml_text,
        api_txt_text=api_txt_text,
        atheris_doc_text=atheris_doc_text,
    )


def create_client(cfg: GenConfig) -> OpenAI:
    kwargs = {}
    if cfg.api_key:
        kwargs["api_key"] = cfg.api_key
    if cfg.base_url:
        kwargs["base_url"] = cfg.base_url
    return OpenAI(**kwargs)


def call_model_responses(client: OpenAI, cfg: GenConfig, system_prompt: str, user_prompt: str) -> str:
    resp = client.responses.create(
        model=cfg.model,
        instructions=system_prompt,
        input=user_prompt,
        max_output_tokens=cfg.max_output_tokens,
        temperature=cfg.temperature,
    )
    text = getattr(resp, "output_text", None)
    if text:
        return text
    try:
        return json.dumps(resp.model_dump(), ensure_ascii=False, indent=2)
    except Exception:
        return str(resp)


def call_model_chat(client: OpenAI, cfg: GenConfig, system_prompt: str, user_prompt: str) -> str:
    completion = client.chat.completions.create(
        model=cfg.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=cfg.temperature,
        max_tokens=cfg.max_output_tokens,
    )
    choice = completion.choices[0]
    content = choice.message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(parts)
    return str(content)


def call_model(client: OpenAI, cfg: GenConfig, system_prompt: str, user_prompt: str) -> Tuple[str, str]:
    errors = []
    modes = [cfg.api_mode] if cfg.api_mode != "auto" else ["responses", "chat"]
    for mode in modes:
        try:
            if mode == "responses":
                return call_model_responses(client, cfg, system_prompt, user_prompt), mode
            if mode == "chat":
                return call_model_chat(client, cfg, system_prompt, user_prompt), mode
            raise ValueError(f"Unsupported api mode: {mode}")
        except Exception as exc:
            errors.append(f"{mode}: {exc}")
    raise HarnessGenerationError("LLM call failed in all modes:\n" + "\n".join(errors))


_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_python_code(text: str) -> str:
    match = _CODE_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip() + "\n"

    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            obj = json.loads(stripped)
            if isinstance(obj, dict):
                code = obj.get("code") or obj.get("python")
                if isinstance(code, str) and code.strip():
                    return code.strip() + "\n"
        except Exception:
            pass

    return stripped + ("\n" if stripped else "")


FIRST_IMPORT_RE = re.compile(r"^\s*(?:#.*\n|\n)*(import\s+\w+|from\s+\w+[\.\w]*\s+import\s+.+)", re.M)


def _api_markers(api_name: str) -> List[str]:
    markers = {"torch", api_name}
    tail = api_name.split(".")[-1].strip()
    if tail:
        markers.add(tail)
    return sorted(markers)


def static_validate_python(code: str, api_name: str) -> Tuple[bool, str]:
    problems: List[str] = []

    if not code.strip():
        return False, "Empty code generated."

    try:
        ast.parse(code)
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc.msg} (line {exc.lineno}, col {exc.offset})"

    m = FIRST_IMPORT_RE.search(code)
    if not m:
        problems.append("Could not determine the first import statement.")
    else:
        first_import_stmt = m.group(1)
        if not first_import_stmt.startswith("import atheris"):
            problems.append(f"First import must be 'import atheris', got: {first_import_stmt!r}")

    expected_markers = [
        "import atheris",
        "@atheris.instrument_func",
        "TestOneInput",
        "atheris.FuzzedDataProvider",
        "atheris.Setup",
        "atheris.Fuzz",
        "torch",
    ]
    for marker in expected_markers:
        if marker not in code:
            problems.append(f"Missing required marker: {marker}")

    fn_sig_ok = (
        "def TestOneInput(data: bytes)" in code
        or "def TestOneInput(data):" in code
    )
    if not fn_sig_ok:
        problems.append("Missing expected TestOneInput(data: bytes) or TestOneInput(data) signature.")

    if "with atheris.instrument_imports():" in code:
        problems.append("Harness must not use 'with atheris.instrument_imports():'.")
    if "atheris.instrument_all()" in code:
        problems.append("Harness must not call 'atheris.instrument_all()'.")

    api_markers = _api_markers(api_name)
    if not any(marker in code for marker in api_markers):
        problems.append(
            "Generated code does not appear to reference the target API. "
            f"Expected one of: {api_markers}"
        )

    return len(problems) == 0, "\n".join(problems) if problems else "OK"


def build_repair_prompt(
    original_prompt: str,
    bad_response: str,
    bad_code: str,
    validation_error: str,
) -> str:
    return textwrap.dedent(
        f"""
        Your previous output did not pass validation.

        Validation error:
        {validation_error}

        Previous raw response:
        ===== BEGIN BAD RESPONSE =====
        {bad_response}
        ===== END BAD RESPONSE =====

        Extracted code:
        ===== BEGIN BAD CODE =====
        {bad_code}
        ===== END BAD CODE =====

        Please fix the harness.

        Requirements again:
        - Return exactly one fenced Python code block.
        - Must be syntactically valid Python 3.
        - Must import atheris first.
        - Must NOT use `with atheris.instrument_imports():`.
        - Must add `@atheris.instrument_func` immediately before `TestOneInput`.
        - Must define `TestOneInput(data: bytes)` or `TestOneInput(data)`.
        - Must call atheris.Setup(sys.argv, TestOneInput) before atheris.Fuzz().
        - Must target the PyTorch API described in the original prompt.
        - Must stay self-contained in a single file.

        Original task again:
        ===== BEGIN ORIGINAL TASK =====
        {original_prompt}
        ===== END ORIGINAL TASK =====
        """
    ).strip()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    ensure_parent_dir(path)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    cfg = parse_args()

    yaml_inputs = collect_yaml_candidates(cfg)
    api_name = choose_api_name(cfg, yaml_inputs)

    api_txt_text = read_text(cfg.api_txt_path, cfg.max_api_txt_chars)
    atheris_doc_text = read_text(cfg.atheris_doc_path, cfg.max_atheris_chars)

    yaml_summary = build_yaml_summary(api_name, yaml_inputs)
    all_yaml_text = build_all_yaml_text(yaml_inputs)
    prompt = build_prompt(api_name, yaml_summary, all_yaml_text, api_txt_text, atheris_doc_text)

    if cfg.save_prompt:
        write_text(cfg.out_path.with_suffix(cfg.out_path.suffix + ".prompt.txt"), prompt)

    client = create_client(cfg)
    raw_response, used_mode = call_model(client, cfg, DEFAULT_SYSTEM_PROMPT, prompt)
    code = extract_python_code(raw_response)
    ok, validation_msg = static_validate_python(code, api_name)

    repair_round = 0
    while not ok and repair_round < cfg.repair_attempts:
        repair_round += 1
        repair_prompt = build_repair_prompt(prompt, raw_response, code, validation_msg)
        raw_response, used_mode = call_model(client, cfg, DEFAULT_SYSTEM_PROMPT, repair_prompt)
        code = extract_python_code(raw_response)
        ok, validation_msg = static_validate_python(code, api_name)

    if cfg.save_raw_response:
        write_text(cfg.out_path.with_suffix(cfg.out_path.suffix + ".raw.txt"), raw_response)

    if not ok:
        raise HarnessGenerationError(
            "Generated harness failed validation after repair attempts.\n"
            f"Last validation error:\n{validation_msg}"
        )

    banner = textwrap.dedent(
        f"""
        # Auto-generated by llm_harness_codegen_torch.py
        # model={cfg.model}
        # api_mode={used_mode}
        # target_api={api_name}
        # source_yaml_count={len(yaml_inputs)}
        # source_api_txt={cfg.api_txt_path}
        # source_atheris_doc={cfg.atheris_doc_path}
        # repair_attempts_used={repair_round}

        """
    ).lstrip()

    final_code = banner + code
    write_text(cfg.out_path, final_code)

    print(f"[OK] Wrote harness: {cfg.out_path}")
    print(f"[OK] API name: {api_name}")
    print(f"[OK] YAML files used: {len(yaml_inputs)}")
    print(f"[OK] LLM mode used: {used_mode}")
    print(f"[OK] Repair rounds used: {repair_round}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
