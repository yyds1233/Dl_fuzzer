#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pipeline.py

Batch pipeline for:
  Stage 0: load API list
  Stage 1: export_schema.py         -> schema json
  Stage 2: doc_rank_extractor.py    -> per-api rank json
  Stage 3: schema2yaml.py           -> yaml skeletons
  Stage 4: normalize_yaml_skeleton.py -> normalized yaml skeletons
  Stage 5: llm_patch_yaml.py        -> Stage-C multi-rank yaml
  Stage 6: patch_constraints.py     -> final yaml with constraints

Assumptions:
- This file lives in the same directory as:
    export_schema.py
    doc_rank_extractor.py
    schema2yaml.py
    normalize_yaml_skeleton.py
    llm_patch_yaml.py
    patch_constraints.py
- docs_root contains txt docs somewhere under it, recursively searchable by basename:
    <api_name>.txt
  e.g.:
    torch.nn.functional.max_pool2d.txt
"""

import argparse
import json
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Make sure local sibling modules are importable
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

# local imports from your existing scripts
from export_schema import load_api_list, export_torch_api_schema, safe_filename as schema_json_filename
from doc_rank_extractor import (
    extract_shape_info,
    read_text_file,
    build_rank_record,
    write_rank_file,
)
from schema2yaml import convert_one_json
from normalize_yaml_skeleton import normalize_one_yaml


# ---------------------------------------------------------
# helpers
# ---------------------------------------------------------

STAGE_NAMES = ["export", "rank", "skeleton", "normalize", "stagec", "staged"]


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_yaml_obj(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8", errors="ignore"))


def write_yaml_obj(path: Path, obj: Any) -> None:
    path.write_text(
        yaml.safe_dump(obj, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def json_dump(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def safe_name_stagec(s: Any, max_len: int = 120) -> str:
    """
    Must match llm_patch_yaml.py::safe_name()
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


def normalize_stages(raw: str) -> List[str]:
    raw = (raw or "all").strip().lower()
    if raw == "all":
        return STAGE_NAMES[:]
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    bad = [x for x in parts if x not in STAGE_NAMES]
    if bad:
        raise ValueError(f"Unknown stages: {bad}. Valid: {STAGE_NAMES} / all")
    return parts


def run_subprocess(cmd: List[str], cwd: Optional[Path] = None) -> Tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        env=os.environ.copy(),
    )
    return proc.returncode, proc.stdout, proc.stderr


def collect_txt_index(docs_root: Path) -> Dict[str, List[Path]]:
    """
    Build basename -> paths index, recursively.
    Key example:
      torch.nn.functional.max_pool2d.txt
    """
    index: Dict[str, List[Path]] = {}
    for p in docs_root.rglob("*.txt"):
        index.setdefault(p.name, []).append(p.resolve())
    for k in index:
        index[k] = sorted(index[k], key=lambda x: (len(x.parts), len(str(x))))
    return index


def choose_doc_for_api(api_name: str, doc_index: Dict[str, List[Path]]) -> Tuple[Optional[Path], List[Path]]:
    """
    Return (best_match, all_matches).
    Match basename exactly: <api_name>.txt
    """
    key = f"{api_name}.txt"
    cands = doc_index.get(key, [])
    if not cands:
        return None, []
    return cands[0], cands


def normalized_yaml_prefix_for_api(api_name: str) -> str:
    """
    Matches schema2yaml.py output naming style:
      safe_name(api_name)__ov_*.yaml
    schema2yaml.safe_name replaces dots with underscores.
    """
    base = (
        api_name.strip()
        .replace("::", "_")
        .replace(".", "_")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
    )
    return f"{base}__ov_"


def schema_json_path_for_api(schema_dir: Path, api_name: str) -> Path:
    return schema_dir / schema_json_filename(api_name)


def find_stagec_outputs_for_yaml(stagec_dir: Path, api_name: str, overload: str) -> List[Path]:
    prefix = f"{safe_name_stagec(api_name)}__ov_{safe_name_stagec(overload)}__"
    return sorted(stagec_dir.glob(prefix + "*__MULTIRANK.yaml"))


def load_api_name_from_yaml(yaml_path: Path) -> Optional[str]:
    try:
        obj = read_yaml_obj(yaml_path)
        if isinstance(obj, dict):
            v = obj.get("api_name")
            return str(v) if v else None
    except Exception:
        return None
    return None


def load_api_and_overload_from_yaml(yaml_path: Path) -> Tuple[Optional[str], Optional[str]]:
    try:
        obj = read_yaml_obj(yaml_path)
        if not isinstance(obj, dict):
            return None, None
        api_name = obj.get("api_name")
        aten = obj.get("aten") or {}
        overload = None
        if isinstance(aten, dict):
            overload = aten.get("overload", "default")
        return (str(api_name) if api_name else None, str(overload) if overload is not None else "default")
    except Exception:
        return None, None


def is_truthy_file(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


# ---------------------------------------------------------
# pipeline stages
# ---------------------------------------------------------

def stage_export(
    api_list: List[str],
    schema_dir: Path,
    overwrite: bool,
    summary: Dict[str, Any],
) -> None:
    ensure_dir(schema_dir)
    for api in api_list:
        out_path = schema_json_path_for_api(schema_dir, api)
        if out_path.exists() and not overwrite:
            summary["export"]["skipped"] += 1
            continue
        try:
            export_torch_api_schema(api, schema_dir)
            summary["export"]["ok"] += 1
        except Exception as e:
            summary["export"]["failed"] += 1
            summary["errors"].append(
                {"stage": "export", "api": api, "error": repr(e), "traceback": traceback.format_exc()}
            )


def stage_rank(
    api_list: List[str],
    api_to_doc: Dict[str, Path],
    rank_dir: Path,
    overwrite: bool,
    with_evidence: bool,
    focus_sections: bool,
    merge_rank: bool,
    summary: Dict[str, Any],
) -> None:
    ensure_dir(rank_dir)
    for api in api_list:
        doc_path = api_to_doc.get(api)
        if not doc_path:
            summary["rank"]["missing_doc"] += 1
            continue

        out_path = rank_dir / (
            api.strip().replace("::", "_").replace(".", "_").replace("/", "_").replace(" ", "_").replace("-", "_")
            + ".rank.json"
        )
        if out_path.exists() and not overwrite:
            summary["rank"]["skipped"] += 1
            continue

        try:
            text = read_text_file(doc_path)
            rec = extract_shape_info(text, focus_sections=focus_sections)
            record = build_rank_record(api, rec, with_evidence=with_evidence)
            write_rank_file(rank_dir, api, record, merge=merge_rank)
            summary["rank"]["ok"] += 1
        except Exception as e:
            summary["rank"]["failed"] += 1
            summary["errors"].append(
                {"stage": "rank", "api": api, "doc": str(doc_path), "error": repr(e), "traceback": traceback.format_exc()}
            )


def stage_skeleton(
    api_list: List[str],
    schema_dir: Path,
    skeleton_dir: Path,
    rank_dir: Path,
    overwrite: bool,
    summary: Dict[str, Any],
) -> None:
    ensure_dir(skeleton_dir)
    for api in api_list:
        schema_path = schema_json_path_for_api(schema_dir, api)
        if not schema_path.exists():
            summary["skeleton"]["missing_schema"] += 1
            continue

        # cheap resume: if matching yaml skeletons already exist and not overwrite -> skip
        prefix = normalized_yaml_prefix_for_api(api)
        existing = list(skeleton_dir.glob(prefix + "*.yaml"))
        if existing and not overwrite:
            summary["skeleton"]["skipped"] += 1
            continue

        try:
            outs = convert_one_json(schema_path, skeleton_dir, rank_index_dir=rank_dir)
            if outs:
                summary["skeleton"]["ok"] += len(outs)
            else:
                summary["skeleton"]["empty"] += 1
        except Exception as e:
            summary["skeleton"]["failed"] += 1
            summary["errors"].append(
                {"stage": "skeleton", "api": api, "schema": str(schema_path), "error": repr(e), "traceback": traceback.format_exc()}
            )


def stage_normalize(
    skeleton_dir: Path,
    normalized_dir: Path,
    overwrite: bool,
    summary: Dict[str, Any],
) -> None:
    ensure_dir(normalized_dir)
    for yp in sorted(list(skeleton_dir.glob("*.yaml")) + list(skeleton_dir.glob("*.yml"))):
        out_path = normalized_dir / yp.name
        if out_path.exists() and not overwrite:
            summary["normalize"]["skipped"] += 1
            continue

        try:
            data = read_yaml_obj(yp)
            if not isinstance(data, dict):
                summary["normalize"]["failed"] += 1
                summary["errors"].append(
                    {"stage": "normalize", "yaml": str(yp), "error": "input yaml is not a mapping"}
                )
                continue

            new_data, stats = normalize_one_yaml(data, fail_on_parse_error=False)
            write_yaml_obj(out_path, new_data)
            if stats.get("files_changed", 0):
                summary["normalize"]["ok"] += 1
            else:
                summary["normalize"]["unchanged"] += 1
        except Exception as e:
            summary["normalize"]["failed"] += 1
            summary["errors"].append(
                {"stage": "normalize", "yaml": str(yp), "error": repr(e), "traceback": traceback.format_exc()}
            )


def stage_stagec(
    scripts_dir: Path,
    api_to_doc: Dict[str, Path],
    normalized_dir: Path,
    stagec_dir: Path,
    overwrite: bool,
    model: str,
    max_doc_chars: int,
    temperature: float,
    max_tokens_stage_c: int,
    max_retries: int,
    fail_on_invalid: bool,
    keep_variant_constraints_by_rank: bool,
    summary: Dict[str, Any],
) -> None:
    ensure_dir(stagec_dir)
    script = scripts_dir / "llm_patch_yaml.py"

    for yp in sorted(list(normalized_dir.glob("*.yaml")) + list(normalized_dir.glob("*.yml"))):
        api_name, overload = load_api_and_overload_from_yaml(yp)
        if not api_name:
            summary["stagec"]["failed"] += 1
            summary["errors"].append({"stage": "stagec", "yaml": str(yp), "error": "cannot read api_name"})
            continue

        doc_path = api_to_doc.get(api_name)
        if not doc_path:
            summary["stagec"]["missing_doc"] += 1
            continue

        existing = find_stagec_outputs_for_yaml(stagec_dir, api_name, overload or "default")
        if existing and not overwrite:
            summary["stagec"]["skipped"] += 1
            continue

        cmd = [
            sys.executable,
            str(script),
            "--doc_txt", str(doc_path),
            "--yaml_in", str(yp),
            "--yaml_out_dir", str(stagec_dir),
            "--model", model,
            "--max_doc_chars", str(max_doc_chars),
            "--temperature", str(temperature),
            "--max_tokens", str(max_tokens_stage_c),
            "--max_retries", str(max_retries),
        ]
        if fail_on_invalid:
            cmd.append("--fail_on_invalid")
        if keep_variant_constraints_by_rank:
            cmd.append("--keep_variant_constraints_by_rank")

        rc, stdout, stderr = run_subprocess(cmd, cwd=scripts_dir)
        if rc == 0:
            summary["stagec"]["ok"] += 1
        else:
            summary["stagec"]["failed"] += 1
            summary["errors"].append(
                {
                    "stage": "stagec",
                    "api": api_name,
                    "yaml_in": str(yp),
                    "doc": str(doc_path),
                    "returncode": rc,
                    "stdout": stdout[-8000:],
                    "stderr": stderr[-8000:],
                }
            )


def stage_staged(
    scripts_dir: Path,
    api_to_doc: Dict[str, Path],
    stagec_dir: Path,
    final_dir: Path,
    overwrite: bool,
    model: str,
    max_doc_chars: int,
    temperature: float,
    max_tokens_stage_d: int,
    max_retries: int,
    fail_on_invalid: bool,
    summary: Dict[str, Any],
) -> None:
    ensure_dir(final_dir)
    script = scripts_dir / "patch_constraints.py"

    for yp in sorted(list(stagec_dir.glob("*.yaml")) + list(stagec_dir.glob("*.yml"))):
        api_name = load_api_name_from_yaml(yp)
        if not api_name:
            summary["staged"]["failed"] += 1
            summary["errors"].append({"stage": "staged", "yaml": str(yp), "error": "cannot read api_name"})
            continue

        doc_path = api_to_doc.get(api_name)
        if not doc_path:
            summary["staged"]["missing_doc"] += 1
            continue

        out_path = final_dir / yp.name
        if out_path.exists() and not overwrite:
            summary["staged"]["skipped"] += 1
            continue

        cmd = [
            sys.executable,
            str(script),
            "--doc_txt", str(doc_path),
            "--yaml_in", str(yp),
            "--yaml_out", str(out_path),
            "--model", model,
            "--max_doc_chars", str(max_doc_chars),
            "--temperature", str(temperature),
            "--max_tokens", str(max_tokens_stage_d),
            "--max_retries", str(max_retries),
        ]
        if fail_on_invalid:
            cmd.append("--fail_on_invalid")

        rc, stdout, stderr = run_subprocess(cmd, cwd=scripts_dir)
        if rc == 0:
            summary["staged"]["ok"] += 1
        else:
            summary["staged"]["failed"] += 1
            summary["errors"].append(
                {
                    "stage": "staged",
                    "api": api_name,
                    "yaml_in": str(yp),
                    "doc": str(doc_path),
                    "returncode": rc,
                    "stdout": stdout[-8000:],
                    "stderr": stderr[-8000:],
                }
            )


# ---------------------------------------------------------
# summary / report
# ---------------------------------------------------------

def make_summary() -> Dict[str, Any]:
    return {
        "export": {"ok": 0, "skipped": 0, "failed": 0},
        "rank": {"ok": 0, "skipped": 0, "failed": 0, "missing_doc": 0},
        "skeleton": {"ok": 0, "skipped": 0, "failed": 0, "missing_schema": 0, "empty": 0},
        "normalize": {"ok": 0, "skipped": 0, "failed": 0, "unchanged": 0},
        "stagec": {"ok": 0, "skipped": 0, "failed": 0, "missing_doc": 0},
        "staged": {"ok": 0, "skipped": 0, "failed": 0, "missing_doc": 0},
        "errors": [],
    }


def print_summary(summary: Dict[str, Any]) -> None:
    print("\n========== PIPELINE SUMMARY ==========")
    for stage in STAGE_NAMES:
        rec = summary.get(stage, {})
        print(f"[{stage}]")
        for k, v in rec.items():
            print(f"  {k}: {v}")
    print(f"[errors] {len(summary.get('errors', []))}")


# ---------------------------------------------------------
# main
# ---------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Batch pipeline for PyTorch API YAML generation")
    ap.add_argument("--api_list", required=True, help="txt/json/pkl api list")
    ap.add_argument("--docs_root", required=True, help="root dir containing api_name.txt docs recursively")
    ap.add_argument("--work_dir", required=True, help="working directory for all intermediate/final outputs")

    ap.add_argument("--scripts_dir", default=str(THIS_DIR), help="directory containing your stage scripts")
    ap.add_argument("--stages", default="all", help="all or comma-separated: export,rank,skeleton,normalize,stagec,staged")

    ap.add_argument("--overwrite", action="store_true", help="overwrite existing outputs instead of skipping")
    ap.add_argument("--strict", action="store_true", help="exit non-zero if any error occurs")
    ap.add_argument("--limit", type=int, default=0, help="only process first N apis (0 means all)")

    ap.add_argument("--with_evidence", action="store_true", help="include evidence in rank json")
    ap.add_argument("--no_focus", action="store_true", help="disable focused shape section extraction in rank stage")
    ap.add_argument("--merge_rank", action="store_true", help="merge into existing rank file if exists")

    ap.add_argument("--model", default="gpt-5-codex")
    ap.add_argument("--max_doc_chars", type=int, default=80000)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_tokens_stage_c", type=int, default=2000)
    ap.add_argument("--max_tokens_stage_d", type=int, default=1200)
    ap.add_argument("--max_retries", type=int, default=2)
    ap.add_argument("--fail_on_invalid", action="store_true")
    ap.add_argument("--keep_variant_constraints_by_rank", action="store_true")

    args = ap.parse_args()

    stages = normalize_stages(args.stages)
    scripts_dir = Path(args.scripts_dir).resolve()
    docs_root = Path(args.docs_root).resolve()
    work_dir = Path(args.work_dir).resolve()

    if not docs_root.exists():
        raise SystemExit(f"docs_root not found: {docs_root}")

    ensure_dir(work_dir)

    schema_dir = ensure_dir(work_dir / "01_schema_json")
    rank_dir = ensure_dir(work_dir / "02_rank_hints")
    skeleton_dir = ensure_dir(work_dir / "03_yaml_skeleton")
    normalized_dir = ensure_dir(work_dir / "04_yaml_normalized")
    stagec_dir = ensure_dir(work_dir / "05_stagec_multirank")
    final_dir = ensure_dir(work_dir / "06_final_yaml")
    manifest_dir = ensure_dir(work_dir / "00_manifest")

    # 1) load api list
    api_list = load_api_list(args.api_list)
    if args.limit and args.limit > 0:
        api_list = api_list[: args.limit]

    print(f"[+] loaded {len(api_list)} apis")

    # 2) index docs
    print(f"[+] indexing docs under: {docs_root}")
    doc_index = collect_txt_index(docs_root)

    api_to_doc: Dict[str, Path] = {}
    duplicate_doc_hits: Dict[str, List[str]] = {}
    missing_docs: List[str] = []

    for api in api_list:
        best, matches = choose_doc_for_api(api, doc_index)
        if best is None:
            missing_docs.append(api)
        else:
            api_to_doc[api] = best
            if len(matches) > 1:
                duplicate_doc_hits[api] = [str(x) for x in matches]

    json_dump(manifest_dir / "api_to_doc.json", {k: str(v) for k, v in api_to_doc.items()})
    json_dump(manifest_dir / "missing_docs.json", missing_docs)
    json_dump(manifest_dir / "duplicate_doc_hits.json", duplicate_doc_hits)

    print(f"[+] docs matched: {len(api_to_doc)} / {len(api_list)}")
    if missing_docs:
        print(f"[!] missing docs for {len(missing_docs)} apis")

    summary = make_summary()
    summary["meta"] = {
        "api_count": len(api_list),
        "docs_matched": len(api_to_doc),
        "docs_missing": len(missing_docs),
        "work_dir": str(work_dir),
        "scripts_dir": str(scripts_dir),
        "stages": stages,
    }

    # 3) run stages
    try:
        if "export" in stages:
            print("\n[Stage export] exporting schema json ...")
            stage_export(api_list, schema_dir, overwrite=args.overwrite, summary=summary)

        if "rank" in stages:
            print("\n[Stage rank] extracting rank hints from docs ...")
            stage_rank(
                api_list=api_list,
                api_to_doc=api_to_doc,
                rank_dir=rank_dir,
                overwrite=args.overwrite,
                with_evidence=bool(args.with_evidence),
                focus_sections=(not args.no_focus),
                merge_rank=bool(args.merge_rank),
                summary=summary,
            )

        if "skeleton" in stages:
            print("\n[Stage skeleton] generating yaml skeletons ...")
            stage_skeleton(
                api_list=api_list,
                schema_dir=schema_dir,
                skeleton_dir=skeleton_dir,
                rank_dir=rank_dir,
                overwrite=args.overwrite,
                summary=summary,
            )

        if "normalize" in stages:
            print("\n[Stage normalize] normalizing yaml skeletons ...")
            stage_normalize(
                skeleton_dir=skeleton_dir,
                normalized_dir=normalized_dir,
                overwrite=args.overwrite,
                summary=summary,
            )

        if "stagec" in stages:
            print("\n[Stage stagec] running LLM Stage-C patches ...")
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is required for Stage-C")
            stage_stagec(
                scripts_dir=scripts_dir,
                api_to_doc=api_to_doc,
                normalized_dir=normalized_dir,
                stagec_dir=stagec_dir,
                overwrite=args.overwrite,
                model=args.model,
                max_doc_chars=args.max_doc_chars,
                temperature=args.temperature,
                max_tokens_stage_c=args.max_tokens_stage_c,
                max_retries=args.max_retries,
                fail_on_invalid=bool(args.fail_on_invalid),
                keep_variant_constraints_by_rank=bool(args.keep_variant_constraints_by_rank),
                summary=summary,
            )

        if "staged" in stages:
            print("\n[Stage staged] running LLM Stage-D constraints patch ...")
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is required for Stage-D")
            stage_staged(
                scripts_dir=scripts_dir,
                api_to_doc=api_to_doc,
                stagec_dir=stagec_dir,
                final_dir=final_dir,
                overwrite=args.overwrite,
                model=args.model,
                max_doc_chars=args.max_doc_chars,
                temperature=args.temperature,
                max_tokens_stage_d=args.max_tokens_stage_d,
                max_retries=args.max_retries,
                fail_on_invalid=bool(args.fail_on_invalid),
                summary=summary,
            )

    finally:
        json_dump(work_dir / "pipeline_summary.json", summary)
        print_summary(summary)
        print(f"\n[+] summary written to: {work_dir / 'pipeline_summary.json'}")

    if args.strict and summary.get("errors"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()