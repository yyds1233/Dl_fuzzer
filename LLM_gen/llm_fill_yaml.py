#!/usr/bin/env python3
# llm_fill_yaml.py
import os
import argparse
import json
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field

from openai import OpenAI
from llm_prompts import YAML_FILL_SYSTEM_PROMPT


# ---------- Structured output schema (Pydantic) ----------
class YamlFillResult(BaseModel):
    updated_yaml: str = Field(..., description="补全后的 YAML（完整文本），必须可被 YAML 解析")
    changes: List[str] = Field(default_factory=list, description="本次做了哪些补全/修改（要点列表）")
    confidence: float = Field(ge=0.0, le=1.0, description="对补全质量的置信度 0~1")
    warnings: List[str] = Field(default_factory=list, description="可能不确定/需要人工确认的点")


# ---------- helpers ----------
def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")


def read_yaml_text(p: Path) -> str:
    # 保留原始 yaml 文本（因为你可能有注释/顺序/风格要求）
    return read_text(p)


def safe_truncate(text: str, max_chars: int) -> str:
    """文档太长时截断：保留头尾，避免只给头部导致缺信息。"""
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-max_chars // 2 :]
    return head + "\n\n[...TRUNCATED...]\n\n" + tail


def validate_yaml(yaml_text: str) -> Optional[str]:
    """返回 None 表示通过；否则返回错误字符串。"""
    try:
        yaml.safe_load(yaml_text)
        return None
    except Exception as e:
        return str(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc_txt", required=True, help="官方文档 txt 文件路径")
    ap.add_argument("--yaml_in", required=True, help="待补全的 yaml 骨架文件路径")
    ap.add_argument("--yaml_out", required=True, help="输出补全后的 yaml 路径")
    ap.add_argument("--model", default="gpt-4o-2024-08-06", help="模型名（你可替换成你账户可用的）")
    ap.add_argument("--max_doc_chars", type=int, default=80000, help="doc.txt 太长时截断的最大字符数")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_output_tokens", type=int, default=3000)
    args = ap.parse_args()

    doc_path = Path(args.doc_txt).resolve()
    yaml_in_path = Path(args.yaml_in).resolve()
    yaml_out_path = Path(args.yaml_out).resolve()
    meta_out_path = yaml_out_path.with_suffix(yaml_out_path.suffix + ".meta.json")

    doc_text = safe_truncate(read_text(doc_path), args.max_doc_chars)
    yaml_skeleton = read_yaml_text(yaml_in_path)

    # -------- prompt（先给一个可运行版本；你后续再迭代提示词就行）--------
    system_prompt = YAML_FILL_SYSTEM_PROMPT

    user_prompt = (
        "=== OFFICIAL DOCUMENTATION (TXT) ===\n"
        f"{doc_text}\n\n"
        "=== YAML SKELETON (TO BE COMPLETED) ===\n"
        f"{yaml_skeleton}\n\n"
        "Please fill ONLY:\n"
        "1) shape_vars\n"
        "2) constraints\n"
        "\n"
        "Do not change other YAML sections unless strictly necessary for consistency.\n"
        "All constraints must be eval()-safe boolean expressions and must reference only defined variables.\n"
    )

    # client = OpenAI()
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.gpt.ge/v1/")

    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY env var")

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers={"x-foo": "true"},  # 你如果不需要可以删掉
    )

    # 使用 responses.parse + Pydantic 结构化输出（推荐方式）
    # 文档示例：client.responses.parse(..., text_format=YourPydanticModel) :contentReference[oaicite:1]{index=1}
    resp = client.responses.parse(
        model=args.model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        text_format=YamlFillResult,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
    )

    result: YamlFillResult = resp.output_parsed

    # -------- 基本校验：确保返回 YAML 可解析 --------
    err = validate_yaml(result.updated_yaml)
    if err is not None:
        # 不直接失败：把错误写进 meta，并把原输出也落盘，方便你调 prompt
        result.warnings.append(f"YAML parse failed: {err}")

    yaml_out_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_out_path.write_text(result.updated_yaml, encoding="utf-8")

    meta_out = {
        "model": args.model,
        "doc_txt": str(doc_path),
        "yaml_in": str(yaml_in_path),
        "yaml_out": str(yaml_out_path),
        "confidence": result.confidence,
        "changes": result.changes,
        "warnings": result.warnings,
    }
    meta_out_path.write_text(json.dumps(meta_out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[+] wrote: {yaml_out_path}")
    print(f"[+] wrote: {meta_out_path}")
    if result.warnings:
        print("[!] warnings:")
        for w in result.warnings:
            print("   -", w)


if __name__ == "__main__":
    main()
