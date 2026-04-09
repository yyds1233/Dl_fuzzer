#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path


KEYWORDS = ["__RANK_TODO__", "TODO_SHAPE"]


def contains_keywords(file_path: Path, keywords: list[str]) -> bool:
    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            content = file_path.read_text(encoding="utf-8-sig")
        except Exception as e:
            print(f"[WARN] 跳过无法读取的文件: {file_path} ({e})", file=sys.stderr)
            return False
    except Exception as e:
        print(f"[WARN] 跳过无法读取的文件: {file_path} ({e})", file=sys.stderr)
        return False

    return any(keyword in content for keyword in keywords)


def find_yaml_files(folder: Path):
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}:
            yield path


def main():
    if len(sys.argv) != 2:
        print(f"用法: python {Path(sys.argv[0]).name} <目录路径>")
        sys.exit(1)

    folder = Path(sys.argv[1])

    if not folder.exists():
        print(f"[ERROR] 目录不存在: {folder}", file=sys.stderr)
        sys.exit(1)

    if not folder.is_dir():
        print(f"[ERROR] 不是目录: {folder}", file=sys.stderr)
        sys.exit(1)

    matched = []

    for yaml_file in find_yaml_files(folder):
        if contains_keywords(yaml_file, KEYWORDS):
            matched.append(yaml_file)

    if matched:
        for file_path in matched:
            print(file_path.name)
    else:
        print("未找到包含目标关键词的 YAML 文件。")


if __name__ == "__main__":
    main()