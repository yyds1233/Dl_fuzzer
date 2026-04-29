import os
import json
import argparse
from collections import OrderedDict
from pathlib import Path

try:
    import yaml
except ImportError:
    raise ImportError(
        "缺少 PyYAML，请先安装：pip install pyyaml"
    )


def api_name_to_txt_path(api_name: str, txt_root: str) -> str:
    """
    把 api_name 转成 txt 路径：

    torch.addbmm -> /root/torch_api_txt/torch_addbmm.txt
    tf.argsort   -> /root/tf_api_txt/tf_argsort.txt
    """
    txt_name = api_name + ".txt"
    return os.path.join(txt_root, txt_name)


def collect_yaml_api_info(input_dirs, output_json, txt_root="/root/torch_api_txt"):
    """
    遍历多个输入目录，收集所有 yaml/yml 文件。

    如果多个 yaml 对应同一个 api_name，则合并到同一个 item 的 yaml 列表里。

    输出格式：

    [
      {
        "api": "torch.addbmm",
        "yaml": [
          "/path/to/torch.addbmm.default.yaml",
          "/path/to/torch.addbmm.out.yaml"
        ],
        "txt": "/root/torch_api_txt/torch_addbmm.txt"
      }
    ]
    """

    # 用 OrderedDict 保持第一次遇到 api 的顺序
    api_map = OrderedDict()

    total_yaml_count = 0
    valid_yaml_count = 0
    skipped_count = 0

    for input_dir in input_dirs:
        input_dir = os.path.abspath(input_dir)

        if not os.path.isdir(input_dir):
            print(f"[跳过] 文件夹不存在：{input_dir}")
            skipped_count += 1
            continue

        for root, _, files in os.walk(input_dir):
            for filename in sorted(files):
                if not (filename.endswith(".yaml") or filename.endswith(".yml")):
                    continue

                total_yaml_count += 1
                yaml_path = os.path.abspath(os.path.join(root, filename))

                try:
                    with open(yaml_path, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f)
                except Exception as e:
                    print(f"[读取失败] {yaml_path}: {e}")
                    skipped_count += 1
                    continue

                if not isinstance(data, dict):
                    print(f"[跳过] YAML 顶层不是 dict：{yaml_path}")
                    skipped_count += 1
                    continue

                api_name = data.get("api_name")

                if not api_name:
                    print(f"[跳过] 没有 api_name 字段：{yaml_path}")
                    skipped_count += 1
                    continue

                valid_yaml_count += 1

                if api_name not in api_map:
                    api_map[api_name] = {
                        "api": api_name,
                        "yaml": [],
                        "txt": api_name_to_txt_path(api_name, txt_root),
                    }

                api_map[api_name]["yaml"].append(yaml_path)

    results = list(api_map.values())

    # 每个 api 内部的 yaml 路径排序，方便结果稳定
    for item in results:
        item["yaml"] = sorted(item["yaml"])

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("完成")
    print(f"扫描到 YAML 文件数：{total_yaml_count}")
    print(f"有效 YAML 文件数：{valid_yaml_count}")
    print(f"跳过数量：{skipped_count}")
    print(f"唯一 API 数量：{len(results)}")
    print(f"输出文件：{os.path.abspath(output_json)}")


def main():
    parser = argparse.ArgumentParser(
        description="遍历文件夹下所有 YAML 文件，按 api_name 合并，并生成 JSON"
    )

    parser.add_argument(
        "input_dirs",
        nargs="+",
        help="一个或多个输入文件夹"
    )

    parser.add_argument(
        "-o",
        "--output",
        default="api_yaml_txt_mapping.json",
        help="输出 JSON 文件路径，默认 api_yaml_txt_mapping.json"
    )

    parser.add_argument(
        "--txt-root",
        default="/root/torch_api_txt",
        help="txt 文件所在根目录，默认 /root/torch_api_txt"
    )

    args = parser.parse_args()

    collect_yaml_api_info(
        input_dirs=args.input_dirs,
        output_json=args.output,
        txt_root=args.txt_root,
    )


if __name__ == "__main__":
    main()