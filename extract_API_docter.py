import os
import argparse


def extract_api_names(input_dir, output_txt):
    api_names = []

    for name in os.listdir(input_dir):
        full_path = os.path.join(input_dir, name)

        # 只处理子文件夹，并且名字以 .yaml_workdir 结尾
        if os.path.isdir(full_path) and name.endswith(".yaml_workdir"):
            api_name = name[:-len(".yaml_workdir")]
            api_names.append(api_name)

    api_names.sort()

    with open(output_txt, "w", encoding="utf-8") as f:
        for api_name in api_names:
            f.write(api_name + "\n")

    print(f"提取完成，共找到 {len(api_names)} 个 API")
    print(f"结果已写入：{output_txt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="从 *.yaml_workdir 子文件夹名中提取 API 名称并写入 txt"
    )

    parser.add_argument(
        "input_dir",
        help="包含 *.yaml_workdir 子文件夹的目录路径"
    )

    parser.add_argument(
        "-o",
        "--output",
        default="api_list_docter.txt",
        help="输出 txt 文件路径，默认 api_list_docter.txt"
    )

    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"错误：目录不存在：{args.input_dir}")
        exit(1)

    extract_api_names(args.input_dir, args.output)