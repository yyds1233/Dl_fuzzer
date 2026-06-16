import os
import argparse


def extract_api_names(input_dir, output_txt):
    api_names = []

    for filename in os.listdir(input_dir):
        # 只处理 *_merged.py 文件
        if filename.endswith("_merged.py"):
            api_name = filename[:-len("_merged.py")]
            api_names.append(api_name)

    # 排序，方便查看
    api_names.sort()

    with open(output_txt, "w", encoding="utf-8") as f:
        for api_name in api_names:
            f.write(api_name + "\n")

    print(f"提取完成，共找到 {len(api_names)} 个 API")
    print(f"结果已写入：{output_txt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="从 *_merged.py 文件名中提取 API 名称并写入 txt"
    )

    parser.add_argument(
        "input_dir",
        help="存放 *_merged.py 文件的文件夹路径"
    )

    parser.add_argument(
        "-o",
        "--output",
        default="api_list_titan_fuzz.txt",
        help="输出 txt 文件路径，默认 api_list_titan_fuzz.txt"
    )

    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"错误：文件夹不存在：{args.input_dir}")
        exit(1)

    extract_api_names(args.input_dir, args.output)