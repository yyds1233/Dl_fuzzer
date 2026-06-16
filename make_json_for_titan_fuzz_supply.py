import json
import argparse
import os


def load_api_txt(txt_path):
    """
    读取 txt 中的 API 名称，一行一个
    """
    apis = set()

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            api = line.strip()
            if api:
                apis.add(api)

    return apis


def main(json_path, txt_path, output_json_path, missing_txt_path):
    # 读取 JSON
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON 顶层格式应该是 list，例如：[ {...}, {...} ]")

    # JSON 里的 harness_id
    json_api_set = set()
    for item in data:
        harness_id = item.get("harness_id")
        if harness_id:
            json_api_set.add(harness_id)

    # 统计 JSON 里 API 数量
    print(f"JSON 中 API 数量：{len(json_api_set)}")

    # 读取 txt 里的 API
    txt_api_set = load_api_txt(txt_path)
    print(f"TXT 中 API 数量：{len(txt_api_set)}")

    # 保留 JSON 里有，但是 TXT 里没有的
    filtered_data = []
    removed_count = 0

    for item in data:
        harness_id = item.get("harness_id")

        if harness_id in txt_api_set:
            removed_count += 1
            continue

        filtered_data.append(item)

    # 写出新的 JSON 文件
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=4)

    print(f"从 JSON 中删除的 API 数量：{removed_count}")
    print(f"过滤后 JSON 剩余数量：{len(filtered_data)}")
    print(f"新的 JSON 已写入：{output_json_path}")

    # 找出 TXT 里有，但 JSON 里没有的 API
    txt_has_json_not_has = sorted(txt_api_set - json_api_set)

    with open(missing_txt_path, "w", encoding="utf-8") as f:
        for api in txt_has_json_not_has:
            f.write(api + "\n")

    print(f"TXT 中有但 JSON 中没有的 API 数量：{len(txt_has_json_not_has)}")
    print(f"结果已写入：{missing_txt_path}")

    if txt_has_json_not_has:
        print("\nTXT 中有但 JSON 中没有的 API：")
        for api in txt_has_json_not_has:
            print(api)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="统计 JSON 中 harness_id 数量，并根据 txt 中的 API 过滤 JSON"
    )

    parser.add_argument(
        "json_file",
        help="输入 JSON 文件路径"
    )

    parser.add_argument(
        "txt_file",
        help="输入 API txt 文件路径，一行一个 API"
    )

    parser.add_argument(
        "-o",
        "--output-json",
        default="/root/screen/supply_titan_fuzz.json",
        help="输出的新 JSON 文件路径，默认 filtered_output.json"
    )

    parser.add_argument(
        "-m",
        "--missing-txt",
        default="txt_has_but_json_not_has.txt",
        help="输出 TXT 中有但 JSON 中没有的 API 文件，默认 txt_has_but_json_not_has.txt"
    )

    args = parser.parse_args()

    if not os.path.isfile(args.json_file):
        print(f"错误：JSON 文件不存在：{args.json_file}")
        exit(1)

    if not os.path.isfile(args.txt_file):
        print(f"错误：TXT 文件不存在：{args.txt_file}")
        exit(1)

    main(
        json_path=args.json_file,
        txt_path=args.txt_file,
        output_json_path=args.output_json,
        missing_txt_path=args.missing_txt
    )