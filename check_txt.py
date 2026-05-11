import os

api_txt_file = "candidate_api.txt"      # 包含 API 名称的 TXT 文件
folder_path = "/root/api_txt"    # 文件夹路径
output_file = "existing_apis.txt"  # 输出文件

# 读取 API 名称列表
with open(api_txt_file, "r", encoding="utf-8") as f:
    api_list = [line.strip() for line in f if line.strip()]

# 检查哪些 API 的 .txt 文件存在于文件夹中
existing_apis = []
for api in api_list:
    expected_file = os.path.join(folder_path, f"{api}.txt")
    if os.path.isfile(expected_file):
        existing_apis.append(api)

# 写入输出文件
with open(output_file, "w", encoding="utf-8") as f:
    for api in existing_apis:
        f.write(api + "\n")

print(f"处理完成，共 {len(existing_apis)} 个 API 文件存在，已保存到 {output_file}")