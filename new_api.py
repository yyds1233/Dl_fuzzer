import os

folder_path = "/root/fuzz_output_all"  # 文件夹路径
output_file = "final_experiment_apis.txt"     # 输出文件

api_names = []

# 遍历文件夹及子文件夹
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith(".py") and file.startswith("llm."):
            # 去掉前缀 llm. 和后缀 .py
            api_name = file[len("llm."):-len(".py")]
            api_names.append(api_name)

# 去重（可选）
api_names = sorted(set(api_names))

with open(output_file, "w", encoding="utf-8") as f:
    for api in api_names:
        f.write(api + "\n")

print(f"处理完成，共 {len(api_names)} 个 API 名称，已保存到 {output_file}")