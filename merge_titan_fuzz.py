import os
from collections import defaultdict

folder_path = "/root/seed_torch"  # 例如 "C:/projects/apis"
output_folder = "merged_scripts_titan_fuzz"
os.makedirs(output_folder, exist_ok=True)

# 分组
api_files = defaultdict(list)
for f in os.listdir(folder_path):
    if f.endswith(".py") and "_seed" in f:
        api_name = f.split("_seed")[0]
        api_files[api_name].append(f)

# 合并
for api, files in api_files.items():
    # 修改为 PyTorch，并加上 import os
    merged_lines = ["import os\n", "import torch\n", "import numpy as np\n\n"]
    counter = 0
    for f in sorted(files):
        with open(os.path.join(folder_path, f), "r", encoding="utf-8") as ff:
            lines = ff.readlines()
        # 去掉旧的 import tensorflow 和 numpy 行
        content_lines = [l for l in lines if not l.startswith("import tensorflow") and not l.startswith("import numpy")]
        # 可选：为每个 seed 生成唯一变量名，防止冲突
        content_lines = [l.replace("input_data", f"input_data_{counter}").replace("output_data", f"output_data_{counter}") for l in content_lines]
        merged_lines.extend(content_lines)
        merged_lines.append("\n")  # 分隔不同 seed
        counter += 1

    merged_file_path = os.path.join(output_folder, f"{api}_merged.py")
    with open(merged_file_path, "w", encoding="utf-8") as f:
        f.writelines(merged_lines)