import os
import json

# 输入文件夹
input_folder = "/root/fuzz_output_experiment"
# 输出文件
json_output = "/root/screen/auto_harness_experiment.json"
txt_output = "/root/apis_experiment.txt"

harness_list = []
api_names = []

# 遍历文件夹及子文件夹
for root, dirs, files in os.walk(input_folder):
    for file in files:
        if file.startswith("llm.") and file.endswith(".py"):
            # 提取api名：去掉前面的llm.和后面的.py
            api_name = file[len("llm."):-len(".py")]
            file_path = os.path.join(root, file)

            harness_list.append({
                "harness_id": api_name,
                "harness_path": file_path,
                "profiles": []
            })
            api_names.append(api_name)

# 写json文件
with open(json_output, "w", encoding="utf-8") as f_json:
    json.dump(harness_list, f_json, indent=4, ensure_ascii=False)

# 写txt文件
with open(txt_output, "w", encoding="utf-8") as f_txt:
    for name in api_names:
        f_txt.write(name + "\n")

print(f"生成完成: {json_output} 和 {txt_output}")