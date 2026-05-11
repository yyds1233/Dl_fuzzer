# 文件名：process_api_txt.py

input_file = "DocTer_api_raw.txt"   # 输入文件路径
output_file = "DocTer_api.txt"      # 输出文件路径

with open(input_file, "r", encoding="utf-8") as f:
    text = f.read()

# 按空格拆分内容
items = text.split()

# 去掉 .yaml 后缀
items = [item[:-5] if item.endswith(".yaml") else item for item in items]

# 写入新文件，每行一个
with open(output_file, "w", encoding="utf-8") as f:
    for item in items:
        f.write(item + "\n")

print(f"处理完成，结果已保存到 {output_file}")