
file_a = "new_candidate_api.txt"   # a 中的 TXT 文件
file_b = "common_values.txt"   # b 中的 TXT 文件
output_file = "candidate_api.txt"  # 输出文件

# 读取 a.txt 内容
with open(file_a, "r", encoding="utf-8") as f:
    a_values = set(line.strip() for line in f if line.strip())

# 读取 b.txt 内容
with open(file_b, "r", encoding="utf-8") as f:
    b_values = set(line.strip() for line in f if line.strip())

# 找出 a 中不在 b 中的值
diff_values = sorted(a_values - b_values)

# 写入新文件
with open(output_file, "w", encoding="utf-8") as f:
    for val in diff_values:
        f.write(val + "\n")

print(f"处理完成，共 {len(diff_values)} 个值，已保存到 {output_file}")