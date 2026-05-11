# 文件名：compare_txt.py

file1 = "final_experiment_apis.txt"  # 第一个 TXT 文件
file2 = "titan_fuzz_api.txt"  # 第二个 TXT 文件
output_file = "api4titan_fuzz.txt"  # 输出文件（可选）

# 读取文件内容，去掉每行的空格和换行
with open(file1, "r", encoding="utf-8") as f:
    lines1 = set(line.strip() for line in f if line.strip())

with open(file2, "r", encoding="utf-8") as f:
    lines2 = set(line.strip() for line in f if line.strip())

# 找出交集
common = sorted(lines1 & lines2)

# 找出 file1 中有但 file2 中没有的
only_in_file1 = sorted(lines1 - lines2)

# 输出到屏幕
print("两个文件中相同的值：")
for item in common:
    print(item)

print("\nfile1 中有，但 file2 中没有的值：")
for item in only_in_file1:
    print(item)

# 保存到新文件
with open(output_file, "w", encoding="utf-8") as f:
    # 先写相同的值
    for item in common:
        f.write(item + "\n")
    # 再写 file1 独有的值，并加标记
    for item in only_in_file1:
        f.write(item + "  # only in file1\n")

print(f"\n处理完成，共 {len(common)} 个相同值，{len(only_in_file1)} 个仅在 file1 中，已保存到 {output_file}")