import os
import sys

root_dir = sys.argv[1]  # 根目录，从命令行传入

# 遍历子目录
for current_root, dirs, files in os.walk(root_dir):
    for d in dirs:
        if d == "crash" or d == "crashes":
            crash_path = os.path.join(current_root, d)
            # 检查 crash 文件夹里是否有文件
            if any(os.path.isfile(os.path.join(crash_path, f)) for f in os.listdir(crash_path)):
                print(crash_path)