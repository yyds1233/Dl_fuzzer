# 文件名：extract_low_dcov_with_value.py
import re
import sys

txt_file = sys.argv[1]  # 日志文件路径

pattern_dcov = re.compile(r"Δcov=(\d+)")
pattern_harness = re.compile(r"harness=([\w\.\_]+)")

with open(txt_file, "r", encoding="utf-8") as f:
    for line in f:
        dcov_match = pattern_dcov.search(line)
        harness_match = pattern_harness.search(line)
        if dcov_match and harness_match:
            dcov_value = int(dcov_match.group(1))
            harness_name = harness_match.group(1)
            if dcov_value < 100:
                print(f"harness={harness_name}, Δcov={dcov_value}")