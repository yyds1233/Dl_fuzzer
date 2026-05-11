import json

# 需要删除的 harness_id 列表
REMOVE_IDS = [
    "torch.distributed.send",
    "torch.distributed.broadcast",
    "torch.utils.cpp_extension.load_inline",
    "torch.distributed.all_gather",
    "torch.hub.set_dir",
    "torch.utils.cpp_extension.include_paths",
    "torch.hub.list",
    "torch.distributed.init_process_group",
    "torch.distributed.barrier",
    "torch.distributed.isend",
    "torch.distributed.new_group",
    "torch.distributed.irecv",
    "torch.utils.cpp_extension.load",
    "torch.nn.conv3d",
    "torch.autograd.profiler.load_nvprof",
    "torch.distributed.all_reduce",
    "torch.hub.download_url_to_file"
]

# 原 JSON 文件
INPUT_JSON = "/root/screen/auto_harness_all.json"
# 输出新 JSON 文件
OUTPUT_JSON = "/root/screen/auto_harness_all_filtered.json"

# 读取原 JSON
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

# 过滤掉指定 harness_id
filtered_data = [item for item in data if item["harness_id"] not in REMOVE_IDS]

# 写入新 JSON
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(filtered_data, f, indent=4, ensure_ascii=False)

print(f"处理完成，去掉指定 harness_id 后生成 {OUTPUT_JSON}，原数量 {len(data)}，新数量 {len(filtered_data)}")