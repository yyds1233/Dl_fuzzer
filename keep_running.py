import os
import sys
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# ==================== 配置项 ====================
# Harness 脚本所在的文件夹路径
HARNESS_DIR = "/root/fuzz_output_all"  # 请修改为实际的 harness 所在目录

# 结果存储基础路径
CORPUS_BASE_DIR = "/root/fuzz_output_result/corpus"
LOG_BASE_DIR = "/root/fuzz_output_result/logs"
CRASH_BASE_DIR = "/root/fuzz_output_result/crashes"

# 运行参数
FUZZ_TIME_MINUTES = 1.5  # 每个 harness 运行的时间（分钟）
MAX_WORKERS = 64        # 并行执行的最大进程数
# ===============================================


def run_single_harness(harness_path, api_name):
    """
    执行单个 harness 任务
    """
    corpus_dir = os.path.join(CORPUS_BASE_DIR, api_name)
    crash_dir = os.path.join(CRASH_BASE_DIR, api_name)
    log_path = os.path.join(LOG_BASE_DIR, f"{api_name}.log")

    # 确保所需路径存在
    os.makedirs(corpus_dir, exist_ok=True)
    os.makedirs(crash_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    duration_seconds = FUZZ_TIME_MINUTES * 60

    # 组装完整的 Fuzzing 命令
    # 结合了 libFuzzer 的原生 -max_total_time，便于它安全保存语料
    cmd = [
        "python3",
        harness_path,
        corpus_dir,
        f"-artifact_prefix={crash_dir}/",
        "-ignore_timeouts=1",
        "-rss_limit_mb=8192",
        "-use_value_profile=1",
        "-entropic=1",
        f"-max_total_time={duration_seconds}"
    ]

    print(f"[START] 执行: {api_name} | 日志存放在: {log_path}")

    # 给 Python 进程额外的 10 秒缓冲时间，防止因某些原因挂起
    sub_timeout = duration_seconds + 10

    try:
        with open(log_path, "w") as log_file:
            # 执行子进程并将 stdout, stderr 全部重定向到对应的日志文件
            subprocess.run(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=sub_timeout,
                text=True,
                check=False
            )
        print(f"[SUCCESS] {api_name} 执行完毕。")
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {api_name} 超出预设时间 ({FUZZ_TIME_MINUTES} 分钟)，已被脚本终止。")
    except Exception as e:
        print(f"[ERROR] {api_name} 执行过程中发生异常: {e}")


def main():
    if not os.path.isdir(HARNESS_DIR):
        print(f"错误: 配置的 HARNESS_DIR '{HARNESS_DIR}' 不是一个有效的目录。")
        sys.exit(1)

    # 扫描指定文件夹，寻找符合 'llm.*.py' 命名模式的文件
    tasks = []
    for filename in os.listdir(HARNESS_DIR):
        if filename.startswith("llm.") and filename.endswith(".py"):
            harness_path = os.path.join(HARNESS_DIR, filename)
            
            # 提取 API 名字：例如 'llm.tf.argsort.py' -> 'tf.argsort'
            api_name = filename[4:-3]
            
            if api_name:
                tasks.append((harness_path, api_name))

    if not tasks:
        print(f"在 {HARNESS_DIR} 下没有找到符合命名规则（llm.<api_name>.py）的脚本。")
        sys.exit(0)

    print(f"共发现 {len(tasks)} 个任务，将使用最大进程数 {MAX_WORKERS} 并发执行。")
    print("-------------------------------------------------------------------")

    # 使用多进程池并发调度
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(run_single_harness, h_path, api): api 
            for h_path, api in tasks
        }

        for future in as_completed(futures):
            api = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"[CRITICAL] {api} 在线程池调度中抛出未捕获的错误: {e}")

    print("-------------------------------------------------------------------")
    print("所有 Fuzzing 任务处理完毕！")


if __name__ == "__main__":
    main()