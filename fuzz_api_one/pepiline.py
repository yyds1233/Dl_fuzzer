import argparse
import json
import os
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


DEFAULT_ATHERIS_DOC = "/root/atheris-doc/atheris_readme.txt"
DEFAULT_MODEL = "gpt-5.4"
DEFAULT_BASE_URL = "https://api.gpt.ge/v1/"


def to_text(x):
    if x is None:
        return ""
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    return str(x)


def api_to_output_path(api_name: str, out_dir: str) -> str:
    """
    torch.argsort -> /root/fuzz_output_experiment/llm.torch.argsort.py
    """
    filename = f"llm.{api_name}.py"
    return os.path.join(out_dir, filename)


def normalize_yaml_paths(yaml_field):
    """
    兼容两种格式：

    老格式：
      "yaml": "/path/to/one.yaml"

    新格式：
      "yaml": ["/path/a.yaml", "/path/b.yaml"]
    """
    if isinstance(yaml_field, str):
        return [yaml_field]

    if isinstance(yaml_field, list):
        return [x for x in yaml_field if isinstance(x, str) and x.strip()]

    return []


def shell_join(cmd):
    return " ".join(shlex.quote(str(x)) for x in cmd)


def run_one_task(
    item,
    llm_script,
    atheris_doc,
    out_dir,
    model,
    base_url,
    timeout,
    skip_existing,
    api_mode,
    repair_attempts,
    save_raw_response,
    save_prompt,
):
    api = item.get("api")
    yaml_paths = normalize_yaml_paths(item.get("yaml"))
    txt_path = item.get("txt")

    out_path = api_to_output_path(api, out_dir) if api else None

    if not api or not yaml_paths or not txt_path:
        return {
            "api": api or "UNKNOWN",
            "status": "FAILED",
            "reason": "JSON item 缺少 api/yaml/txt 字段，或 yaml 为空",
            "cmd": None,
            "stdout": "",
            "stderr": "",
            "returncode": None,
            "out": out_path,
            "yaml_count": len(yaml_paths),
            "yaml_paths": yaml_paths,
        }

    if skip_existing and os.path.exists(out_path):
        return {
            "api": api,
            "status": "SKIPPED",
            "reason": f"输出文件已存在：{out_path}",
            "cmd": None,
            "stdout": "",
            "stderr": "",
            "returncode": 0,
            "out": out_path,
            "yaml_count": len(yaml_paths),
            "yaml_paths": yaml_paths,
        }

    missing_yamls = [p for p in yaml_paths if not os.path.exists(p)]
    if missing_yamls:
        return {
            "api": api,
            "status": "FAILED",
            "reason": "以下 yaml 文件不存在：" + ", ".join(missing_yamls),
            "cmd": None,
            "stdout": "",
            "stderr": "",
            "returncode": None,
            "out": out_path,
            "yaml_count": len(yaml_paths),
            "yaml_paths": yaml_paths,
        }

    if not os.path.exists(txt_path):
        return {
            "api": api,
            "status": "FAILED",
            "reason": f"api txt 文件不存在：{txt_path}",
            "cmd": None,
            "stdout": "",
            "stderr": "",
            "returncode": None,
            "out": out_path,
            "yaml_count": len(yaml_paths),
            "yaml_paths": yaml_paths,
        }

    if not os.path.exists(atheris_doc):
        return {
            "api": api,
            "status": "FAILED",
            "reason": f"atheris doc 文件不存在：{atheris_doc}",
            "cmd": None,
            "stdout": "",
            "stderr": "",
            "returncode": None,
            "out": out_path,
            "yaml_count": len(yaml_paths),
            "yaml_paths": yaml_paths,
        }

    cmd = [
        "python3",
        llm_script,
    ]

    for yaml_path in yaml_paths:
        cmd.extend(["--yaml", yaml_path])

    cmd.extend([
        "--api-name",
        api,
        "--api-txt",
        txt_path,
        "--atheris-doc",
        atheris_doc,
        "--out",
        out_path,
        "--model",
        model,
        "--base-url",
        base_url,
        "--api-mode",
        api_mode,
        "--repair-attempts",
        str(repair_attempts),
    ])

    if save_raw_response:
        cmd.append("--save-raw-response")

    if save_prompt:
        cmd.append("--save-prompt")

    start_time = datetime.now()

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )

        end_time = datetime.now()
        status = "SUCCESS" if proc.returncode == 0 else "FAILED"

        return {
            "api": api,
            "status": status,
            "reason": "",
            "cmd": shell_join(cmd),
            "stdout": to_text(proc.stdout),
            "stderr": to_text(proc.stderr),
            "returncode": proc.returncode,
            "out": out_path,
            "yaml_count": len(yaml_paths),
            "yaml_paths": yaml_paths,
            "txt": txt_path,
            "start_time": start_time.isoformat(timespec="seconds"),
            "end_time": end_time.isoformat(timespec="seconds"),
            "duration_seconds": round((end_time - start_time).total_seconds(), 2),
        }

    except subprocess.TimeoutExpired as e:
        end_time = datetime.now()

        return {
            "api": api,
            "status": "FAILED",
            "reason": f"执行超时，超过 {timeout} 秒",
            "cmd": shell_join(cmd),
            "stdout": to_text(e.stdout),
            "stderr": to_text(e.stderr),
            "returncode": None,
            "out": out_path,
            "yaml_count": len(yaml_paths),
            "yaml_paths": yaml_paths,
            "txt": txt_path,
            "start_time": start_time.isoformat(timespec="seconds"),
            "end_time": end_time.isoformat(timespec="seconds"),
            "duration_seconds": round((end_time - start_time).total_seconds(), 2),
        }

    except Exception as e:
        end_time = datetime.now()

        return {
            "api": api,
            "status": "FAILED",
            "reason": repr(e),
            "cmd": shell_join(cmd),
            "stdout": "",
            "stderr": "",
            "returncode": None,
            "out": out_path,
            "yaml_count": len(yaml_paths),
            "yaml_paths": yaml_paths,
            "txt": txt_path,
            "start_time": start_time.isoformat(timespec="seconds"),
            "end_time": end_time.isoformat(timespec="seconds"),
            "duration_seconds": round((end_time - start_time).total_seconds(), 2),
        }


def write_log_entry(log_f, result):
    log_f.write("=" * 100 + "\n")
    log_f.write(f"API: {result.get('api')}\n")
    log_f.write(f"STATUS: {result.get('status')}\n")
    log_f.write(f"OUT: {result.get('out')}\n")
    log_f.write(f"TXT: {result.get('txt')}\n")
    log_f.write(f"YAML_COUNT: {result.get('yaml_count')}\n")
    log_f.write(f"RETURNCODE: {result.get('returncode')}\n")

    yaml_paths = result.get("yaml_paths") or []
    if yaml_paths:
        log_f.write("\nYAML_PATHS:\n")
        for p in yaml_paths:
            log_f.write(f"  - {p}\n")

    if result.get("reason"):
        log_f.write(f"\nREASON: {result.get('reason')}\n")

    if result.get("start_time"):
        log_f.write(f"START: {result.get('start_time')}\n")

    if result.get("end_time"):
        log_f.write(f"END: {result.get('end_time')}\n")

    if result.get("duration_seconds") is not None:
        log_f.write(f"DURATION_SECONDS: {result.get('duration_seconds')}\n")

    if result.get("cmd"):
        log_f.write("\nCMD:\n")
        log_f.write(to_text(result["cmd"]) + "\n")

    log_f.write("\nSTDOUT:\n")
    log_f.write(to_text(result.get("stdout")))
    log_f.write("\n")

    log_f.write("\nSTDERR:\n")
    log_f.write(to_text(result.get("stderr")))
    log_f.write("\n\n")

    log_f.flush()


def main():
    parser = argparse.ArgumentParser(
        description="批量调用 llm_gen_harness.py，根据 JSON 里的多个 yaml/txt 生成 PyTorch harness"
    )

    parser.add_argument(
        "--json",
        required=True,
        help="输入 JSON 文件，格式为 [{'api': ..., 'yaml': [...], 'txt': ...}]",
    )

    parser.add_argument(
        "--llm-script",
        default="llm_gen_harness.py",
        help="llm_gen_harness.py 的路径",
    )

    parser.add_argument(
        "--atheris-doc",
        default=DEFAULT_ATHERIS_DOC,
        help=f"atheris doc 路径，默认 {DEFAULT_ATHERIS_DOC}",
    )

    parser.add_argument(
        "--out-dir",
        default="/root/fuzz_output_experiment",
        help="生成 harness 的输出目录",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"模型名，默认 {DEFAULT_MODEL}",
    )

    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"base url，默认 {DEFAULT_BASE_URL}",
    )

    parser.add_argument(
        "--api-mode",
        choices=["responses", "chat", "auto"],
        default="auto",
        help="LLM API mode，默认 auto",
    )

    parser.add_argument(
        "--repair-attempts",
        type=int,
        default=1,
        help="静态验证失败后的修复次数，默认 1",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="并发数量，建议 1-2，默认 1",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="单个任务超时时间，单位秒，默认 1800",
    )

    parser.add_argument(
        "--log",
        default="batch_gen_harness_torch.log",
        help="日志文件路径",
    )

    parser.add_argument(
        "--summary-json",
        default="batch_gen_harness_torch_summary.json",
        help="结果汇总 JSON",
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="如果输出 harness 已存在，则跳过",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只处理前 N 条，方便测试",
    )

    parser.add_argument(
        "--save-raw-response",
        action="store_true",
        help="让 llm_gen_harness.py 保存原始模型响应",
    )

    parser.add_argument(
        "--save-prompt",
        action="store_true",
        help="让 llm_gen_harness.py 保存最终 prompt",
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.json, "r", encoding="utf-8") as f:
        items = json.load(f)

    if not isinstance(items, list):
        raise ValueError("输入 JSON 顶层必须是 list")

    if args.limit is not None:
        items = items[: args.limit]

    workers = max(1, args.workers)

    print(f"共读取 {len(items)} 条任务")
    print(f"并发数：{workers}")
    print(f"输出目录：{args.out_dir}")
    print(f"日志文件：{args.log}")

    results = []

    with open(args.log, "w", encoding="utf-8") as log_f:
        log_f.write(f"BATCH START: {datetime.now().isoformat(timespec='seconds')}\n")
        log_f.write(f"TASKS: {len(items)}\n")
        log_f.write(f"WORKERS: {workers}\n")
        log_f.write(f"OUT_DIR: {args.out_dir}\n")
        log_f.write(f"LLM_SCRIPT: {args.llm_script}\n")
        log_f.write(f"MODEL: {args.model}\n")
        log_f.write(f"BASE_URL: {args.base_url}\n")
        log_f.write(f"API_MODE: {args.api_mode}\n\n")
        log_f.flush()

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    run_one_task,
                    item,
                    args.llm_script,
                    args.atheris_doc,
                    args.out_dir,
                    args.model,
                    args.base_url,
                    args.timeout,
                    args.skip_existing,
                    args.api_mode,
                    args.repair_attempts,
                    args.save_raw_response,
                    args.save_prompt,
                )
                for item in items
            ]

            for idx, future in enumerate(as_completed(futures), 1):
                result = future.result()
                results.append(result)

                write_log_entry(log_f, result)

                print(
                    f"[{idx}/{len(items)}] "
                    f"{result.get('status')} "
                    f"{result.get('api')} "
                    f"yaml_count={result.get('yaml_count')} "
                    f"-> {result.get('out')}"
                )

        success_count = sum(1 for r in results if r["status"] == "SUCCESS")
        failed_count = sum(1 for r in results if r["status"] == "FAILED")
        skipped_count = sum(1 for r in results if r["status"] == "SKIPPED")

        log_f.write("=" * 100 + "\n")
        log_f.write("SUMMARY\n")
        log_f.write(f"SUCCESS: {success_count}\n")
        log_f.write(f"FAILED: {failed_count}\n")
        log_f.write(f"SKIPPED: {skipped_count}\n")
        log_f.write(f"BATCH END: {datetime.now().isoformat(timespec='seconds')}\n")
        log_f.flush()

    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n完成")
    print(f"SUCCESS: {success_count}")
    print(f"FAILED: {failed_count}")
    print(f"SKIPPED: {skipped_count}")
    print(f"日志：{os.path.abspath(args.log)}")
    print(f"汇总 JSON：{os.path.abspath(args.summary_json)}")


if __name__ == "__main__":
    main()