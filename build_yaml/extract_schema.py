import importlib
import inspect
import json
import os
import sys
from pathlib import Path

import torch

Config_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs"))
if Config_DIR not in sys.path:
    sys.path.insert(0, Config_DIR)

from get_api_info import *  # noqa: F403,F401


def resolve_obj_from_qualname(qualname: str):
    """
    'torch.sparse_csc_tensor' -> 实际对象
    只处理形如 module.submodule.attr 的形式
    """
    qualname = qualname.strip()
    if qualname.endswith("()"):
        qualname = qualname[:-2]

    module_name, _, attr_name = qualname.rpartition(".")
    if not module_name:
        raise ValueError(f"Invalid qualified name: {qualname}")

    module = importlib.import_module(module_name)
    obj = getattr(module, attr_name)
    return obj


def python_signature_to_dict(sig: inspect.Signature):
    """
    把 inspect.signature 返回的 Signature 转成 JSON 友好的 dict
    """
    params = []
    for idx, (name, param) in enumerate(sig.parameters.items()):
        # 默认值
        has_default = param.default is not inspect._empty
        default_repr = None if not has_default else repr(param.default)

        # 类型注解
        has_annot = param.annotation is not inspect._empty
        annot_repr = None if not has_annot else repr(param.annotation)

        params.append(
            {
                "name": name,
                "position": idx,
                # POSITIONAL_ONLY / VAR_POSITIONAL / KEYWORD_ONLY 等
                "kind": param.kind.name,
                "has_default": has_default,
                "default": default_repr,
                "annotation": annot_repr,
            }
        )

    # 返回值类型
    ret_annot = None if sig.return_annotation is inspect._empty else repr(sig.return_annotation)

    return {
        "signature_str": str(sig),
        "parameters": params,
        "return_annotation": ret_annot,
    }


def aten_function_schema_to_dict(fschema):
    """
    单个 FunctionSchema -> dict
    """
    args = []
    for idx, arg in enumerate(fschema.arguments):
        type_str = str(arg.type)  # e.g. "Tensor", "ScalarType?", "int[]?"
        optional = "?" in type_str  # 粗略判断 Optional[T]

        has_default = hasattr(arg, "default_value")
        default_repr = repr(arg.default_value) if has_default else None

        args.append(
            {
                "name": arg.name,
                "position": idx,
                "type": type_str,
                "optional": optional,
                "kw_only": arg.kwarg_only,
                "has_default": has_default,
                "default": default_repr,
            }
        )

    rets = [{"name": ret.name, "type": str(ret.type)} for ret in fschema.returns]

    return {
        "schema_str": str(fschema),
        "arguments": args,
        "returns": rets,
    }


def get_aten_schemas_from_name(aten_name: str):
    """
    尝试从 torch.ops.aten.<aten_name> 拿 schema：
      - 如果是 OpOverloadPacket: 用 ._schemas (dict)
      - 如果是单个 OpOverload:   用 ._schema
    """
    try:
        packet_or_op = getattr(torch.ops.aten, aten_name)
    except AttributeError:
        return None  # 没有对应的 aten op

    # 多 overload 的情况
    if hasattr(packet_or_op, "_schemas"):
        schemas_dict = {}
        for overload_name, fschema in packet_or_op._schemas.items():
            schemas_dict[overload_name] = aten_function_schema_to_dict(fschema)
        return schemas_dict

    # 只有单个 overload
    if hasattr(packet_or_op, "_schema"):
        fschema = packet_or_op._schema
        return {"default": aten_function_schema_to_dict(fschema)}

    # 理论上不会走到这里
    return None


def safe_filename(api_name: str) -> str:
    """
    把 'torch.sparse_csc_tensor' 变成 'torch_sparse_csc_tensor_schema.json'
    """
    base = api_name.strip()
    if base.endswith("()"):
        base = base[:-2]
    base = base.replace(".", "_")
    return f"{base}_schema.json"


def export_torch_api_schema(api_name: str):
    """
    对单个 api_name 导出 schema 信息到独立 JSON 文件：
      1) 先用 inspect.signature
      2) 失败再尝试 aten schema
      3) 都失败则写 error
    """
    api_info = {
        "api_name": api_name,
        "python_signature": None,
        "aten": None,
        "error": None,
    }

    # 1) python 层 signature
    sig = None
    try:
        obj = resolve_obj_from_qualname(api_name)
        sig = inspect.signature(obj)
    except Exception as e:
        api_info["error"] = f"inspect.signature failed: {e}"

    if sig is not None:
        api_info["python_signature"] = python_signature_to_dict(sig)

    # 2) aten schema（即使 python signature 成功也可以保留 aten 信息）
    aten_name = api_name.split(".")[-1]
    aten_schemas = get_aten_schemas_from_name(aten_name)
    if aten_schemas is not None:
        api_info["aten"] = {"aten_name": aten_name, "overloads": aten_schemas}
    else:
        if api_info["python_signature"] is None:
            api_info["error"] = api_info["error"] or "no aten op found"

    # 3) 写入 JSON 文件
    script_dir = Path(__file__).resolve().parent
    out_dir = script_dir / "sparse_api_schema"
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = out_dir / safe_filename(api_name)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(api_info, f, indent=2, ensure_ascii=False)

    print(f"[+] Saved schema for {api_name} to {filename}")


if __name__ == "__main__":
    # 这里放你的 sparse API 清单（目前代码实际使用的是 load_api_list 的结果）
    API_LIST = [
        "torch.sparse_csc_tensor",
        "torch.sparse_csr_tensor",
        "torch.sparse_coo_tensor",
    ]

    api_list = load_api_list(API_PICKLE_PATH)  # noqa: F405

    for api in api_list:
        export_torch_api_schema(api)
