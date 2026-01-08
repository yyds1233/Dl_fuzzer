#!/usr/bin/env python3
# export_schema_standalone.py
import argparse
import importlib
import inspect
import json
import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


# -----------------------
# 1) API list loader (no get_api_info)
# -----------------------
def load_api_list(path: str) -> List[str]:
    """
    支持：
      - .txt   : 每行一个 API
      - .json  : JSON array of strings
      - .pkl/.pickle : pickle(list[str])
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    if p.suffix.lower() in (".txt",):
        lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
        apis = []
        for ln in lines:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            apis.append(ln)
        return apis

    if p.suffix.lower() in (".json",):
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("json must be a list of api strings")
        return [str(x).strip() for x in data if str(x).strip()]

    if p.suffix.lower() in (".pkl", ".pickle"):
        with p.open("rb") as f:
            data = pickle.load(f)
        if not isinstance(data, list):
            raise ValueError("pickle must be a list of api strings")
        return [str(x).strip() for x in data if str(x).strip()]

    raise ValueError(f"Unsupported api list file: {path}")


# -----------------------
# 2) helpers
# -----------------------
def resolve_obj_from_qualname(qualname: str):
    """
    'torch.sparse_csc_tensor' -> 实际对象
    只处理 module.submodule.attr
    """
    qualname = qualname.strip()
    if qualname.endswith("()"):
        qualname = qualname[:-2]

    module_name, _, attr_name = qualname.rpartition(".")
    if not module_name:
        raise ValueError(f"Invalid qualified name: {qualname}")

    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def python_signature_to_dict(sig: inspect.Signature) -> Dict[str, Any]:
    params = []
    for idx, (name, param) in enumerate(sig.parameters.items()):
        has_default = param.default is not inspect._empty
        default_repr = None if not has_default else repr(param.default)

        has_annot = param.annotation is not inspect._empty
        annot_repr = None if not has_annot else repr(param.annotation)

        params.append(
            {
                "name": name,
                "position": idx,
                "kind": param.kind.name,
                "has_default": has_default,
                "default": default_repr,
                "annotation": annot_repr,
            }
        )

    ret_annot = None if sig.return_annotation is inspect._empty else repr(sig.return_annotation)

    return {
        "signature_str": str(sig),
        "parameters": params,
        "return_annotation": ret_annot,
    }


def _arg_has_default_in_schema(schema_str: str, arg_name: str) -> bool:
    """
    以 schema_str 作为权威来源判断参数是否“真的有默认值”。

    原因：
      在不同 torch 版本里，FunctionSchemaArgument 可能总带 default_value 字段，
      但这不等价于 Python/ATen 调用层面真的可省略参数。

    规则：
      只要在 schema_str 的参数部分出现了 'arg_name='（允许空格），就认为有默认值。
      例如：
        'Tensor? bias=None'      -> bias 有默认
        'SymInt[2] stride=[1,1]' -> stride 有默认
        'Tensor input'          -> input 没默认（必选）
    """
    # schema_str 示例：aten::conv2d(Tensor input, Tensor weight, Tensor? bias=None, ...)
    # 这里用一个保守的匹配：在分隔符之后出现 arg_name，再出现 '='
    # 避免把其它位置的字符串误判成默认（例如返回部分）
    pattern = rf"(^|[\s,(]){re.escape(arg_name)}\s*="
    return re.search(pattern, schema_str) is not None


def aten_function_schema_to_dict(fschema) -> Dict[str, Any]:
    """
    单个 FunctionSchema -> dict
    """
    schema_str = str(fschema)

    args = []
    for idx, arg in enumerate(fschema.arguments):
        type_str = str(arg.type)  # e.g. "Tensor", "ScalarType?", "int[]?"
        # 更稳的 optional 判断：Optional[...] 或者类型字符串带 '?'
        optional = ("Optional[" in type_str) or ("?" in type_str)

        # ✅ 关键修复：默认值判断以 schema_str 为准，而不是 default_value 属性是否存在
        has_default = _arg_has_default_in_schema(schema_str, arg.name)

        # default_repr 只有在“确实有默认值”时才填
        default_repr = None
        if has_default:
            # 若底层确实提供 default_value，就用它；否则尽量从 schema_str 解析（这里先 best-effort）
            if hasattr(arg, "default_value"):
                try:
                    default_repr = repr(arg.default_value)
                except Exception:
                    default_repr = None

        args.append(
            {
                "name": arg.name,
                "position": idx,
                "type": type_str,
                "optional": optional,
                "kw_only": getattr(arg, "kwarg_only", False),
                "has_default": has_default,
                "default": default_repr,
            }
        )

    rets = [{"name": ret.name, "type": str(ret.type)} for ret in fschema.returns]
    return {
        "schema_str": schema_str,
        "arguments": args,
        "returns": rets,
    }


def safe_filename(api_name: str) -> str:
    base = api_name.strip()
    if base.endswith("()"):
        base = base[:-2]
    base = base.replace(".", "_")
    return f"{base}_schema.json"


# -----------------------
# 3) ATen schema extractors
# -----------------------
def get_aten_schemas_from_torch_ops(aten_name: str) -> Optional[Dict[str, Any]]:
    """
    尝试从 torch.ops.aten.<aten_name> 拿 schema：
      - OpOverloadPacket:  ._schemas (dict) 或 overload._schema
      - OpOverload:        ._schema
    """
    try:
        packet_or_op = getattr(torch.ops.aten, aten_name)
    except AttributeError:
        return None

    # 多 overload：优先用 _schemas（如果存在）
    if hasattr(packet_or_op, "_schemas"):
        out = {}
        for overload_name, fschema in packet_or_op._schemas.items():
            out[str(overload_name)] = aten_function_schema_to_dict(fschema)
        return out

    # 单个 overload
    if hasattr(packet_or_op, "_schema"):
        return {"default": aten_function_schema_to_dict(packet_or_op._schema)}

    # 兜底：有些版本没有 _schemas 但可以枚举 overload 属性
    overloads = {}
    for attr in dir(packet_or_op):
        if attr.startswith("_"):
            continue
        try:
            ov = getattr(packet_or_op, attr)
        except Exception:
            continue
        if hasattr(ov, "_schema"):
            overloads[attr] = aten_function_schema_to_dict(ov._schema)
    return overloads or None


def get_aten_schemas_from_jit(aten_name: str) -> Optional[Dict[str, Any]]:
    """
    Fallback：torch._C._jit_get_schemas_for_operator("aten::<name>")
    返回 list[FunctionSchema]，我们用 overload_0/1/... 命名
    """
    qname = f"aten::{aten_name}"
    try:
        schemas = torch._C._jit_get_schemas_for_operator(qname)
    except Exception:
        return None
    if not schemas:
        return None
    out = {}
    for i, fs in enumerate(schemas):
        out[f"overload_{i}"] = aten_function_schema_to_dict(fs)
    return out


def get_aten_schemas(aten_name: str) -> Optional[Dict[str, Any]]:
    # 先 torch.ops.aten，再 jit fallback
    s = get_aten_schemas_from_torch_ops(aten_name)
    if s is not None:
        return s
    return get_aten_schemas_from_jit(aten_name)


# -----------------------
# 4) main exporter
# -----------------------
def export_torch_api_schema(api_name: str, out_dir: Path) -> None:
    api_info: Dict[str, Any] = {
        "api_name": api_name,
        "python_signature": None,
        "aten": None,
        "error": None,
    }

    # (1) Python signature (best-effort)
    try:
        obj = resolve_obj_from_qualname(api_name)
        sig = inspect.signature(obj)
        api_info["python_signature"] = python_signature_to_dict(sig)
    except Exception as e:
        api_info["error"] = f"inspect.signature failed: {e}"

    # (2) ATen schema (best-effort)
    aten_name = api_name.split(".")[-1]
    try:
        overloads = get_aten_schemas(aten_name)
        if overloads is not None:
            api_info["aten"] = {"aten_name": aten_name, "overloads": overloads}
        else:
            # 没有 aten schema，不作为 fatal
            if api_info["python_signature"] is None:
                api_info["error"] = api_info["error"] or "no aten schema found"
    except Exception as e:
        api_info["error"] = (api_info["error"] + " | " if api_info["error"] else "") + f"aten schema failed: {e}"

    out_dir.mkdir(parents=True, exist_ok=True)
    filename = out_dir / safe_filename(api_name)
    filename.write_text(json.dumps(api_info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[+] Saved schema for {api_name} -> {filename}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api_list", required=True, help="txt/json/pkl, each entry is a qualified api name")
    ap.add_argument("--out_dir", default="./api_schema", help="output dir for schema json files")
    args = ap.parse_args()

    api_list = load_api_list(args.api_list)
    out_dir = Path(args.out_dir).resolve()

    print(f"[+] loaded {len(api_list)} apis from {args.api_list}")
    for api in api_list:
        export_torch_api_schema(api, out_dir)


if __name__ == "__main__":
    main()
