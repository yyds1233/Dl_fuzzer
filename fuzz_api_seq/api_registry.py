# api_registry.py
from pathlib import Path
from typing import Dict, Any
import yaml

def load_api_spec_from_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    return spec

def build_spec_registry(dir_path: str) -> Dict[str, Dict[str, Any]]:
    """
    扫描某个目录下所有 *.yaml，把每个 api_name 映射到它的 spec。
    例如:
      conv2d.yaml  -> api_name == "torch.nn.functional.conv2d"
      relu.yaml    -> api_name == "torch.nn.functional.relu"
    """
    base = Path(dir_path)
    registry: Dict[str, Dict[str, Any]] = {}
    for p in base.glob("*.yaml"):
        spec = load_api_spec_from_file(p)
        api_name = spec["api_name"]
        registry[api_name] = spec
    return registry

