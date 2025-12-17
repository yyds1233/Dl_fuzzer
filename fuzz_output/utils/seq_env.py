# sequence_env.py（你可以放在一个文件里，也可以直接写在 harness 里）

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import torch
import atheris

@dataclass
class TensorInfo:
    tensor: torch.Tensor
    name: str
    origin_api: str

class SequenceEnv:
    def __init__(self) -> None:
        self.tensors: List[TensorInfo] = []

    def add_tensor(self, t: torch.Tensor, name: str, origin_api: str) -> None:
        self.tensors.append(TensorInfo(tensor=t, name=name, origin_api=origin_api))

    def get_tensor_by_name(self, name: str) -> Optional[torch.Tensor]:
        for info in reversed(self.tensors):
            if info.name == name:
                return info.tensor
        return None

    def pick_any_tensor(self, fdp: atheris.FuzzedDataProvider) -> Optional[torch.Tensor]:
        if not self.tensors:
            return None
        idx = fdp.ConsumeIntInRange(0, len(self.tensors) - 1)
        return self.tensors[idx].tensor

