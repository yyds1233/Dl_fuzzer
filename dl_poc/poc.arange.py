# poc_torch_arange_fpe.py
import torch

print("torch version:", torch.__version__)

# Minimal reproducer:
# fractional step + integral dtype
x = torch.arange(0, 10, 0.5, dtype=torch.int64)

print(x)