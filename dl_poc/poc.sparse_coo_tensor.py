
# Auto-generated Minimal POC for Sparse COO Crash
import torch

print("torch version:", torch.__version__)

# Indices
indices = torch.tensor([[-4702111234474983746]], dtype=torch.int64)

# Values
values = torch.tensor([-16.0], dtype=torch.float32)

# Size
size = (1,)

print(f"[*] Indices shape: {indices.shape}")
print(f"[*] Values shape: {values.shape}")
print(f"[*] Size: {size}")

# Create tensor
t = torch.sparse_coo_tensor(
    indices=indices,
    values=values,
    size=size,
    dtype=torch.float32,
    device='cpu',
    check_invariants=False  # Disable checks to ensure crash
)

print("[*] Triggering to_dense()...")
# This should crash
result = t.to_dense()
print("Success:", result.shape)
