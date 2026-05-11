# poc.nn.quantized.functional.conv2d.py
import torch

print("torch version:", torch.__version__)
torch.backends.quantized.engine = 'qnnpack'

# 关键参数组合：形状是关键，数值不重要
N, C_in, H_in, W_in = 2, 8, 16, 7
C_out = 6
groups = 8  # 关键：groups == C_in

# 构造输入 (数值全为0不影响崩溃，只要形状对)
x_fp32 = torch.zeros(N, C_in, H_in, W_in, dtype=torch.float32)
x_quant = torch.quantize_per_tensor(x_fp32, scale=0.01, zero_point=0, dtype=torch.quint8)

# 构造权重 (形状是关键：(C_out, C_in//groups, K_h, K_w) = (6, 1, 7, 1))
w_fp32 = torch.zeros(C_out, C_in // groups, 7, 1, dtype=torch.float32)
w_quant = torch.quantize_per_tensor(w_fp32, scale=0.01, zero_point=-128, dtype=torch.qint8)

print(f"[*] Running with Groups={groups}, Weight Shape={tuple(w_quant.shape)}")
print("[*] This should trigger an FPE crash...")

# 触发崩溃
try:
    output = torch.nn.quantized.functional.conv2d(
        x_quant, w_quant, None,
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        groups=groups,
        scale=0.01,
        zero_point=0
    )
    print(f"[!] Unexpected success: {output.shape}")
except RuntimeError as e:
    # 如果只是普通报错，说明被前端拦住了
    print(f"Caught RuntimeError: {e}")
except Exception as e:
    print(f"Caught Exception: {e}")