import torch
import torch.nn as nn

torch.set_default_device("cuda")

# llama-8b params
hidden_size = 4096 # H
intermediate_size = 14336 # I
max_position_embeddings = 8192
num_attention_heads = 32 # H
num_key_value_heads = 8 # K
rms_norm_eps = 1e-05
vocab_size = 128256 # V
head_dim = hidden_size // num_attention_heads
kv_dim = num_key_value_heads * head_dim
scaling_factor = 1.0 / (head_dim ** 0.5)

@torch.compile
def rms_norm_kernel(x_BSH, eps=rms_norm_eps):
    return x_BSH * torch.rsqrt(x_BSH.pow(2).mean(-1, keepdim=True) + eps)

qkv_proj_packed = nn.Linear(hidden_size, hidden_size + 2 * kv_dim, bias=False)
o_proj_HH = nn.Linear(hidden_size, hidden_size, bias=False)
gate_proj_HI = nn.Linear(hidden_size, intermediate_size, bias=False)
down_proj_IH = nn.Linear(intermediate_size, hidden_size, bias=False)
def attention_kernel(qkv):
    q, k, v = qkv.split([hidden_size, kv_dim, kv_dim], dim=-1)
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v,
        is_causal=True,
        scale=scaling_factor
    )

def time_kernel(kernel, *args):
    # warmup
    for _ in range(10):
        kernel(*args)
    torch.cuda.synchronize()

    # time
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)


    start.record()
    for _ in range(100):
        kernel(*args)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / 100

if __name__ == "__main__":
    # time all kernels
    x_BSH = torch.randn(1, 8192, hidden_size)

    norm_time = time_kernel(rms_norm_kernel, x_BSH)


    # qkv_BHS = qkv_proj_packed(x_BSH)
    # attn_BHS = attention_kernel(qkv_BHS)

    # o_proj_BSH = o_proj_HH(attn_BHS)
    # down_proj_BSH = down_proj_IH(o_proj_BSH)

