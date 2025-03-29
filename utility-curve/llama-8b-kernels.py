import torch
import torch.nn as nn
from vllm.model_executor.layers.layernorm import RMSNorm
from flash_attn import flash_attn_func, flash_attn_with_kvcache
from green_ctx import make_shard

torch.set_default_device("cuda")
torch.set_default_dtype(torch.bfloat16)

# llama-8b params
hidden_size = 4096  # H
intermediate_size = 14336  # I
max_position_embeddings = 8192
num_attention_heads = 32  # H
num_key_value_heads = 8  # K
rms_norm_eps = 1e-05
vocab_size = 128256  # V
head_dim = hidden_size // num_attention_heads
kv_dim = num_key_value_heads * head_dim
scaling_factor = 1.0 / (head_dim**0.5)

embedding = nn.Embedding(vocab_size, hidden_size)  # [128256, 4096]
rms_norm = RMSNorm(hidden_size, eps=rms_norm_eps)  # [4096]
qkv_proj = nn.Linear(hidden_size, hidden_size + 2 * kv_dim,
                     bias=False)  # [4096, 6144]
o_proj = nn.Linear(hidden_size, hidden_size, bias=False)  # [4096, 4096]
up_proj = nn.Linear(hidden_size, intermediate_size,
                    bias=False)  # [4096, 14336]
down_proj = nn.Linear(intermediate_size, hidden_size,
                      bias=False)  # [14336, 4096]


def prefill_attn(qkv_BS_HKV):
    # cannot use flash_attn_qkvpacked_func because it doesn't supprot GQA
    q, k, v = qkv_BS_HKV.split([hidden_size, kv_dim, kv_dim], dim=-1)
    q = q.view(B, S, num_attention_heads, head_dim)
    k = k.view(B, S, num_key_value_heads, head_dim)
    v = v.view(B, S, num_key_value_heads, head_dim)

    flash_attn_func(q, k, v, causal=True)


def decode_attn(q_B1H, kv_cache_BS_KV):
    q_B1H = q_B1H.view(B, 1, num_attention_heads, head_dim)
    k_cache, v_cache = kv_cache_BS_KV.split([head_dim, head_dim], dim=-1)
    flash_attn_with_kvcache(q_B1H, k_cache, v_cache, causal=True)


def time_kernel(kernel, *args):
    # time
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    num_trials = 50

    start.record()
    for _ in range(num_trials):
        kernel(*args)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / num_trials


def concurrent_launch(kernel, *args, num_sms_per_stream, num_streams):
    # create mutually exclusive green contexts
    ctxs = [make_shard(num_sms_per_stream, i) for i in range(num_streams)]
    go_event = torch.cuda.Event(enable_timing=False)
    stream_events = [(torch.cuda.Event(enable_timing=True),
                      torch.cuda.Event(enable_timing=True))
                     for _ in range(num_streams)]

    num_trials = 50

    for i, ctx in enumerate(ctxs):
        with ctx.with_context(), ctx.with_torch_stream() as stream:
            stream.wait_event(go_event)

            stream_events[i][0].record()
            for _ in range(num_trials):
                kernel(*args)
            stream_events[i][1].record()

    go_event.record()

    times = []
    for start, end in stream_events:
        start.synchronize()
        end.synchronize()
        times.append(start.elapsed_time(end) / num_trials)
    return times


def warmup_device():
    a = torch.randn(128, 81920)
    b = torch.randn(81920, 128)
    for _ in range(100):
        a @ b


if __name__ == "__main__":
    # time all kernels
    B, S = 1, 8192
    # for matuls
    x_BSH = torch.randn(B, S, hidden_size)
    up_projected_BSH = up_proj(x_BSH)
    # for prefill attn
    qkv_BSH_HKV = qkv_proj(x_BSH)
    # for decode attn
    q_B1H = qkv_BSH_HKV[:, 0, :hidden_size]
    kv_cache_BS_KV = torch.randn(B, S, num_key_value_heads, head_dim * 2)

    warmup_device()

    kernels = {
        "rms_norm": rms_norm,
        "qkv_proj": qkv_proj,
        "o_proj": o_proj,
        "up_proj": up_proj,
        "down_proj": down_proj,
        "prefill_attn": prefill_attn,
        "decode_attn": decode_attn,
    }

    # print("standalone kernel time")
    # num_sms = [8, 16, 32, 48, 64, 80, 96, 112, 128]
    # ctxs = [make_shard(i) for i in num_sms]
    # for kernel_name, kernel in kernels.items():
    #     for ctx, n in zip(ctxs, num_sms):
    #         with ctx.with_context():
    #             if kernel_name == "prefill_attn":
    #                 time = time_kernel(kernel, qkv_BSH_HKV)
    #             elif kernel_name == "decode_attn":
    #                 time = time_kernel(kernel, q_B1H, kv_cache_BS_KV)
    #             elif kernel_name == "down_proj":
    #                 time = time_kernel(kernel, up_projected_BSH)
    #             else:
    #                 time = time_kernel(kernel, x_BSH)
    #             print(f"{kernel_name} kernel: {time} ms, {n} SMs")

    print("concurrent kernel time")
    scenarios = [
        (8, [1, 2, 4, 8, 16]),
        (16, [1, 2, 4, 8]),
        (32, [1, 2, 4]),
        (64, [1, 2]),
        (128, [1]),
    ]
    for num_sms, num_streams in scenarios:
        for num_stream in num_streams:
            for kernel_name, kernel in kernels.items():
                if kernel_name == "prefill_attn":
                    time = concurrent_launch(kernel,
                                             qkv_BSH_HKV,
                                             num_sms_per_stream=num_sms,
                                             num_streams=num_stream)
                elif kernel_name == "decode_attn":
                    time = concurrent_launch(kernel,
                                             q_B1H,
                                             kv_cache_BS_KV,
                                             num_sms_per_stream=num_sms,
                                             num_streams=num_stream)
                elif kernel_name == "down_proj":
                    time = concurrent_launch(kernel,
                                             up_projected_BSH,
                                             num_sms_per_stream=num_sms,
                                             num_streams=num_stream)
                else:
                    time = concurrent_launch(kernel,
                                             x_BSH,
                                             num_sms_per_stream=num_sms,
                                             num_streams=num_stream)
                print(
                    f"{kernel_name} kernel: {time} ms, {num_sms} SMs, {num_stream} streams"
                )
