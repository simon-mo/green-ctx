import torch
import torch.nn as nn
from vllm.model_executor.layers.layernorm import RMSNorm

from flash_attn import flash_attn_with_kvcache
from vllm_flash_attn import flash_attn_varlen_func

from green_ctx import make_shard
from green_ctx.kernels import run_global_timer, run_sleep_kernel, run_barrier_kernel
from triton.testing import do_bench_cudagraph

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


def prefill_attn(qkv_BS_HKV, cu_seqlens_q, sm_margin=0):
    # cannot use flash_attn_qkvpacked_func because it doesn't supprot GQA
    q, k, v = qkv_BS_HKV.split([hidden_size, kv_dim, kv_dim], dim=-1)
    q = q.view(B * S, num_attention_heads, head_dim)
    k = k.view(B * S, num_key_value_heads, head_dim)
    v = v.view(B * S, num_key_value_heads, head_dim)

    # flash_attn_func(q, k, v, causal=True)
    flash_attn_varlen_func(q,
                           k,
                           v,
                           cu_seqlens_q=cu_seqlens_q,
                           cu_seqlens_k=cu_seqlens_q,
                           max_seqlen_q=S,
                           max_seqlen_k=S,
                           causal=True,
                           fa_version=3,
                           sm_margin=sm_margin)


def decode_attn(q_B1H, kv_cache_BS_KV, sm_margin=0):
    q_B1H = q_B1H.view(B, 1, num_attention_heads, head_dim)
    k_cache, v_cache = kv_cache_BS_KV.split([head_dim, head_dim], dim=-1)
    flash_attn_with_kvcache(
        q_B1H, k_cache, v_cache,
        causal=True)  # note this doesn't have sm_margin yet.


num_trials = 50


def time_kernel(kernel, *args, **kwargs):
    # return do_bench_cudagraph(lambda: kernel(*args))

    # time
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # barrier_buf = torch.zeros(1, dtype=torch.uint64)
    # run_barrier_kernel(barrier_buf, 1, 1000000)
    start.record()
    for _ in range(num_trials):
        kernel(*args, **kwargs)
    end.record()
    # barrier_buf[0] = 1
    torch.cuda.synchronize()

    return start.elapsed_time(end) / num_trials


def concurrent_launch(kernel,
                      args,
                      kwargs,
                      num_sms_per_stream,
                      num_streams,
                      timing_buffer_tensor=None):
    # create mutually exclusive green contexts
    max_exclusive_shards = 128 // num_sms_per_stream
    ctxs = [
        make_shard(num_sms_per_stream, i % max_exclusive_shards)
        for i in range(num_streams)
    ]
    stream_events = [(torch.cuda.Event(enable_timing=True),
                      torch.cuda.Event(enable_timing=True))
                     for _ in range(num_streams)]

    if timing_buffer_tensor is not None:
        assert timing_buffer_tensor.numel() == num_streams * num_trials

    barrier_buf = torch.zeros(1, dtype=torch.uint64)

    for i, ctx in enumerate(ctxs):
        with ctx.with_context(), ctx.with_torch_stream() as stream:
            for j in range(num_trials):
                stream_id = ctx.raw_stream_id
                if j == 0:
                    run_barrier_kernel(barrier_buf,
                                       num_streams,
                                       10,
                                       stream=stream_id)
                    stream_events[i][0].record()
                kernel(*args, **kwargs)
                # if i == 0:
                #     for _ in range(10):
                #         run_sleep_kernel(1000, stream_id)
                run_global_timer(timing_buffer_tensor, i * num_trials + j,
                                 stream_id)
                if j == num_trials - 1:
                    stream_events[i][1].record()

    times = []
    for start, end in stream_events:
        start.synchronize()
        end.synchronize()
        times.append(start.elapsed_time(end) / num_trials)

    torch.cuda.synchronize()

    return times


from dataclasses import dataclass
from typing import Callable, Tuple, Optional, List
from contextlib import contextmanager


@dataclass
class Process:
    kernel: Callable
    args: Tuple
    kwargs: dict
    num_sms_per_stream: int
    resource_idx: int = 0

    ctx = None

    def __post_init__(self):
        self.ctx = make_shard(self.num_sms_per_stream, self.resource_idx)

    @contextmanager
    def with_stream(self):
        with self.ctx.with_context(), self.ctx.with_torch_stream():
            yield self.ctx.raw_stream_id


def multiplexed_launch(processes: List[Process],
                       timing_buffer_tensor: Optional[torch.Tensor] = None):
    stream_events = [(torch.cuda.Event(enable_timing=True),
                      torch.cuda.Event(enable_timing=True))
                     for _ in range(len(processes))]

    barrier_buf = torch.zeros(1, dtype=torch.uint64)
    for i, process in enumerate(processes):
        with process.with_stream() as stream_id:
            # NOTE(simon): this hangs, very weird
            # run_barrier_kernel(barrier_buf,
            #                    len(processes),
            #                    10,
            #                    stream=stream_id)
            stream_events[i][0].record()
            for j in range(num_trials):
                process.kernel(*process.args, **process.kwargs)
                run_global_timer(timing_buffer_tensor, i * num_trials + j,
                                 stream_id)
            stream_events[i][1].record()

    times = []
    for start, end in stream_events:
        start.synchronize()
        end.synchronize()
        times.append(start.elapsed_time(end) / num_trials)

    return times


def warmup_device(x_BSH, up_projected_BSH, qkv_BSH_HKV, q_B1H, kv_cache_BS_KV,
                  cu_seqlens_q):
    for _ in range(100):
        rms_norm(x_BSH)
        qkv_proj(x_BSH)
        o_proj(x_BSH)
        up_proj(x_BSH)
        down_proj(up_projected_BSH)
        prefill_attn(qkv_BSH_HKV, cu_seqlens_q)
        decode_attn(q_B1H, kv_cache_BS_KV)

    run_sleep_kernel(1)
    run_global_timer(torch.zeros(1, dtype=torch.uint64))

    barrier_ptr = torch.zeros(1, dtype=torch.uint64)
    stream_1 = torch.cuda.Stream()
    stream_2 = torch.cuda.Stream()
    run_barrier_kernel(barrier_ptr, 2, 1000000, stream=stream_1.cuda_stream)
    run_barrier_kernel(barrier_ptr, 2, 1000000, stream=stream_2.cuda_stream)
    torch.cuda.synchronize()


def plot_timing(relative_timing):
    # Convert timing data to milliseconds for better readability
    timing_ms = relatived_timing.float(
    ) / 1e6  # Convert from nanoseconds to milliseconds

    import matplotlib.pyplot as plt
    import numpy as np

    # Set up a high-quality figure with a modern style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 8), dpi=300)

    # Create scatter plot
    for i in range(timing_out.shape[0]):
        plt.scatter(
            timing_ms[i].cpu().numpy(),
            np.ones_like(timing_ms[i].cpu().numpy()) * i,
            alpha=0.7,
            s=50,  # Marker size
            label=f"Stream {i}" if i < 5 else None,  # Limit legend entries
            edgecolors='k',
            linewidths=0.5)

    # Add a colorful vertical line for the average time per stream
    stream_means = timing_ms.mean(dim=1).cpu().numpy()
    for i, mean_val in enumerate(stream_means):
        plt.axvline(x=mean_val,
                    color=plt.cm.tab10(i % 10),
                    linestyle='--',
                    alpha=0.5)

    # Customize the plot
    plt.title('Kernel Execution Timing per Stream and Trial', fontsize=16)
    plt.xlabel('Time (milliseconds)', fontsize=14)
    plt.ylabel('Stream ID', fontsize=14)
    plt.yticks(range(timing_out.shape[0]))
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add legend for the first few streams to avoid overcrowding
    plt.legend(loc='upper right', framealpha=0.9)

    # Add statistics annotation
    textstr = f"Total streams: {timing_out.shape[0]}\nTrials per stream: {num_trials}"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.annotate(textstr,
                 xy=(0.02, 0.98),
                 xycoords='axes fraction',
                 fontsize=10,
                 va='top',
                 ha='left',
                 bbox=props)

    plt.tight_layout()
    plt.savefig('kernel_timing_scatter.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    # time all kernels
    B, S = 1, 2048
    # B, S = 8, 1

    # for matuls
    x_BSH = torch.randn(B, S, hidden_size)
    up_projected_BSH = up_proj(x_BSH)
    # for prefill attn
    qkv_BSH_HKV = qkv_proj(x_BSH)
    # for decode attn
    q_B1H = qkv_BSH_HKV[:, 0, :hidden_size]
    # q_B1H = torch.randn(B, 1, num_attention_heads, head_dim)
    kv_cache_BS_KV = torch.randn(B, S, num_key_value_heads, head_dim * 2)

    cu_seqlens_q = torch.arange(B + 1, dtype=torch.int32) * S

    warmup_device(x_BSH, up_projected_BSH, qkv_BSH_HKV, q_B1H, kv_cache_BS_KV,
                  cu_seqlens_q)

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
    # num_sms = [8, 16, 32, 48, 64, 80, 96, 112, 128, 132]
    # ctxs = [make_shard(i) for i in num_sms]
    # print("kernel_name,time(ms),num_sms")
    # for kernel_name, kernel in kernels.items():
    #     for ctx, n in zip(ctxs, num_sms):
    #         with ctx.with_context():
    #             if kernel_name == "prefill_attn":
    #                 time = time_kernel(kernel,
    #                                    qkv_BSH_HKV,
    #                                    cu_seqlens_q,
    #                                    sm_margin=132 - n)
    #             elif kernel_name == "decode_attn":
    #                 time = time_kernel(kernel,
    #                                    q_B1H,
    #                                    kv_cache_BS_KV,
    #                                    sm_margin=132 - n)
    #             elif kernel_name == "down_proj":
    #                 time = time_kernel(kernel, up_projected_BSH)
    #             else:
    #                 time = time_kernel(kernel, x_BSH)
    #             print(f"{kernel_name},{time},{n}")

    # print("concurrent kernel time")
    # timing_buffer_tensor = None
    # scenarios = [
    #     # (8, [1, 2, 4, 8, 16, 32]),
    #     # (16, [1, 2, 4, 8, 16]),
    #     (32, [1, 2, 4, 8]),
    #     # (64, [1, 2, 4]),
    #     # (128, [1, 2]),
    # ]
    # print("kernel_name,time(ms),num_sms,num_streams")
    # for num_sms, num_streams in scenarios:
    #     for num_stream in num_streams:
    #         for kernel_name, kernel in kernels.items():
    #             if kernel_name == "prefill_attn":
    #                 time = concurrent_launch(
    #                     kernel,
    #                     args=(qkv_BSH_HKV, cu_seqlens_q),
    #                     kwargs={"sm_margin": 132 - num_sms},
    #                     num_sms_per_stream=num_sms,
    #                     num_streams=num_stream)
    #             elif kernel_name == "decode_attn":
    #                 time = concurrent_launch(
    #                     kernel,
    #                     args=(q_B1H, kv_cache_BS_KV),
    #                     kwargs={"sm_margin": 132 - num_sms},
    #                     num_sms_per_stream=num_sms,
    #                     num_streams=num_stream)
    #             elif kernel_name == "down_proj":
    #                 time = concurrent_launch(kernel,
    #                                          args=(up_projected_BSH, ),
    #                                          kwargs={},
    #                                          num_sms_per_stream=num_sms,
    #                                          num_streams=num_stream)
    #             else:
    #                 if num_sms == 32 and num_stream == 4 and kernel_name == "qkv_proj":
    #                     assert timing_buffer_tensor is None
    #                     timing_buffer_tensor = torch.zeros(num_trials *
    #                                                        num_stream,
    #                                                        dtype=torch.uint64)
    #                     timing_ = timing_buffer_tensor
    #                 else:
    #                     timing_ = None
    #                 time = concurrent_launch(kernel,
    #                                          args=(x_BSH, ),
    #                                          kwargs={},
    #                                          num_sms_per_stream=num_sms,
    #                                          num_streams=num_stream,
    #                                          timing_buffer_tensor=timing_)
    #             time = sum(time) / len(time)
    #             print(f"{kernel_name},{time},{num_sms},{num_stream}")

    # timing_out = timing_buffer_tensor.reshape(4, num_trials).to(torch.float64)
    # relatived_timing = timing_out - timing_out[0, 0]
    # plot_timing(relatived_timing)

    print("multiplexed launch")

    qkv_process_sm_64_first = Process(
        kernel=kernels["qkv_proj"],
        args=(x_BSH, ),
        kwargs={},
        num_sms_per_stream=64,
        resource_idx=0,
    )
    qkv_process_sm_64_second = Process(
        kernel=kernels["qkv_proj"],
        args=(x_BSH, ),
        kwargs={},
        num_sms_per_stream=64,
        resource_idx=1,
    )
    prefill_process_sm_64_second = Process(
        kernel=kernels["prefill_attn"],
        args=(qkv_BSH_HKV, cu_seqlens_q),
        kwargs={"sm_margin": 132 - 64},
        num_sms_per_stream=64,
        resource_idx=1,
    )
    decode_process_sm_64_second = Process(
        kernel=kernels["decode_attn"],
        args=(q_B1H, kv_cache_BS_KV),
        kwargs={"sm_margin": 132 - 64},
        num_sms_per_stream=64,
        resource_idx=1,
    )

    workloads = [
        ("qkv + qkv", (qkv_process_sm_64_first, qkv_process_sm_64_second)),
        ("qkv + prefill", (qkv_process_sm_64_first,
                           prefill_process_sm_64_second)),
        ("qkv + decode", (qkv_process_sm_64_first,
                          decode_process_sm_64_second)),
    ]

    for name, processes in workloads:
        times = multiplexed_launch(list(processes))
        standalone_times = []
        for p in processes:
            standalone_time = time_kernel(p.kernel, *p.args, **p.kwargs)
            standalone_times.append(standalone_time)
        print(f"{name=} {times=} {standalone_times=}")
