import torch
import numpy as np

from green_ctx import make_shard, init
from green_ctx.kernels import count_sm_ids
from green_ctx.timing import cuda_timing_decorator


def example_shard():
    print("---\nExample showcasing limiting the number of SMs used by a context.")
    green_ctx = make_shard(8)
    print(f"Using {green_ctx.sm_count} SMs.")

    print("Before: SM utilizations ({SM ID: usage count})")
    print(count_sm_ids(200))

    with green_ctx.with_context():
        print("After: SM utilizations ({SM ID: usage count})")
        print(count_sm_ids(200))


def benchmark_matmul():
    print("---\nBenchmarking matrix multiplication (M=N=K=1024) with different number of SMs.")
    @cuda_timing_decorator
    def matmul(a, b, c):
        torch.matmul(a, b, out=c)

    a = torch.randn(1024, 1024).cuda()
    b = torch.randn(1024, 1024).cuda()
    c = torch.zeros(1024, 1024).cuda()

    # warm ups
    [matmul(a, b, c) for _ in range(5)]

    sms_to_ms = dict()
    for i in [16, 32, 64, 128]:
        green_ctx = make_shard(i)
        with green_ctx.with_context():
            durations = [matmul(a, b, c) for _ in range(5)]
            sms_to_ms[i] = np.mean(durations)

    print("Benchmark results (SMs: milliseconds)")
    for sm, ms in sms_to_ms.items():
        print(f"{sm}: {ms:.2f}")

def main():
    example_shard()
    benchmark_matmul()


if __name__ == "__main__":
    init()
    main()
