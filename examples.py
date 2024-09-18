import torch
import numpy as np

from green_ctx import make_shard, init, partition
from green_ctx.kernels import count_sm_ids, launch_smid
from green_ctx.timing import cuda_timing_decorator
from collections import Counter


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

def multistream_dispatch():
    print("---\nExample showcasing multi-stream dispatch under the same context.")
    green_ctx = make_shard(8)
    stream_1 = green_ctx.make_stream()
    stream_2 = green_ctx.make_stream()

    with green_ctx.with_context():
        num_sm = 200
        ids_1 = np.zeros(num_sm, dtype=np.int32)
        ids_2 = np.zeros(num_sm, dtype=np.int32)

        launch_smid(ids_1, num_sm, stream=stream_1)
        launch_smid(ids_2, num_sm, stream=stream_2)
        torch.cuda.synchronize()

        print("SM utilizations for stream 1 ({SM ID: usage count})")
        print(Counter(ids_1))
        print("SM utilizations for stream 2 ({SM ID: usage count})")
        print(Counter(ids_2))

def partition_two():
    print("---\nExample showcasing partitioning a device into two shards")
    green_ctx_1, green_ctx_2 = partition(32, 8)
    print(f"Created {green_ctx_1.sm_count} SMs for shard 1: {green_ctx_1.sm_ids}")
    print(f"Created {green_ctx_2.sm_count} SMs for shard 2. {green_ctx_2.sm_ids}")

    with green_ctx_1.with_context():
        print("SM utilizations for shard 1 ({SM ID: usage count})")
        print(count_sm_ids(200))

    with green_ctx_2.with_context():
        print("SM utilizations for shard 2 ({SM ID: usage count})")
        print(count_sm_ids(200))

    # dispatch in parallel using streams
    print("Dispatching in parallel using streams")
    stream_1 = green_ctx_1.make_stream()
    stream_2 = green_ctx_2.make_stream()

    with green_ctx_1.with_context():
        num_sm = 200
        ids_1 = np.zeros(num_sm, dtype=np.int32)
        ids_2 = np.zeros(num_sm, dtype=np.int32)

        launch_smid(ids_1, num_sm, stream=stream_1)
        launch_smid(ids_2, num_sm, stream=stream_2)
        torch.cuda.synchronize()

        print("SM utilizations for stream 1 ({SM ID: usage count})")
        print(Counter(ids_1))
        print("SM utilizations for stream 2 ({SM ID: usage count})")
        print(Counter(ids_2))


def main():
    # example_shard()
    # benchmark_matmul()
    # multistream_dispatch()
    partition_two()


if __name__ == "__main__":
    init()
    main()
