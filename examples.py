import torch
import numpy as np

from green_ctx import make_shard, init, partition, get_sms_in_range
from green_ctx.kernels import count_sm_ids, launch_smid
from green_ctx.timing import cuda_timing_decorator
from green_ctx.utils import print_current_context_id

from collections import Counter
import time

from cuda import cuda, cudart
from green_ctx.utils import CHECK_CUDA


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
    print(
        "---\nBenchmarking matrix multiplication (M=N=K=1024) with different number of SMs."
    )

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


def partition_with_torch():
    print("---\nExample showcasing two torch streams")
    green_ctx_1, green_ctx_2 = partition(32, 8)
    print(f"Created {green_ctx_1.sm_count} SMs for shard 1: {green_ctx_1.sm_ids}")
    print(f"Created {green_ctx_2.sm_count} SMs for shard 2. {green_ctx_2.sm_ids}")

    stream_1 = green_ctx_1.make_stream()
    stream_2 = green_ctx_2.make_stream()

    torch_stream_1 = torch.cuda.ExternalStream(int(stream_1))
    torch_stream_2 = torch.cuda.ExternalStream(int(stream_2))

    a = torch.randn(1024, 1024).cuda()
    b = torch.randn(1024, 1024).cuda()

    with green_ctx_1.with_context():
        with torch.cuda.stream(torch_stream_1):
            e_start_1, e_end_1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            e_start_1.record()
            c_1 = torch.matmul(a, b)
            e_end_1.record()

    with green_ctx_2.with_context():
        with torch.cuda.stream(torch_stream_2):
            e_start_2, e_end_2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            e_start_2.record()
            c_2 = torch.matmul(a, b)
            e_end_2.record()

    torch.cuda.synchronize()

    print(f"Stream 1: {e_start_1.elapsed_time(e_end_1)} ms")
    print(f"Stream 2: {e_start_2.elapsed_time(e_end_2)} ms")


def benchmark_set_context():
    print("---\nBenchmarking the overhead of setting a context")

    green_ctx = make_shard(8)
    push_times = []
    pop_times = []
    for _ in range(100):
        start = time.perf_counter_ns()
        with green_ctx.with_context():
            enter = time.perf_counter_ns()
        exit = time.perf_counter_ns()
        push_times.append(enter - start)
        pop_times.append(exit - enter)

    print(f"Average time to push context: {np.mean(push_times) * 1e-3:.2f} us")
    print(f"Average time to pop context: {np.mean(pop_times) * 1e-3:.2f} us")

def check_get_sms_in_range():
    print(("---\nGetting SMs in a range"))

    green_ctx = get_sms_in_range(0, 80)
    green_ctx2 = get_sms_in_range(80, 120)

    print("Green ctx1 SM count", green_ctx.sm_count)
    print("Green ctx2 SM count", green_ctx2.sm_count)

    overlaps = set(green_ctx.sm_ids) & set(green_ctx2.sm_ids)
    print("Overlap SM count", len(overlaps))
    if overlaps:
        print("Overlaps found in sm_ids", overlaps)
    else:
        print("No overlaps in sm_ids between the two contexts.")




def main():
    example_shard()
    benchmark_matmul()
    multistream_dispatch()
    partition_two()
    partition_with_torch()
    benchmark_set_context()
    check_get_sms_in_range()


if __name__ == "__main__":
    init()
    main()
