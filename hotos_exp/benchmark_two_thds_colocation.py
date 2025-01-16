"""
Setup: Seems to only work on H100 GPUs.

Run: python benchmark_two_thds_colocation.py

Explained:
This script sets up a preliminary experiment for the HotOS submission, focusing on GPU resource partitioning. We co-locate two threads, APP 1 and APP 2, on the same GPU, with each thread continuously performing matrix multiplication for 100ms. The threads are assigned to two separate green contexts, and context switching occurs every 33ms, dividing the experiment into three distinct stages.

In the first stage, APP 1 is allocated a low number of SMs (16), while APP 2 is allocated a high number of SMs (88). Upon transitioning to the second stage, the resource allocation is reversed: APP 1 receives high SMs and APP 2 low SMs. During this transition, we measure the GPU repartitioning time, which is primarily determined by the delay caused by pending CUDA kernels in the task queue and waiting the running CUDA kernel on GPU to finish.

In the third stage, the SM allocation is switched back to the initial configuration (APP 1 with low SMs and APP 2 with high SMs), and the GPU repartitioning time is measured again.
"""

import time
import threading

import torch

from green_ctx import partition

# Total execution duration in s.
DURATION = 100 / 1000
# Number of stages.
NUM_STAGES = 3  # Each thread switches SM allocation twice.
# Matrix size. Each thread computes a x b = c, where a, b, c are in SIZE x SIZE.
SIZE = 2048
# Each thread switches between low and high SMs.
NUM_SMS_LOW = 16
NUM_SMS_HIGH = 88


def create_gctxs():
    """Create green ctxes for two threads."""
    setup1 = partition(NUM_SMS_LOW, NUM_SMS_HIGH)
    setup2 = partition(NUM_SMS_HIGH, NUM_SMS_LOW)

    gctxs = list(zip(setup1, setup2))
    return gctxs


def busy_sleep_us(us: int):
    start = time.perf_counter_ns()
    end = start + (us * 1000)
    while time.perf_counter_ns() < end:
        pass


def matmul_thd(tid, gctxs, thd_event):
    # Intialize the GPU
    stream = torch.cuda.Stream()
    cuda_event = torch.cuda.Event()

    # Prepare matrices
    a = torch.randn(SIZE, SIZE, device="cuda")
    b = torch.randn(SIZE, SIZE, device="cuda")
    c = torch.empty(SIZE, SIZE, device="cuda")

    def warmup(num_repeat: int = 5):
        tic = time.perf_counter()
        with torch.cuda.stream(stream):
            with gctxs[0].with_context():
                [torch.matmul(a, b, out=c) for _ in range(num_repeat)]
            cuda_event.record()
            cuda_event.synchronize()
        toc = time.perf_counter()
        avg_dur = (toc - tic) * 1000 / num_repeat
        print(f"warmup matmul exec: {avg_dur}ms")

        thd_event.set()
        thd_event.wait()
        thd_event.clear()

    warmup()

    STAGE_DUR = DURATION / NUM_STAGES
    rnd = 0
    tic = time.perf_counter()
    with torch.cuda.stream(stream):
        while (time.perf_counter() - tic) < DURATION:
            with gctxs[rnd % len(gctxs)].with_context():
                while (time.perf_counter() - tic) < STAGE_DUR * (rnd + 1):
                    torch.matmul(a, b, out=c)
                    busy_sleep_us(4000)

            cuda_event.record()
            cuda_event.synchronize()

            actual_dur = (time.perf_counter() - tic) * 1000
            expect_dur = STAGE_DUR * (rnd + 1) * 1000

            print(
                f"Thread {tid} stage {rnd} "
                f"actual duration: {actual_dur:.2f}ms, "
                f"expected duration {expect_dur:.2f}ms, "
                f"switch latency {actual_dur - expect_dur:.2f}ms"
            )

            thd_event.set()
            thd_event.wait()
            thd_event.clear()

            # Switch to the next green context.
            rnd += 1


def main():
    thd_gctxs = create_gctxs()
    thd_event = threading.Event()

    thds = []
    for i in range(2):
        print(f"Starting thread {i}")
        thd = threading.Thread(target=matmul_thd, args=(i, thd_gctxs[i], thd_event))
        thd.start()
        thds.append(thd)

    for thd in thds:
        thd.join()


if __name__ == "__main__":
    main()
