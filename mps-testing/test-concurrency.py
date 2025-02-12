import torch
import torch.multiprocessing as mp
from typing import List
import statistics
import numpy as np
from green_ctx import init, partition
from green_ctx.kernels import run_crash_kernel

test_crash = True

def matmul_benchmark(thread_id: int, shared_dict, barrier):

    init()
    ctx = partition(8, 8)[thread_id]
    ctx.enter()

    # Create random matrices
    a = torch.randn(256, 256, device='cuda')
    b = torch.randn(256, 256, device='cuda')

    # Warmup
    for _ in range(10):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize()

    # Wait for both processes to be ready
    barrier.wait()

    if test_crash and thread_id == 0:
        run_crash_kernel()

    # Benchmark
    times = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(100):
        c = torch.matmul(a, b)
    end_event.record()
    end_event.synchronize()

    times.append(start_event.elapsed_time(end_event))

    # Store results in shared memory
    shared_dict[thread_id] = times

def main():
    # Enable CUDA multiprocessing
    mp.set_start_method('spawn')

    num_processes = 2

    # Create shared memory manager
    manager = mp.Manager()
    shared_dict = manager.dict()
    barrier = mp.Barrier(num_processes)

    # Create processes
    processes = []
    for i in range(num_processes):
        p = mp.Process(target=matmul_benchmark, args=(i, shared_dict, barrier))
        processes.append(p)
        p.start()

    # Wait for completion
    for p in processes:
        p.join()

    # Convert shared dict to regular results list
    if test_crash:
        results = [shared_dict[i] for i in range(1, num_processes)]
    else:
        results = [shared_dict[i] for i in range(num_processes)]

    # Print results
    print("\nResults Summary:")
    print("-" * 50)
    for proc_id, times in enumerate(results):
        print(f"\nProcess {proc_id}: {times}")

if __name__ == "__main__":
    main()
