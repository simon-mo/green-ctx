import torch
import torch.multiprocessing as mp
from green_ctx.kernels import run_crash_kernel
import time

def run_process(process_id: int, shared_dict, barrier):
    print(f"Process {process_id}: Initialized CUDA context")

    # Create and run a simple matmul to show context is working
    a = torch.randn(256, 256, device='cuda')
    b = torch.randn(256, 256, device='cuda')
    c = torch.matmul(a, b)
    torch.cuda.synchronize()

    print(f"Process {process_id}: Successfully ran initial matmul")

    # Wait for both processes to be ready
    barrier.wait()

    # Process 0 will trigger the crash
    if process_id == 0:
        print(f"Process {process_id}: Triggering crash...")
        run_crash_kernel()
    else:
        # Process 1 waits a bit then tries to use CUDA
        time.sleep(2)  # Wait for crash to occur

    # Try to run matmul again - should fail for both processes
    print(f"Process {process_id}: Attempting matmul after crash...")
    try:
        d = torch.matmul(a, b)
        torch.cuda.synchronize()
    except Exception as e:
        print(f"Process {process_id}: Error: {str(e)}")
    else:
        print(f"Process {process_id}: Successfully ran matmul after crash")

def main():
    mp.set_start_method('spawn')

    # Create shared memory manager
    manager = mp.Manager()
    shared_dict = manager.dict()
    barrier = mp.Barrier(2)

    # Create processes
    processes = []
    for i in range(2):
        p = mp.Process(target=run_process, args=(i, shared_dict, barrier))
        processes.append(p)
        p.start()

    # Wait for completion
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
