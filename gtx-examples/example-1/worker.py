"""
Example client that allocates 8 SMs and performs a matrix multiplication of A @ B = C.
A is a 1024x1024 bf16 matrix exclusively allocated.
B is a 1024x2 bf16 matrix shared by all workers by name "B".
C is the local GPU memory.

We will run the matmul many times and time it.
"""

import torch
import torch.multiprocessing as mp
from green_ctx.client.client import GPUClient


def main(mp_barrier: mp.Barrier):
    # Create a client connection
    client = GPUClient(host="localhost", port=50051)

    torch.set_default_device("cuda")

    try:
        status = client.health_check()
        print(f"Server status: {status}")

        alloc_uuid, gtx = client.request_exclusive_SMs(8)
        gtx.enter()
        print(f"Allocated SMs: {gtx.sm_ids}")

        # Allocate a tensor
        A = client.alloc_tensor(
            shape=[1024, 61440],
            dtype="bfoat16",
        )
        print(f"tensor B already exists: {client.exists_tensor('B')}")
        B = client.alloc_tensor(
            shape=[61440, 1024],
            dtype="bfoat16",
            name="B",
            get_if_exists=True,
        )
        C = torch.empty([1024, 1024], dtype=torch.bfloat16)

        # Perform the matmul and time it using cuda event
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100):
            torch.matmul(A, B, out=C)
        end.record()
        torch.cuda.synchronize()
        print(f"Avg time for each matmul: {start.elapsed_time(end)/100} ms")

        # Clean up resources
        print("\nCleaning up resources...")
        gtx.exit()
        client.free_SMs(alloc_uuid)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        client.close()

def run_multiple_workers():
    num_workers = 8
    barrier = mp.Barrier(num_workers)

    workers = []
    for _ in range(num_workers):
        p = mp.Process(target=main, kwargs={"mp_barrier": barrier})
        p.start()
        workers.append(p)

    for w in workers:
        w.join()

if __name__ == "__main__":
    # main()
    run_multiple_workers()