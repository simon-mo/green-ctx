"""
Benchmarking how fast can we switch between contexts.
"""

import torch
from green_ctx.client.client import GPUClient


def main():
    # Create a client connection
    client = GPUClient(host="localhost", port=50051)

    torch.set_default_device("cuda")

    try:
        status = client.health_check()
        print(f"Server status: {status}")

        # Requests 10 different contexts
        alloc_uuids = []
        gtx_list = []
        for _ in range(10):
            alloc_uuid, gtx = client.request_exclusive_SMs(8)
            gtx_list.append(gtx)
            alloc_uuids.append(alloc_uuid)
            print(f"Allocated SMs: {gtx.sm_ids}")

        A = torch.randn([1024, 61440], dtype=torch.bfloat16)
        B = torch.randn([61440, 1024], dtype=torch.bfloat16)
        C = torch.empty([1024, 1024], dtype=torch.bfloat16)

        # warmup the entire GPU
        for _ in range(100):
            torch.matmul(A, B, out=C)

        first_gtx = gtx_list[0]
        first_gtx.enter()

        # Perform the matmul and time it using cuda events for one context
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100):
            torch.matmul(A, B, out=C)
        end.record()
        torch.cuda.synchronize()
        first_gtx.exit()
        print(f"Total time with first GTX: {start.elapsed_time(end)} ms")

        # Cycle over all contexts and time the matmul
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(10):
            for gtx in gtx_list:
                gtx.enter()
                torch.matmul(A, B, out=C)
                gtx.exit()
        end.record()
        torch.cuda.synchronize()
        print(f"Total time with all GTXs: {start.elapsed_time(end)} ms")
        # Clean up resources

        print("\nCleaning up resources...")
        for alloc_uuid in alloc_uuids:
            client.free_SMs(alloc_uuid)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    main()