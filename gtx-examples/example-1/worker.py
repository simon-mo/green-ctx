"""
Example client that allocates 16 SMs and performs a matrix multiplication of A @ B = C.
A is a 1024x1024 bf16 matrix exclusively allocated.
B is a 1024x2 bf16 matrix shared by all workers by name "B".
C is the local GPU memory.

We will run the matmul many times and time it.
"""

import time

import numpy as np
from green_ctx.client.client import GPUClient


def main():
    # Create a client connection
    client = GPUClient(host="localhost", port=50051)

    try:
        status = client.health_check()
        print(f"Server status: {status}")

        allocated_sms = client.request_exclusive_SMs(8)
        print(f"Allocated SMs: {allocated_sms}")

        # Allocate a tensor
        tensor_info = client.alloc_tensor(
            shape=[1024, 1024],
            dtype="bfoat16",
        )


        # Simulate some work
        print("\nSimulating computation work...")
        time.sleep(2)

        # Clean up resources
        print("\nCleaning up resources...")
        client.free_SMs(allocated_sms)

        print("\nAll resources cleaned up successfully!")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    main()
