#!/usr/bin/env python3
"""
Example script demonstrating the GPU multiplexer client API usage.
This example shows how to:
1. Connect to the server
2. Request exclusive SMs
3. Allocate and manage tensors
4. Clean up resources
"""

import time
from green_ctx.client.client import GPUClient
import numpy as np

def main():
    # Create a client connection
    client = GPUClient(host="localhost", port=5555)

    try:
        # Check server health
        print("Checking server health...")
        status = client.health_check()
        print(f"Server status: {status}")

        # Request exclusive SMs
        print("\nRequesting 4 SMs...")
        allocated_sms = client.request_exclusive_SMs(4)
        print(f"Allocated SMs: {allocated_sms}")

        # Allocate a tensor
        print("\nAllocating tensor...")
        tensor_info = client.alloc_tensor(
            shape=[1000, 1000],
            dtype="float32",
            name="example_matrix"
        )
        print(f"Allocated tensor: {tensor_info}")

        # Get tensor info
        print("\nRetrieving tensor info...")
        tensor_info = client.get_tensor("example_matrix")
        print(f"Retrieved tensor info: {tensor_info}")

        # Simulate some work
        print("\nSimulating computation work...")
        time.sleep(2)

        # Clean up resources
        print("\nCleaning up resources...")
        client.free_tensor("example_matrix")
        client.free_SMs(allocated_sms)

        print("\nAll resources cleaned up successfully!")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    main()