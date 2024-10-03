from concurrent.futures import ThreadPoolExecutor
import ctypes
from cuda import cuda
import os
import torch
from green_ctx import init, make_shard, CHECK_CUDA


def matmul_kernel(green_ctx_obj, a, b):
    with green_ctx_obj.with_context():
        torch.matmul(a, b)


if __name__ == '__main__':
    cuda_device = os.getenv('CUDA_DEVICE', 0)
    device = torch.device(f'cuda:{cuda_device}')

    init()
    green_ctx_obj = make_shard(1, 64)
    stream = CHECK_CUDA(
        cuda.cuGreenCtxStreamCreate(
            green_ctx_obj.raw_context, cuda.CUstream_flags.CU_STREAM_NON_BLOCKING, 0
        )
    )

    stream_ptr = stream.getPtr()
    torch.cuda.synchronize()
    print('stream ptr:', stream_ptr)
    torch_stream = torch.cuda.ExternalStream(stream_ptr, device=device)
    torch.cuda.synchronize()
    print('torch stream:', torch_stream)

    # normal_stream = torch.cuda.Stream()
    # print('normal stream:', normal_stream)
    # with torch.cuda.stream(normal_stream):
    #     print('using normal stream')
    #     a = torch.tensor([1, 2, 3], device=device)
    
    with green_ctx_obj.with_context():
        print('using context')
        # a = torch.tensor([1, 2, 3], device=device)
        # b = torch.tensor([-1, -2, -3], device=device)
        
        with torch.cuda.stream(torch_stream):
            print('using stream')
            a = torch.tensor([1, 2, 3], device=device)
            
            # a = torch.randn(1000, 1000, dtype=torch.bfloat16, device=device)
            # b = torch.randn(1000, 1000, dtype=torch.bfloat16, device=device)
            # torch.matmul(a, b)

    # with ThreadPoolExecutor(max_workers=2) as executor:
    #     # Submit tasks for both matrix multiplications to run concurrently in separate streams
    #     future1 = executor.submit(matmul_kernel, green_ctx_obj, a, b)
    #     future2 = executor.submit(matmul_kernel, green_ctx_obj, a, b)

    #     # Wait for both tasks to complete
    #     future1.result()
    #     future2.result()