# this is the python version of test.cu using cuda-python api

from cuda import cuda, nvrtc
import torch
import numpy as np
import ctypes

# Define CHECK_CUDA function
def _cudaGetErrorEnum(error):
    if isinstance(error, cuda.CUresult):
        err, name = cuda.cuGetErrorName(error)
        return name if err == cuda.CUresult.CUDA_SUCCESS else "<unknown>"
    elif isinstance(error, nvrtc.nvrtcResult):
        return nvrtc.nvrtcGetErrorString(error)[1]
    else:
        raise RuntimeError('Unknown error type: {}'.format(error))

def CHECK_CUDA(result):
    if result[0].value:
        raise RuntimeError("CUDA error code={}({})".format(result[0].value, _cudaGetErrorEnum(result[0])))
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]

def cuda_timing_decorator(func):
    def wrapper(*args, **kwargs):
        # Create start and stop events for timing
        start_event = torch.cuda.Event(enable_timing=True)
        stop_event = torch.cuda.Event(enable_timing=True)

        # Record the start event
        start_event.record()

        # Execute the function (e.g., a kernel or operation)
        func(*args, **kwargs)

        # Record the stop event and synchronize
        stop_event.record()
        torch.cuda.synchronize()

        # Measure elapsed time
        milliseconds = start_event.elapsed_time(stop_event)

        return milliseconds
    
    return wrapper

# Define CUDA kernel
kernel_code = """
__device__ unsigned int __smid(void) {
     unsigned int ret;
     asm("mov.u32 %0, %%smid;" : "=r"(ret));
     return ret;
}

extern "C" __global__ void write_smid(int *d_sm_ids, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_sm_ids[idx] = __smid();
    }
}
"""

# Compile the CUDA kernel
def compile_kernel():
    prog = CHECK_CUDA(nvrtc.nvrtcCreateProgram(str.encode(kernel_code), b"smid.cu", 0, [], []))
    opts = [b"--fmad=false", b"--gpu-architecture=sm_90"]
    try:
        CHECK_CUDA(nvrtc.nvrtcCompileProgram(prog, len(opts), opts))
    except RuntimeError as e:
        log_size = CHECK_CUDA(nvrtc.nvrtcGetProgramLogSize(prog))
        log = b" " * log_size
        CHECK_CUDA(nvrtc.nvrtcGetProgramLog(prog, log))
        print(f"NVRTC Compilation Error:\n{log.decode()}")
        raise e

    ptxSize = CHECK_CUDA(nvrtc.nvrtcGetPTXSize(prog))
    ptx = b" " * ptxSize
    CHECK_CUDA(nvrtc.nvrtcGetPTX(prog, ptx))

    ptx = np.char.array(ptx)
    module = CHECK_CUDA(cuda.cuModuleLoadData(ptx.ctypes.data))
    return CHECK_CUDA(cuda.cuModuleGetFunction(module, b"write_smid"))

# Launch SMID kernel
def launch_smid(h_sm_ids, num_sm, stream=None):
    bytes_size = num_sm * ctypes.sizeof(ctypes.c_int)

    # Allocate device memory
    d_sm_ids = CHECK_CUDA(cuda.cuMemAlloc(bytes_size))

    # Copy input data to device
    CHECK_CUDA(cuda.cuMemcpyHtoDAsync(d_sm_ids, h_sm_ids.ctypes.data, bytes_size, stream))

    # Launch kernel
    threads_per_block = 1
    blocks_per_grid = num_sm

    kernel = compile_kernel()

    # Prepare kernel arguments
    d_sm_ids_arg = np.array([int(d_sm_ids)], dtype=np.uint64)
    num_sm_arg = np.array([num_sm], dtype=np.uint64)

    args = [d_sm_ids_arg, num_sm_arg]
    args = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)
    CHECK_CUDA(cuda.cuLaunchKernel(
        kernel,
        blocks_per_grid,  # grid x dim
        1,  # grid y dim
        1,  # grid z dim
        threads_per_block,  # block x dim
        1,  # block y dim
        1,  # block z dim
        0,  # dynamic shared memory
        stream,  # stream (using default stream)
        args.ctypes.data,  # kernel arguments
        0  # extra (ignore)
    ))

    # Copy result back to host
    CHECK_CUDA(cuda.cuMemcpyDtoHAsync(h_sm_ids.ctypes.data, d_sm_ids, bytes_size, stream))

    # Free device memory
    CHECK_CUDA(cuda.cuMemFree(d_sm_ids))

def create_green_ctx(device, sm_request):
    # Get SM resource
    sm_resource = CHECK_CUDA(cuda.cuDeviceGetDevResource(device, cuda.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM))
    # print(f"SM Resource: {sm_resource.sm.smCount}")

    # Split the SM resource
    result_resources, nb_groups, remaining = CHECK_CUDA(cuda.cuDevSmResourceSplitByCount(1, sm_resource,
                                                cuda.CUdevSmResourceSplit_flags.CU_DEV_SM_RESOURCE_SPLIT_MAX_POTENTIAL_CLUSTER_SIZE,
                                                sm_request))
    # print(f"Number of groups created: {nb_groups}")

    # for i in range(nb_groups):
    #     print(f"Group {i}: {result_resources[i].sm.smCount} SMs")

    # print(f"Remaining SMs: {remaining.sm.smCount}")

    desc = CHECK_CUDA(cuda.cuDevResourceGenerateDesc([result_resources[0]], 1))
    green_ctx = CHECK_CUDA(cuda.cuGreenCtxCreate(desc, device, cuda.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM))
    green_sm_resource = CHECK_CUDA(cuda.cuGreenCtxGetDevResource(green_ctx, cuda.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM))
    # print(f"Green SM Resource: {green_sm_resource.sm.smCount}")

    green_ctx_ctx = CHECK_CUDA(cuda.cuCtxFromGreenCtx(green_ctx))
    return green_ctx_ctx, green_ctx

def workload_smid():
    num_sm = 200
    h_sm_ids = np.zeros(num_sm, dtype=np.int32)

    print(f"Launch {num_sm} SMs")

    launch_smid(h_sm_ids, num_sm, stream=None)

    # Sort SM IDs
    # h_sm_ids.sort()
    from collections import Counter
    print(Counter(sorted(h_sm_ids)))

    # print(" ".join(map(str, h_sm_ids)))


def benchmark_matmul(device, input_sizes: list[int]):
    a = torch.randn(14336, 4096, dtype=torch.bfloat16).cuda()

    @cuda_timing_decorator
    def time_matmul(a, b):
        torch.matmul(a, b)

    ret = []
    for input_size in input_sizes:
        b = torch.randn(4096, input_size, dtype=torch.bfloat16).cuda() # prefill size
        torch.cuda.synchronize()

        for sm_cnt in [8, 16, 32, 48, 64, 96, 128, 132]:
            primary_ctx, green_ctx = create_green_ctx(device, sm_cnt)

            # Create a stream for the green context
            CHECK_CUDA(cuda.cuCtxSetCurrent(primary_ctx))

            # warmup
            # for some input sizes, sm_cnt=8 will throw cuda error `CUBLAS_STATUS_EXECUTION_FAILED`
            try:
                for _ in range(3):
                    time_matmul(a, b)
            except RuntimeError as err:
                ret.append((input_size, sm_cnt, err))
                continue

            timings = [time_matmul(a, b) for _ in range(5)]
            ret.append((input_size, sm_cnt, np.mean(timings)))

    return [tuple(map(str, tup)) for tup in ret]

def benchmark_embedding(device, input_sizes: list[int]):
    weight = torch.randn(128256, 4096).cuda()

    @cuda_timing_decorator
    def time_embedding(weight, indices):
        torch.embedding(weight, indices)

    ret = []
    for input_size in input_sizes:
        indices = torch.randint(0, 128256, (input_size,)).cuda()
        torch.cuda.synchronize()

        for sm_cnt in [8, 16, 32, 48, 64, 96, 128, 132]:
            primary_ctx, green_ctx = create_green_ctx(device, sm_cnt)

            # Create a stream for the green context
            CHECK_CUDA(cuda.cuCtxSetCurrent(primary_ctx))

            # warmup
            for _ in range(3):
                time_embedding(weight, indices)

            timings = [time_embedding(weight, indices) for _ in range(5)]
            ret.append((input_size, sm_cnt, np.mean(timings)))
    
    return [tuple(map(str, tup)) for tup in ret]

def benchmark_flash_attention(device, batch_sizes: list[int], seqlens: list[int]):
    from flash_attn import flash_attn_with_kvcache
    nheads = 32
    nkvheads = 8
    headdim = 128

    @cuda_timing_decorator
    def time_flash_attn(q, k_cache, v_cache):
        flash_attn_with_kvcache(q, k_cache, v_cache)

    ret = []
    for batch_size in batch_sizes:
        for seqlen in seqlens:
            q = torch.randn(batch_size, 1, nheads, headdim, dtype=torch.bfloat16, device='cuda')
            k_cache = torch.randn(batch_size, seqlen, nkvheads, headdim, dtype=torch.bfloat16, device='cuda')
            v_cache = torch.randn(batch_size, seqlen, nkvheads, headdim, dtype=torch.bfloat16, device='cuda')
            torch.cuda.synchronize()

            for sm_cnt in [8, 16, 32, 48, 64, 96, 128, 132]:
                primary_ctx, green_ctx = create_green_ctx(device, sm_cnt)

                # Create a stream for the green context
                CHECK_CUDA(cuda.cuCtxSetCurrent(primary_ctx))

                # warmup
                time_flash_attn(q, k_cache, v_cache)

                timings = [time_flash_attn(q, k_cache, v_cache) for _ in range(5)]
                ret.append((batch_size, seqlen, sm_cnt, np.mean(timings)))
    
    return [tuple(map(str, tup)) for tup in ret]

from torch import nn
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device='cuda'))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

def benchmark_rms_norm(device, batch_sizes: list[int], seqlens: list[int]):
    dim = 4096  # from llama3 ModelArgs.dim
    rms_norm = LlamaRMSNorm(dim)
    rms_norm.cuda()
    rms_norm = torch.compile(rms_norm)

    @cuda_timing_decorator
    def time_rms_norm(x):
        rms_norm(x)

    ret = []
    for batch_size in batch_sizes:
        for seqlen in seqlens:
            x = torch.randn(batch_size, seqlen, dim, device='cuda')
            torch.cuda.synchronize()

            for sm_cnt in [8, 16, 32, 48, 64, 96, 128, 132]:
                primary_ctx, green_ctx = create_green_ctx(device, sm_cnt)

                # Create a stream for the green context
                CHECK_CUDA(cuda.cuCtxSetCurrent(primary_ctx))

                # warmup
                time_rms_norm(x)

                timings = [time_rms_norm(x) for _ in range(5)]
                ret.append((batch_size, seqlen, sm_cnt, np.mean(timings)))
    
    return [tuple(map(str, tup)) for tup in ret]

def write_csv(results, kernel):
    with open(f'{kernel}_results.csv', 'w') as f:
        if kernel in ('matmul', 'embedding'):
            f.write('input size,SM count,milliseconds\n')
        else:
            f.write('batch size,seqlen,SM count,milliseconds\n')
        
        for tup in results:
            f.write(','.join(tup) + '\n')

def test_multiple_streams(device):
    green_primary_ctx, green_ctx = create_green_ctx(device, 8)
    green_primary_ctx_larger, green_ctx_larger = create_green_ctx(device, 128)

    CHECK_CUDA(cuda.cuCtxSetCurrent(green_primary_ctx))

    stream1 = CHECK_CUDA(cuda.cuGreenCtxStreamCreate(green_ctx, cuda.CUstream_flags.CU_STREAM_NON_BLOCKING, 0))
    stream2 = CHECK_CUDA(cuda.cuGreenCtxStreamCreate(green_ctx_larger, cuda.CUstream_flags.CU_STREAM_NON_BLOCKING, 0))

    h_sm_ids_stream_1 = np.zeros(200, dtype=np.int32)
    h_sm_ids_stream_2 = np.zeros(200, dtype=np.int32)

    launch_smid(h_sm_ids_stream_1, 200, stream=stream1)
    launch_smid(h_sm_ids_stream_2, 200, stream=stream2)

    torch.cuda.synchronize()

    from collections import Counter
    print(Counter(sorted(h_sm_ids_stream_1)))
    print(Counter(sorted(h_sm_ids_stream_2)))


def main():
    # CHECK_CUDA(cuda.cuInit(0))
    torch.cuda.init()

    # Initialize cublas somehow...
    a = torch.randn(1024, 1024).cuda()
    b = torch.randn(1024, 1024).cuda()
    torch.cuda.synchronize()
    torch.matmul(a, b)
    torch.cuda.synchronize()

    device = CHECK_CUDA(cuda.cuDeviceGet(0))
    context = CHECK_CUDA(cuda.cuCtxCreate(0, device))
    CHECK_CUDA(cuda.cuCtxSetCurrent(context))

    # green_primary_ctx, green_ctx = create_green_ctx(device, 8)
    # CHECK_CUDA(cuda.cuCtxSetCurrent(green_primary_ctx))
    # workload_smid()

    # try create two streams for the same green context, and dispatch the workload to them
    # test_multiple_streams(device)

    # these are 'prefill' sizes (?)
    input_sizes = [8, 1024, 2048, 4096, 8192, 16384, 32768]
    batch_sizes = [2**i for i in range(8)]
    
    matmul_res = benchmark_matmul(device, input_sizes)
    write_csv(matmul_res, 'matmul')
    
    embedding_res = benchmark_embedding(device, input_sizes)
    write_csv(embedding_res, 'embedding')
    
    flash_attn_res = benchmark_flash_attention(device, batch_sizes, input_sizes)
    write_csv(flash_attn_res, 'flash_attn')
    
    # decrease test range here due to large allocations
    rms_norm_res = benchmark_rms_norm(device, batch_sizes[:-1], input_sizes[:-1])
    write_csv(rms_norm_res, 'rms_norm')

    CHECK_CUDA(cuda.cuCtxDestroy(context))

if __name__ == "__main__":
    main()
