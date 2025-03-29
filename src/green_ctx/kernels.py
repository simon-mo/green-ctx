from cuda import nvrtc, cuda
from green_ctx.utils import CHECK_CUDA
import numpy as np
import ctypes
import torch
from collections import Counter

# Define CUDA kernel
simd_code = """
__device__ unsigned int __smid(void) {
     unsigned int ret;
     asm("mov.u32 %0, %%smid;" : "=r"(ret));
     return ret;
}

extern "C" __global__ void my_func(int *d_sm_ids, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_sm_ids[idx] = __smid();
    }
}
"""


# Compile the CUDA kernel
def compile_kernel(kernel_code: str, function_name: str = "my_func"):
    prog = CHECK_CUDA(
        nvrtc.nvrtcCreateProgram(str.encode(kernel_code), b"smid.cu", 0, [],
                                 []))
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
    return CHECK_CUDA(cuda.cuModuleGetFunction(module, function_name.encode()))


# Launch SMID kernel
def launch_smid(h_sm_ids, num_sm, stream=None):
    bytes_size = num_sm * ctypes.sizeof(ctypes.c_int)

    # Allocate device memory
    d_sm_ids = CHECK_CUDA(cuda.cuMemAlloc(bytes_size))

    # Copy input data to device
    CHECK_CUDA(
        cuda.cuMemcpyHtoDAsync(d_sm_ids, h_sm_ids.ctypes.data, bytes_size,
                               stream))

    # Launch kernel
    threads_per_block = 1
    blocks_per_grid = num_sm

    kernel = compile_kernel(simd_code)

    # Prepare kernel arguments
    d_sm_ids_arg = np.array([int(d_sm_ids)], dtype=np.uint64)
    num_sm_arg = np.array([num_sm], dtype=np.uint64)

    args = [d_sm_ids_arg, num_sm_arg]
    args = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)
    CHECK_CUDA(
        cuda.cuLaunchKernel(
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
    CHECK_CUDA(
        cuda.cuMemcpyDtoHAsync(h_sm_ids.ctypes.data, d_sm_ids, bytes_size,
                               stream))

    # Free device memory
    CHECK_CUDA(cuda.cuMemFree(d_sm_ids))


def count_sm_ids(num_sm: int):
    h_sm_ids = np.zeros(num_sm, dtype=np.int32)
    launch_smid(h_sm_ids, num_sm, stream=None)
    torch.cuda.synchronize()
    return Counter(sorted(h_sm_ids.tolist()))


crash_code = """
__device__ void __crash(void) {
    // Force illegal memory access by dereferencing null pointer
    volatile int* ptr = nullptr;
    *ptr = 42; // This will cause illegal memory access
}

extern "C" __global__ void my_func(void) {
    __crash();
}
"""


def run_crash_kernel(stream=None):
    kernel = compile_kernel(crash_code)
    CHECK_CUDA(
        cuda.cuLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, stream, None, 0))


timer_code = """
// Define uint64_t directly instead of including stdint.h
typedef unsigned long long uint64_t;

__device__ uint64_t __globaltimer(void) {
     uint64_t ret;
     asm("mov.u64 %0, %%globaltimer;" : "=l"(ret));
     return ret;
}

extern "C" __global__ void global_timer(uint64_t *d_timer, uint64_t offset) {
    *(d_timer + offset) = __globaltimer();
}
"""


def run_global_timer(timing_buffer_tensor, offset=0, stream=None):
    kernel = compile_kernel(timer_code, "global_timer")

    # Convert arguments to numpy arrays first
    ptr_arg = np.array([timing_buffer_tensor.data_ptr()], dtype=np.uint64)
    offset_arg = np.array([offset], dtype=np.uint64)

    # Create array of argument pointers
    args = np.array([ptr_arg.ctypes.data, offset_arg.ctypes.data],
                    dtype=np.uint64)

    CHECK_CUDA(
        cuda.cuLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, stream,
                            args.ctypes.data, 0))
    torch.cuda.synchronize()
