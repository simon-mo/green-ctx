# This script is to use to show case the ability to leverage CUDA dynamic control flow
# to select different kernels under SM constraints using switch node.

import torch
from green_ctx.utils import set_cublas_sm_count
from cuda import cuda, cudart  # Import CUDA driver API
import ctypes  # For pointer types needed by CUDA API

torch.set_default_device("cuda")
torch.set_default_dtype(torch.bfloat16)


# --- Helper function to check CUDA errors and return results ---
def CHECK_CUDA(call_result):
    err = call_result[0] if isinstance(call_result, tuple) else call_result
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA error: {err}")
    # Return the rest of the tuple if it exists
    if isinstance(call_result, tuple) and len(call_result) > 1:
        return call_result[1:]
    # Return None or handle single-value returns if necessary,
    # but for the calls used here, we mostly expect tuples or just the error.
    # If a call like cuEventRecord returns just the error, CHECK_CUDA(err) works.
    return None


def CHECK_CUDART(call_result):
    if len(call_result) == 2:
        err, result = call_result
    elif len(call_result) > 2:
        err, *result = call_result
    else:
        err, = call_result
        result = None
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {err}")
    return result


# -------------------------------------------------------------

# ------------------------------------------

# make a switch graph based on input memory state to select different kernels
# 1. 8 SMs
# 2. 16 SMs
# 3. 32 SMs
# 4. 64 SMs
# 5. 128 SMs

A = torch.randn(1024, 16384)
B = torch.randn(16384, 1024)
SM_COUNTS = [8, 16, 32, 64, 128]

# warmup
torch.matmul(A, B)
torch.cuda.synchronize()
print("--------------------")

for sm_count in SM_COUNTS:
    with set_cublas_sm_count(sm_count):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(1):
            torch.matmul(A, B)
        end_event.record()
        torch.cuda.synchronize()
        print(
            f"SM count: {sm_count}, time taken: {start_event.elapsed_time(end_event)/1:.4f} ms"
        )

print("--------------------")

# # Create CUDA event handles using the driver API
cuda_event_pairs = []
for _ in range(len(SM_COUNTS)):
    start_event, = CHECK_CUDA(
        cuda.cuEventCreate(0))  # 0 corresponds to cudaEventDefault
    end_event, = CHECK_CUDA(cuda.cuEventCreate(0))
    cuda_event_pairs.append((start_event, end_event))

stream = torch.cuda.Stream()
graph = torch.cuda.CUDAGraph()

# Capture graph
with torch.cuda.stream(stream):
    graph.capture_begin()
    stream_handle = stream.cuda_stream  # Get the underlying cudaStream_t handle

    for i, sm_count in enumerate(SM_COUNTS):
        start_event, end_event = cuda_event_pairs[i]
        with set_cublas_sm_count(sm_count):
            # Record events using CUDA driver API with cudaEventBlockingSync flag
            CHECK_CUDA(
                cuda.cuEventRecordWithFlags(start_event, stream_handle,
                                            1))  # Use flag 1
            torch.matmul(A, B)
            CHECK_CUDA(cuda.cuEventRecordWithFlags(end_event, stream_handle,
                                                   1))  # Use flag 1

    graph.capture_end()

# Replay graph
graph.replay()
torch.cuda.synchronize()

# Measure and print elapsed time using CUDA driver API
for i, sm_count in enumerate(SM_COUNTS):
    start_event, end_event = cuda_event_pairs[i]
    # Remove the explicit synchronization, rely on replay+torch.sync and blocking flag
    # Correct call for cuda-python: returns (err, time_ms)
    time_ms, = CHECK_CUDA(cuda.cuEventElapsedTime(start_event, end_event))
    print(f"SM count: {sm_count}, time taken: {time_ms:.4f} ms")

# Clean up CUDA events
for start_event, end_event in cuda_event_pairs:
    CHECK_CUDA(cuda.cuEventDestroy(start_event))
    CHECK_CUDA(cuda.cuEventDestroy(end_event))

# Section above always works, and is the baseline.
# --------------------------------------------
# Now we implement the switch graph
print("--------------------")

from green_ctx.kernels import compile_kernel
import numpy as np

kernel_string = r"""
typedef __device_builtin__ unsigned long long cudaGraphConditionalHandle;
extern "C" __device__ __cudart_builtin__ void cudaGraphSetConditional(cudaGraphConditionalHandle handle, unsigned int value);

extern "C" __global__
void switch_handle_setter(char *dPtr, cudaGraphConditionalHandle handle)
{
    unsigned int value = *dPtr;
    cudaGraphSetConditional(handle, value);
}
"""
# nts: this seems to take about 10us.
switch_handle_setter_kernel = compile_kernel(kernel_string,
                                             "switch_handle_setter")

select_buf = torch.ones(1, dtype=torch.int32, device="cuda")
start_event, end_event = torch.cuda.Event(
    enable_timing=True), torch.cuda.Event(enable_timing=True)

graph = torch.cuda.CUDAGraph()
# graph.enable_debug_mode()

stream = torch.cuda.Stream()
with torch.cuda.stream(stream):
    graph.capture_begin()
    stream_handle = stream.cuda_stream

    err, capture_status, _, graph_handle, _, _ = cudart.cudaStreamGetCaptureInfo(
        stream_handle)
    assert err == cudart.cudaError_t.cudaSuccess
    assert capture_status == cudart.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive

    conditional_handle = CHECK_CUDART(
        cudart.cudaGraphConditionalHandleCreate(graph_handle, 0, 0))

    # 1. Capture the switch handle setter
    # Prepare kernel arguments
    first_arg = np.array([select_buf.data_ptr()], dtype=np.uint64)
    # second_arg = np.array([conditional_handle.getPtr()], dtype=np.uint64)
    args = np.array([first_arg.ctypes.data,
                     conditional_handle.getPtr()],
                    dtype=np.uint64)
    CHECK_CUDA(
        cuda.cuLaunchKernel(
            switch_handle_setter_kernel,
            1,  # grid x dim
            1,  # grid y dim
            1,  # grid z dim
            1,  # block x dim
            1,  # block y dim
            1,  # block z dim
            0,  # dynamic shared memory
            stream_handle,  # stream (using default stream)
            args.ctypes.data,  # kernel arguments
            0  # extra (ignore)
        ))

    # 2. Add the conditional node
    err, _, _, _, dependencies, _ = cudart.cudaStreamGetCaptureInfo(
        stream_handle)
    assert err == cudart.cudaError_t.cudaSuccess
    assert len(dependencies) == 1

    cond_node_params = cudart.cudaGraphNodeParams()
    cond_node_params.type = cudart.cudaGraphNodeType.cudaGraphNodeTypeConditional
    cond_node_params.conditional.handle = conditional_handle
    cond_node_params.conditional.type = cudart.cudaGraphConditionalNodeType.cudaGraphCondTypeSwitch
    cond_node_params.conditional.size = len(SM_COUNTS)
    CHECK_CUDART(
        cudart.cudaGraphAddNode(graph_handle, dependencies, len(dependencies),
                                cond_node_params))

    # 3. For each SM count, add a kernel node
    # Use a separate stream for sub-graph capture
    body_stream = torch.cuda.Stream()
    with torch.cuda.stream(body_stream):
        for i, sm_count in enumerate(SM_COUNTS):
            subgraph = cond_node_params.conditional.phGraph_out[i]
            CHECK_CUDART(
                cudart.cudaStreamBeginCaptureToGraph(
                    body_stream.cuda_stream, subgraph, None, None, 0,
                    cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal))
            with set_cublas_sm_count(sm_count):
                torch.matmul(A, B)
            CHECK_CUDART(cudart.cudaStreamEndCapture(body_stream.cuda_stream))

    graph.capture_end()

# graph.debug_dump("./debug_dump.txt")
# 4. Instantiate the graph
for i in range(len(SM_COUNTS)):
    select_buf[0] = i
    torch.cuda.synchronize()

    start_event.record()
    for _ in range(1):
        graph.replay()
    end_event.record()
    torch.cuda.synchronize()
    print(
        f"SM count: {SM_COUNTS[i]}, time taken: {start_event.elapsed_time(end_event)/1:.4f} ms"
    )
