# The library provide functions to leverage CUDA dynamic control flow
# to select different kernels under SM constraints using switch node.

import torch
from cuda import cuda, cudart
from green_ctx.utils import CHECK_CUDA, CHECK_CUDART
from green_ctx.kernels import compile_kernel
from green_ctx import make_shard, init
import numpy as np

init()

enter_switch_kernel_string = r"""
typedef __device_builtin__ unsigned long long cudaGraphConditionalHandle;
extern "C" __device__ __cudart_builtin__ void cudaGraphSetConditional(cudaGraphConditionalHandle handle, unsigned int value);

extern "C" __global__
void switch_handle_setter(int *dPtr, cudaGraphConditionalHandle handle)
{
    // atomically increment the value
    int old_value = atomicAdd(dPtr, 1);
    if (old_value == 0) {
        // Sole tenant, use the first kernel
        cudaGraphSetConditional(handle, 1);
    } else {
        // Shared tenant, use the second kernel
        cudaGraphSetConditional(handle, 2);
    }
}
"""
# nts: the scheduling bubble between this kernel and the switch node itself is about 10us.
switch_handle_setter_kernel = compile_kernel(enter_switch_kernel_string,
                                             "switch_handle_setter")

exit_kernel_string = r"""
extern "C" __global__
void exit_kernel(int *dPtr)
{
    // atomically decrement the value
    atomicAdd(dPtr, -1);
}
"""
exit_kernel = compile_kernel(exit_kernel_string, "exit_kernel")

select_buf = None


def get_or_init_select_buf():
    global select_buf
    if select_buf is None:
        select_buf = torch.zeros(1, dtype=torch.int32, device="cuda")
    return select_buf


def wrap_op_with_switch(op, num_sm):
    select_buf = get_or_init_select_buf()

    graph = torch.cuda.CUDAGraph()

    graph.enable_debug_mode()

    capture_stream = torch.cuda.Stream()

    with torch.cuda.stream(capture_stream):
        graph.capture_begin()
        capture_stream_handle = capture_stream.cuda_stream

        err, capture_status, _, graph_handle, _, _ = cudart.cudaStreamGetCaptureInfo(
            capture_stream_handle)
        assert err == cudart.cudaError_t.cudaSuccess
        assert capture_status == cudart.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive

        conditional_handle = CHECK_CUDART(
            cudart.cudaGraphConditionalHandleCreate(graph_handle, 0, 0))

        # 1. Capture the switch handle setter
        first_arg = np.array([select_buf.data_ptr()], dtype=np.uint64)
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
                capture_stream_handle,  # stream (using default stream)
                args.ctypes.data,  # kernel arguments
                0  # extra (ignore)
            ))

        # 2. Add the conditional node
        err, _, _, _, dependencies, _ = cudart.cudaStreamGetCaptureInfo(
            capture_stream_handle)
        assert err == cudart.cudaError_t.cudaSuccess
        assert len(dependencies) == 1

        cond_node_params = cudart.cudaGraphNodeParams()
        cond_node_params.type = cudart.cudaGraphNodeType.cudaGraphNodeTypeConditional
        cond_node_params.conditional.handle = conditional_handle
        cond_node_params.conditional.type = cudart.cudaGraphConditionalNodeType.cudaGraphCondTypeSwitch
        cond_node_params.conditional.size = 3
        CHECK_CUDART(
            cudart.cudaGraphAddNode(graph_handle, dependencies,
                                    len(dependencies), cond_node_params))

        # Use a separate stream for sub-graph capture
        for i, num_sms in enumerate([132, num_sm]):
            with make_shard(num_sms).with_context():
                body_stream = torch.cuda.Stream()
                with torch.cuda.stream(body_stream):
                    subgraph = cond_node_params.conditional.phGraph_out[i + 1]
                    CHECK_CUDART(
                        cudart.cudaStreamBeginCaptureToGraph(
                            body_stream.cuda_stream, subgraph, None, None, 0,
                            cudart.cudaStreamCaptureMode.
                            cudaStreamCaptureModeGlobal))
                    op()

                    # 3. Capture the exit kernel
                    first_arg = np.array([select_buf.data_ptr()],
                                         dtype=np.uint64)
                    args = np.array([first_arg.ctypes.data], dtype=np.uint64)
                    CHECK_CUDA(
                        cuda.cuLaunchKernel(
                            exit_kernel,
                            1,  # grid x dim
                            1,  # grid y dim
                            1,  # grid z dim
                            1,  # block x dim
                            1,  # block y dim
                            1,  # block z dim
                            0,  # dynamic shared memory
                            body_stream.
                            cuda_stream,  # stream (using default stream)
                            args.ctypes.data,  # kernel arguments
                            0  # extra (ignore)
                        ))
                CHECK_CUDART(
                    cudart.cudaStreamEndCapture(body_stream.cuda_stream))

        graph.capture_end()

    return graph


if __name__ == "__main__":
    import time
    a = torch.randn((1024, 1024), device="cuda")

    def matmul_op():
        torch.matmul(a, a)

    graph = wrap_op_with_switch(matmul_op, 64)
    graph.debug_dump("./debug_dump.txt")
    graph.replay()
