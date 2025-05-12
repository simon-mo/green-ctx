import torch
from green_ctx import make_shard, init
from green_ctx.kernels import count_sm_ids, compile_kernel, simd_code, CHECK_CUDA, cuda
import numpy as np
from collections import Counter


# Launch SMID kernel
def launch_smid(h_sm_ids, d_sm_ids, stream, num_sm):
    assert h_sm_ids.shape == d_sm_ids.shape

    # Launch kernel
    threads_per_block = 1
    blocks_per_grid = num_sm

    kernel = compile_kernel(simd_code)

    # Prepare kernel arguments
    d_sm_ids_arg = np.array([d_sm_ids.data_ptr()], dtype=np.uint64)
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
        cuda.cuMemcpyDtoHAsync(h_sm_ids.data_ptr(), d_sm_ids.data_ptr(),
                               h_sm_ids.numel() * h_sm_ids.element_size(),
                               stream))


if __name__ == "__main__":
    init()

    print("init")
    print(f"active sms: {len(count_sm_ids(200))}")

    print("shard 8")
    shard = make_shard(8)
    with shard.with_context() as ctx:
        print(f"active sms: {len(count_sm_ids(200))}")

    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    h_sm_ids = torch.zeros(200, device="cpu", dtype=torch.int32)
    d_sm_ids = torch.zeros(200, device="cuda", dtype=torch.int32)
    with torch.cuda.graph(graph, stream=stream):
        launch_smid(h_sm_ids, d_sm_ids, stream.cuda_stream, 200)

    graph.replay()
    torch.cuda.synchronize()
    print(
        f"graph default context: {len(Counter(h_sm_ids.cpu().numpy().tolist()))}"
    )

    h_sm_ids.zero_()
    d_sm_ids.zero_()
    torch.cuda.synchronize()

    with shard.with_context() as ctx:
        graph.replay()
        torch.cuda.synchronize()
        print(
            f"graph shard context: {len(Counter(h_sm_ids.cpu().numpy().tolist()))}"
        )

    # capture new graph under shard context
    graph_2 = torch.cuda.CUDAGraph()
    with shard.with_torch_stream() as stream:
        with torch.cuda.graph(graph_2, stream=stream):
            launch_smid(h_sm_ids, d_sm_ids, stream.cuda_stream, 200)

    graph_2.replay()
    torch.cuda.synchronize()
    print(
        f"graph shard context standalone under 8: {len(Counter(h_sm_ids.cpu().numpy().tolist()))}"
    )

    # capture several graphs under their own SMs
    SMs = [8, 16, 32, 64, 128]
    graphs = []
    for sm in SMs:
        shard = make_shard(sm)
        with shard.with_context() as ctx:
            graph = torch.cuda.CUDAGraph()
            with shard.with_torch_stream() as stream:
                with torch.cuda.graph(graph, stream=stream):
                    launch_smid(h_sm_ids, d_sm_ids, stream.cuda_stream, 200)
            graphs.append(graph)

    global_shard = make_shard(72)
    for sm, graph in zip(SMs, graphs):
        graph.replay()
        torch.cuda.synchronize()
        print(
            f"graph captured under {sm} run globally: {len(Counter(h_sm_ids.cpu().numpy().tolist()))}"
        )

        with global_shard.with_context() as ctx:
            graph.replay()
            torch.cuda.synchronize()
            print(
                f"graph captured under {sm} run under global shard (72): {len(Counter(h_sm_ids.cpu().numpy().tolist()))}"
            )
