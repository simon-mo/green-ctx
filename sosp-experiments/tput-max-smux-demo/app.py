# if you want best tput for a given kernel, you might as well squeeze it and launch of 8x to fully max out things.

import torch
import rich

torch.set_default_device("cuda")
torch.set_default_dtype(torch.bfloat16)

M, K = 14336, 4096
# N_values = [8, 32, 64, 512, 2048, 8192, 16384]
N_values = [64]


def do_bench(fn):
    timings = []
    for _ in range(10):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))
    timings = timings[1:]
    return sum(timings) / len(timings)


def torch_gemm(A, B):
    for _ in range(200):
        C = torch.matmul(A, B)


def run_together(graphs, streams, events):
    for i, (graph, stream, event) in enumerate(zip(graphs, streams, events)):
        with torch.cuda.stream(stream):
            graph.replay()
            event.record()
    for event in events:
        event.wait()


for N in N_values:
    A = torch.randn(M, K, dtype=torch.bfloat16)
    B = torch.randn(K, N, dtype=torch.bfloat16)
    torch_gemm(A, B)

    # Benchmark torch runtime
    torch_gemm_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(torch_gemm_graph):
        torch_gemm(A, B)
    torch_time = do_bench(lambda: torch_gemm_graph.replay())

    # Setup shards and graphs
    from green_ctx import make_shard
    shards = [make_shard(32, i, add_remainder=(i == 3)) for i in range(4)]
    graphs = []
    events = [torch.cuda.Event(enable_timing=False) for _ in range(4)]
    streams = [torch.cuda.Stream() for _ in range(4)]

    # Create graphs
    for i, shard in enumerate(shards):
        with shard.with_torch_stream() as stream:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                for _ in range(50):
                    C = torch.matmul(A, B)
            graphs.append(graph)

    graph_1_time = do_bench(lambda: graphs[0].replay())
    graph_2_time = do_bench(lambda: graphs[1].replay())
    graph_3_time = do_bench(lambda: graphs[2].replay())
    graph_4_time = do_bench(lambda: graphs[3].replay())
    graph_time_avg = (graph_1_time + graph_2_time + graph_3_time +
                      graph_4_time) / 4

    # Benchmark running together
    together_time = do_bench(lambda: run_together(graphs, streams, events))

    rich.print(
        f"N: {N}, torch: {torch_time:.2f} ms, graph avg: {graph_time_avg:.2f} ms, four-multiplexed: {together_time:.2f} ms, 4 streams are {(torch_time - together_time) * 100 / torch_time:.2f} % faster than torch"
    )
    print(
        f"N: {N}, per op torch: {torch_time/200*1000:.2f} us, graph avg: {graph_time_avg/50*1000:.2f} us, four-multiplexed: {together_time/200*1000:.2f} us, 4 streams are {(torch_time - together_time) * 100 / torch_time:.2f} % faster than torch"
    )
