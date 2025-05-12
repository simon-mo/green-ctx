# if you want best tput for a given kernel, you might as well squeeze it and launch of 8x to fully max out things.

import torch

torch.set_default_device("cuda")
torch.set_default_dtype(torch.bfloat16)

M, K, N = 14336, 4096, 2048

A = torch.randn(M, K, dtype=torch.bfloat16)
B = torch.randn(K, N, dtype=torch.bfloat16)


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


def torch_gemm():
    for _ in range(10):
        C = torch.matmul(A, B)


print(f"torch runtime: {do_bench(torch_gemm)}")

# capture 2 graph, each with a 64 gtx
from green_ctx import make_shard

first_half = make_shard(64, 0)
second_half = make_shard(64, 1)

with first_half.with_context():
    first_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(first_graph):
        for _ in range(10):
            C = torch.matmul(A, B)

with second_half.with_context():
    second_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(second_graph):
        for _ in range(10):
            C = torch.matmul(A, B)

print(f"first graph: {do_bench(lambda: first_graph.replay())}")
print(f"second graph: {do_bench(lambda: second_graph.replay())}")

s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()


def run_together():
    with torch.cuda.stream(s1):
        first_graph.replay()
    with torch.cuda.stream(s2):
        second_graph.replay()


# TODO fix the timing here so we can demonstrate the point of tput optimial packing is the squeezing apporahc.

print(f"run together: {do_bench(run_together)}")
