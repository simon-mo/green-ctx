import torch

torch.set_default_device("cuda")
torch.set_default_dtype(torch.bfloat16)

a = torch.empty((1024, 32768), device="cuda")
b = torch.empty((32768, 1024), device="cuda")
c = torch.empty((1024, 1024), device="cuda")

for k in [1024, 2048, 4096, 8192, 16384, 32768]:
    a_ = a[:, :k]
    b_ = b[:k, :]
    torch.mm(a_, b_, out=c)

# torch.mm(a, b, out=c)

# torch.cuda.synchronize()

# start = torch.cuda.Event(enable_timing=True)
# end = torch.cuda.Event(enable_timing=True)

# num_trials = 50
# start.record()
# for _ in range(num_trials):
#     torch.mm(a, b, out=c)
# end.record()

# torch.cuda.synchronize()

# print(start.elapsed_time(end) / num_trials, "ms")
