import torch
from green_ctx.utils import set_cublas_sm_count

torch.set_default_device("cuda")
torch.set_default_dtype(torch.bfloat16)

M, K = 14336, 4096
N = 64
A = torch.randn(M, K, dtype=torch.bfloat16)
B = torch.randn(K, N, dtype=torch.bfloat16)
torch.matmul(A, B)

from green_ctx import make_shard

shard = make_shard(32, 0)
with shard.with_torch_stream():
    torch.matmul(A, B)
