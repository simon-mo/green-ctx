from green_ctx import make_shard, init
from green_ctx.kernels import count_sm_ids
import torch

init()

green_ctx = make_shard(8)
print(green_ctx.sm_count)

print("before")
print(count_sm_ids(200))

green_ctx.set_context()
print("after")
print(count_sm_ids(200))