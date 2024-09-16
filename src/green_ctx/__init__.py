from cuda import cuda, nvrtc
import torch
import numpy as np
import ctypes
from dataclasses import dataclass
from typing import Any

from .utils import CHECK_CUDA

device = None


def init():
    if not torch.cuda.is_initialized():
        torch.cuda.init()

    global device
    device = CHECK_CUDA(cuda.cuDeviceGet(0))
    context = CHECK_CUDA(cuda.cuCtxCreate(0, device))
    CHECK_CUDA(cuda.cuCtxSetCurrent(context))


@dataclass
class GreenContext:
    sm_count: int = 0
    raw_context: Any = None
    primary_context: Any = None
    stream: Any = None

    def set_context(self):
        CHECK_CUDA(cuda.cuCtxSetCurrent(self.primary_context))


def make_shard(sm_request: int) -> GreenContext:
    # Get SM resource
    sm_resource = CHECK_CUDA(
        cuda.cuDeviceGetDevResource(
            device, cuda.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
        )
    )
    # print(f"SM Resource: {sm_resource.sm.smCount}")

    # Split the SM resource
    result_resources, nb_groups, remaining = CHECK_CUDA(
        cuda.cuDevSmResourceSplitByCount(
            1,
            sm_resource,
            cuda.CUdevSmResourceSplit_flags.CU_DEV_SM_RESOURCE_SPLIT_MAX_POTENTIAL_CLUSTER_SIZE,
            sm_request,
        )
    )
    # print(f"Number of groups created: {nb_groups}")

    # for i in range(nb_groups):
    #     print(f"Group {i}: {result_resources[i].sm.smCount} SMs")

    # print(f"Remaining SMs: {remaining.sm.smCount}")

    desc = CHECK_CUDA(cuda.cuDevResourceGenerateDesc([result_resources[0]], 1))
    green_ctx = CHECK_CUDA(
        cuda.cuGreenCtxCreate(
            desc, device, cuda.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM
        )
    )
    green_sm_resource = CHECK_CUDA(
        cuda.cuGreenCtxGetDevResource(
            green_ctx, cuda.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
        )
    )
    # print(f"Green SM Resource: {green_sm_resource.sm.smCount}")
    sm_count = green_sm_resource.sm.smCount

    primary_context = CHECK_CUDA(cuda.cuCtxFromGreenCtx(green_ctx))
    stream = CHECK_CUDA(
        cuda.cuGreenCtxStreamCreate(
            green_ctx, cuda.CUstream_flags.CU_STREAM_NON_BLOCKING, 0
        )
    )
    return GreenContext(sm_count=sm_count, raw_context=green_ctx, primary_context=primary_context, stream=stream)
