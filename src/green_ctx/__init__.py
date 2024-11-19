from cuda import cuda, nvrtc
import torch
import numpy as np
import ctypes
from dataclasses import dataclass
from typing import Any, List, Tuple
from contextlib import contextmanager
import math
from functools import cached_property
import os

from .utils import CHECK_CUDA

device = None


def init():
    global device
    CHECK_CUDA(cuda.cuInit(0))
    device = CHECK_CUDA(cuda.cuDeviceGet(0))

    # Note(simon): we used to use cuCtxCreate to create a context, but we don't really need it
    # because we can use whatever is from PyTorch or implicit primary context.

    # Hack(simon):warmup cublas with a large workspace, this is needed to avoid
    # CUDA error: CUBLAS_STATUS_INTERNAL_ERROR when calling `cublasCreate(handle)`
    torch.cuda.init()
    a = torch.randn((1024, 1024), device="cuda")
    b = torch.randn((1024, 1024), device="cuda")
    torch.matmul(a, b)


@dataclass
class GreenContext:
    sm_count: int = 0
    raw_context: Any = None
    primary_context: Any = None

    @contextmanager
    def with_context(self):
        cuda.cuCtxPushCurrent(self.primary_context)
        yield
        cuda.cuCtxPopCurrent()

    def make_stream(self):
        stream = CHECK_CUDA(
            cuda.cuGreenCtxStreamCreate(
                self.primary_context, cuda.CUstream_flags.CU_STREAM_NON_BLOCKING, 0
            )
        )
        return stream

    @cached_property
    def sm_ids(self):
        from .kernels import launch_smid

        h_sm_ids = np.full(self.sm_count * 32, -1, dtype=np.int32)
        with self.with_context():
            launch_smid(h_sm_ids, self.sm_count, stream=None)
            torch.cuda.synchronize()
            h_sm_ids = np.unique(h_sm_ids)
        return sorted(h_sm_ids.tolist())[1:]  # remove the -1


def get_sms_in_range(start: int, end: int, get_remainder: bool=False) -> GreenContext:
    """gets SMs in range from start to end (non-inclusive)
    NOTE: very sus behavior, idk why it only has 15 x 8 = 120 SMs avail in result_resources
    """

    sm_resource = CHECK_CUDA(
        cuda.cuDeviceGetDevResource(
            device, cuda.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
        )
    )


    result_resources, _, remainder = CHECK_CUDA(
        cuda.cuDevSmResourceSplitByCount(
            16,
            sm_resource,
            cuda.CUdevSmResourceSplit_flags.CU_DEV_SM_RESOURCE_SPLIT_MAX_POTENTIAL_CLUSTER_SIZE,
            8,
        )
    )

    current_sm_id = 0
    group_indices_in_range = set()

    for group_idx, res in enumerate(result_resources):
        group_sm_count = res.sm.smCount
        group_start = current_sm_id
        group_end = current_sm_id + group_sm_count

        if start < group_end and end > group_start:
            group_indices_in_range.add(group_idx)

        current_sm_id += group_sm_count

    selected_resources = [result_resources[idx] for idx in sorted(group_indices_in_range)]

    if get_remainder:
        selected_resources.append(remainder)

    desc = CHECK_CUDA(cuda.cuDevResourceGenerateDesc(selected_resources, len(selected_resources)))
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

    gc = GreenContext(
        sm_count=green_sm_resource.sm.smCount, raw_context=green_ctx, primary_context=CHECK_CUDA(cuda.cuCtxFromGreenCtx(green_ctx))
    )

    # print(gc.sm_ids)

    return gc



def make_shard(sm_request: int) -> GreenContext:
    assert (
        sm_request >= 8 and sm_request % 8 == 0
    ), "On Compute Architecture 9.0+: The minimum count is 8 SMs and must be a multiple of 8."

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
    return GreenContext(
        sm_count=sm_count, raw_context=green_ctx, primary_context=primary_context
    )


def partition(sm_size_a: int, sm_size_b: int) -> Tuple[GreenContext, GreenContext]:
    assert sm_size_a % 8 == sm_size_b % 8 == 0, "must be a multiple of 8"

    sm_resource = CHECK_CUDA(
        cuda.cuDeviceGetDevResource(
            device, cuda.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
        )
    )
    # split into groups of 8, so we want total of 16 groups where 8x16=128
    result_resources, nb_groups, remaining = CHECK_CUDA(
        cuda.cuDevSmResourceSplitByCount(
            16,
            sm_resource,
            cuda.CUdevSmResourceSplit_flags.CU_DEV_SM_RESOURCE_SPLIT_MAX_POTENTIAL_CLUSTER_SIZE,
            8,
        )
    )

    group_1 = result_resources[0 : sm_size_a // 8]
    group_2 = result_resources[len(group_1) : len(group_1) + sm_size_b // 8]
    results = []
    for group in [group_1, group_2]:
        desc = CHECK_CUDA(cuda.cuDevResourceGenerateDesc(group, len(group)))
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
        sm_count = green_sm_resource.sm.smCount
        primary_context = CHECK_CUDA(cuda.cuCtxFromGreenCtx(green_ctx))
        results.append(
            GreenContext(
                sm_count=sm_count,
                raw_context=green_ctx,
                primary_context=primary_context,
            )
        )
    return tuple(results)
