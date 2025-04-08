from cuda import cuda, nvrtc
from nvmath.bindings.cublas import get_sm_count_target, set_sm_count_target
import torch
from contextlib import contextmanager
from typing import Optional


# Define CHECK_CUDA function
def _cudaGetErrorEnum(error):
    if isinstance(error, cuda.CUresult):
        err, name = cuda.cuGetErrorName(error)
        return name if err == cuda.CUresult.CUDA_SUCCESS else "<unknown>"
    elif isinstance(error, nvrtc.nvrtcResult):
        return nvrtc.nvrtcGetErrorString(error)[1]
    else:
        raise RuntimeError("Unknown error type: {}".format(error))


def print_current_context_id():
    ctx = CHECK_CUDA(cuda.cuCtxGetCurrent())
    device = CHECK_CUDA(cuda.cuDeviceGet(0))
    print(f"Current context ID: {ctx}, device ID: {device}")


@contextmanager
def set_cublas_sm_count(sm_count: Optional[int] = None):
    if sm_count is None:
        yield
        return

    handle = torch.cuda.current_blas_handle()
    old_sm_count = get_sm_count_target(handle)
    set_sm_count_target(handle, sm_count)
    try:
        yield
    finally:
        set_sm_count_target(handle, old_sm_count)


def CHECK_CUDA(result):
    if result[0].value:
        raise RuntimeError("CUDA error code={}({})".format(
            result[0].value, _cudaGetErrorEnum(result[0])))
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]
