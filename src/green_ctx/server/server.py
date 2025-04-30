import logging
import pickle
from concurrent import futures
from dataclasses import dataclass
from typing import Dict, List, Optional
from threading import Lock

import grpc
import torch
from torch.multiprocessing.reductions import reduce_tensor
import uuid

from .kv_cache_pool import KVCachePool

from ..proto import gpu_service_pb2
from ..proto import gpu_service_pb2_grpc

logger = logging.getLogger(__name__)

# Default server configuration
DEFAULT_PORT = 50051

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


@dataclass
class TensorMetadata:
    shape: tuple
    dtype: str
    name: str
    serialized_info: bytes  # Serialized GPU memory information
    value: torch.Tensor


class GPUServicer(gpu_service_pb2_grpc.GPUServiceServicer):

    def __init__(self):
        # Initialize resource tracking
        self.total_sms = 120
        self.total_groups = self.total_sms // 8
        self.available_sms_group_indices = list(range(self.total_groups))
        self.allocated_sms_group_indices: Dict[str, List[int]] = {
        }  # alloc_uuid -> sm_group_indices
        self.tensors: Dict[str, TensorMetadata] = {}
        self.tensor_locks: Dict[str, Lock] = {}
        self.global_lock = Lock()

        self.kv_pool: Optional[KVCachePool] = None
        self.kv_pool_lock = Lock()
        self.request_rate = 0

        logger.info(f"Initializing GPU server with {self.total_sms} SMs")

    @property
    def available_sms(self):
        return len(self.available_sms_group_indices) * 8

    def HealthCheck(self, request, context):
        """Return server health status."""
        return gpu_service_pb2.HealthCheckResponse(
            status="healthy",
            total_sms=self.total_sms,
            available_sms=self.available_sms,
            num_tensors=len(self.tensors),
            request_rate=self.request_rate)

    def RequestExclusiveSMs(self, request, context):
        """Allocate exclusive SMs to a client."""
        num_sms = request.num_sms
        client_id = request.client_id

        if num_sms > self.available_sms:
            context.abort(
                grpc.StatusCode.RESOURCE_EXHAUSTED,
                f"Requested {num_sms} SMs but only {self.available_sms} available"
            )
        if num_sms % 8 != 0:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Requested {num_sms} SMs but must be a multiple of 8")

        alloc_uuid = str(uuid.uuid4())
        num_groups = num_sms // 8

        allocated_group_indices = self.available_sms_group_indices[:num_groups]
        self.available_sms_group_indices = self.available_sms_group_indices[
            num_groups:]
        self.allocated_sms_group_indices[alloc_uuid] = allocated_group_indices

        logger.info(
            f"Allocated {num_sms} SMs to client {client_id}: {allocated_group_indices}"
        )
        return gpu_service_pb2.RequestSMsResponse(
            alloc_uuid=alloc_uuid,
            num_groups=self.total_groups,
            min_size=8,
            indices=allocated_group_indices,
            get_remainder=False)

    def FreeSMs(self, request, context):
        """Free previously allocated SMs."""
        alloc_uuid = request.alloc_uuid

        if alloc_uuid not in self.allocated_sms_group_indices:
            context.abort(grpc.StatusCode.NOT_FOUND,
                          f"Alloc UUID {alloc_uuid} has no allocated SMs")

        removed_group_indices = self.allocated_sms_group_indices[alloc_uuid]
        self.available_sms_group_indices.extend(removed_group_indices)
        del self.allocated_sms_group_indices[alloc_uuid]

        logger.info(
            f"Freed {len(removed_group_indices)*8} SMs from alloc UUID {alloc_uuid}"
        )
        return gpu_service_pb2.FreeSMsResponse(success=True)

    def AllocTensor(self, request, context):
        """Allocate a new tensor in GPU memory."""
        name = request.name
        shape = tuple(map(int, request.shape))
        dtype = request.dtype

        if name in self.tensors:
            if request.get_if_exists:
                return gpu_service_pb2.TensorInfo(
                    name=name,
                    shape=shape,
                    dtype=dtype,
                    serialized_info=self.tensors[name].serialized_info)
            else:
                context.abort(grpc.StatusCode.ALREADY_EXISTS,
                              f"Tensor {name} already exists")

        value = torch.empty(shape, dtype=DTYPE_MAP[dtype], device="cuda")

        reduced = pickle.dumps(reduce_tensor(value))
        assert len(
            reduced
        ) < 2 * 1024 * 1024, f"Tensor is too large to serialize ({len(reduced)} bytes)"

        tensor = TensorMetadata(
            shape=tuple(shape),
            dtype=dtype,
            name=name,
            serialized_info=reduced,
            value=value,
        )
        self.tensors[name] = tensor

        logger.info(f"Allocated tensor {name} ({shape}, {dtype})")
        return gpu_service_pb2.TensorInfo(
            name=name,
            shape=shape,
            dtype=dtype,
            serialized_info=tensor.serialized_info)

    def FreeTensor(self, request, context):
        """Free a tensor from GPU memory."""
        name = request.name

        if name not in self.tensors:
            context.abort(grpc.StatusCode.NOT_FOUND,
                          f"Tensor {name} does not exist")

        del self.tensors[name]
        if name in self.tensor_locks:
            if self.tensor_locks[name].locked():
                context.abort(grpc.StatusCode.LOCKED,
                              f"Tensor {name} is locked, cannot free")
            del self.tensor_locks[name]
        logger.debug(f"Freed tensor {name}")
        return gpu_service_pb2.FreeTensorResponse(success=True)

    def GetTensor(self, request, context):
        """Get tensor information by name."""
        name = request.name

        if name not in self.tensors:
            context.abort(grpc.StatusCode.NOT_FOUND,
                          f"Tensor {name} does not exist")

        tensor = self.tensors[name]
        logger.debug(f"Got tensor {name} ({tensor.shape}, {tensor.dtype})")
        return gpu_service_pb2.TensorInfo(
            name=tensor.name,
            shape=list(tensor.shape),
            dtype=tensor.dtype,
            serialized_info=tensor.serialized_info)

    def ExistTensor(self, request, context):
        """Check if a tensor exists."""
        name = request.name
        if name in self.tensors:
            return gpu_service_pb2.ExistTensorResponse(exists=True)
        else:
            return gpu_service_pb2.ExistTensorResponse(exists=False)

    def LockTensor(self, request, context):
        """Acquire exclusive access to a tensor by name.
        Acquiring client will block until lock is free."""
        name = request.name

        with self.global_lock:
            if name not in self.tensor_locks:
                self.tensor_locks[name] = Lock()

        self.tensor_locks[name].acquire()
        return gpu_service_pb2.LockTensorResponse(success=True)

    def UnlockTensor(self, request, context):
        """Release exclusive access to a tensor by name.
        Caller must be holding the lock."""
        name = request.name

        if name not in self.tensor_locks:
            context.abort(grpc.StatusCode.NOT_FOUND,
                          f"Tensor {name} does not exist")

        if not self.tensor_locks[name].locked():
            context.abort(grpc.StatusCode.LOCK_NOT_HELD,
                          f"Tensor {name} is not locked")

        self.tensor_locks[name].release()  # is this fine?
        return gpu_service_pb2.UnlockTensorResponse(success=True)

    def KVPoolInit(self, request, context):
        """Initialize KV block pool."""
        with self.kv_pool_lock:
            logger.info(
                f"Initializing KV pool for model {request.model_name} "
                f"with {request.total_num_blocks} blocks of "
                f"size {request.kv_block_bytes} bytes"
            )
            if self.kv_pool is None:
                self.kv_pool = KVCachePool(request.total_num_blocks *
                                           request.kv_block_bytes)
            self.kv_pool.register_model(
                model_name=request.model_name,
                kv_block_bytes=request.kv_block_bytes)

            return gpu_service_pb2.KVPoolInitResponse(success=True)

    def KVPoolAlloc(self, request, context):
        """Allocate KV blocks from the pool for client.
        Returns list of block ids, or an empty list if not enough blocks."""
        model_name = request.model_name
        num_blocks = request.num_blocks
        if num_blocks <= 0:
            context.abort(grpc.StatusCode.INVALID_ALLOC_REQUEST,
                          f"Cannot allocate {num_blocks} blocks.")

        with self.kv_pool_lock:
            blocks = self.kv_pool.alloc(model_name, num_blocks)
            if blocks is None:  # not enough blocks available
                logger.warning(
                    f"KV pool allocation failed for {num_blocks} blocks")
                return gpu_service_pb2.KVPoolAllocResponse(blocks=[])
            # logger.debug(f"KV pool allocated blocks: {blocks}")
            return gpu_service_pb2.KVPoolAllocResponse(blocks=blocks)

    def KVPoolFree(self, request, context):
        """Free the given block ids and add them back to the pool."""
        with self.kv_pool_lock:
            if self.kv_pool is None:
                context.abort(grpc.StatusCode.NOT_FOUND,
                              "KV pool not initialized")
            self.kv_pool.free(request.model_name, request.blocks)

        return gpu_service_pb2.KVPoolFreeResponse(success=True)

    def ReportRequestRate(self, request, context):
        """Report the request rate for the server."""
        self.request_rate = request.request_rate
        return gpu_service_pb2.RequestRateResponse(success=True)


class GPUServer:

    def __init__(self, port: int = DEFAULT_PORT):
        self.port = port
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        gpu_service_pb2_grpc.add_GPUServiceServicer_to_server(
            GPUServicer(), self.server)

    def start(self):
        """Start the server and listen for requests."""
        self.server.add_insecure_port(f'[::]:{self.port}')
        self.server.start()
        logger.info(f"Server listening on port {self.port}")
        self.server.wait_for_termination()
