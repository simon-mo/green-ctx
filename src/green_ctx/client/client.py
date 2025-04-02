import logging
import uuid
from typing import Dict, List, Optional
import pickle
import torch

import grpc

from ..proto import gpu_service_pb2
from ..proto import gpu_service_pb2_grpc

from ..server.kv_cache_pool import KV_TENSOR_NAME

from green_ctx import GreenContext, get_sms_by_spec, init

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GPUClient:

    def __init__(self,
                 model_name: str,
                 host: str = "localhost",
                 port: int = 50051):
        self.channel = grpc.insecure_channel(f"{host}:{port}")
        self.stub = gpu_service_pb2_grpc.GPUServiceStub(self.channel)
        self.client_id = str(uuid.uuid4())
        self.model_name = model_name
        logger.info(f"Connected to GPU server at {host}:{port}")

        init()
        logger.info(f"Initialized GPUClient for '{model_name}'")

        self.anon_tensors = set()

    def health_check(self) -> Dict[str, any]:
        """Check server health status."""
        response = self.stub.HealthCheck(gpu_service_pb2.HealthCheckRequest())
        return {
            "status": response.status,
            "total_sms": response.total_sms,
            "available_sms": response.available_sms,
            "num_tensors": response.num_tensors
        }

    def request_exclusive_SMs(self, num_sms: int) -> tuple[str, GreenContext]:
        """Request exclusive access to a number of SMs."""
        request = gpu_service_pb2.RequestSMsRequest(num_sms=num_sms,
                                                    client_id=self.client_id)
        response = self.stub.RequestExclusiveSMs(request)
        return response.alloc_uuid, get_sms_by_spec(
            num_groups=response.num_groups,
            min_size=response.min_size,
            indices=response.indices,
            get_remainder=response.get_remainder)

    def free_SMs(self, alloc_uuid: str) -> bool:
        """Free previously allocated SMs."""
        request = gpu_service_pb2.FreeSMsRequest(alloc_uuid=alloc_uuid, )
        response = self.stub.FreeSMs(request)
        return response.success

    def alloc_tensor(self,
                     shape: List[int],
                     dtype: str,
                     name: Optional[str] = None,
                     get_if_exists: bool = False,
                     model_specific: bool = True) -> torch.Tensor:
        """Allocate a new tensor in GPU memory."""
        if name is None:
            name += f"tensor_{uuid.uuid4().hex[:8]}"
            self.anon_tensors.add(name)
        if model_specific:
            name = self.model_name + '.' + name
        request = gpu_service_pb2.AllocTensorRequest(
            shape=shape, dtype=dtype, name=name, get_if_exists=get_if_exists)
        response = self.stub.AllocTensor(request)
        deserializer, payload = pickle.loads(response.serialized_info)

        return deserializer(*payload)

    def free_tensor(self, name: str, model_specific: bool = True) -> bool:
        """Free a tensor from GPU memory."""
        if name in self.anon_tensors:
            self.anon_tensors.remove(name)
        if model_specific:
            name = self.model_name + '.' + name
        request = gpu_service_pb2.FreeTensorRequest(name=name)
        response = self.stub.FreeTensor(request)
        return response.success

    def get_tensor(self,
                   name: str,
                   model_specific: bool = True) -> torch.Tensor:
        """Get tensor information by name."""
        if model_specific:
            name = self.model_name + '.' + name
        request = gpu_service_pb2.GetTensorRequest(name=name)
        response = self.stub.GetTensor(request)
        deserializer, payload = pickle.loads(response.serialized_info)
        return deserializer(*payload)

    def exists_tensor(self, name: str, model_specific: bool = True) -> bool:
        """Check if a tensor exists."""
        if model_specific:
            name = self.model_name + '.' + name
        request = gpu_service_pb2.ExistTensorRequest(name=name)
        response = self.stub.ExistTensor(request)
        return response.exists

    def lock_tensor(self, name: str, model_specific: bool = True) -> bool:
        """Acquire exclusive access for a tensor."""
        if model_specific:
            name = self.model_name + '.' + name
        request = gpu_service_pb2.LockTensorRequest(name=name)
        response = self.stub.LockTensor(request)
        return response.success

    def unlock_tensor(self, name: str, model_specific: bool = True) -> bool:
        """Release exclusive access for a tensor."""
        if model_specific:
            name = self.model_name + '.' + name
        request = gpu_service_pb2.UnlockTensorRequest(name=name)
        response = self.stub.UnlockTensor(request)
        return response.success

    def set_kv_pool_memory_bytes(self, memory_bytes: int) -> bool:
        """Set the total memory bytes allocated for KV cache."""
        request = gpu_service_pb2.SetKVPoolMemoryBytesRequest(
            memory_bytes=memory_bytes)
        response = self.stub.SetKVPoolMemoryBytes(request)
        return response.success

    def get_kv_pool_memory_bytes(self) -> int:
        """Get the total memory bytes allocated for KV cache."""
        request = gpu_service_pb2.GetKVPoolMemoryBytesRequest()
        response = self.stub.GetKVPoolMemoryBytes(request)
        return response.memory_bytes

    def get_kv_tensor(self, shape: List[int], dtype: str) -> torch.Tensor:
        """Allocate a new tensor in GPU memory."""
        name = KV_TENSOR_NAME
        get_if_exists = True
        model_specific = False  # KV tensor is not model-specific
        self.lock_tensor(name, model_specific)
        if self.exists_tensor(name, model_specific):
            kv_tensor = self.get_tensor(name, model_specific)
        else:
            kv_tensor = self.alloc_tensor(shape, dtype, name, get_if_exists,
                                          model_specific)
        self.unlock_tensor(name, model_specific=False)
        return kv_tensor

    def free_kv_tensor(self) -> bool:
        name = KV_TENSOR_NAME
        model_specific = False  # KV tensor is not model-specific
        return self.free_tensor(name, model_specific)

    def kv_pool_init(self, total_num_blocks: int, kv_block_bytes: int) -> bool:
        """Initialize KV block pool."""
        request = gpu_service_pb2.KVPoolInitRequest(
            model_name=self.model_name,
            total_num_blocks=total_num_blocks,
            kv_block_bytes=kv_block_bytes)
        response = self.stub.KVPoolInit(request)
        return response.success

    def kv_pool_alloc(self, num_blocks: int) -> List[int]:
        """Allocate blocks from KV pool."""
        request = gpu_service_pb2.KVPoolAllocRequest(
            model_name=self.model_name, num_blocks=num_blocks)
        response = self.stub.KVPoolAlloc(request)
        return list(response.blocks)

    def kv_pool_free(self, blocks: List[int]) -> bool:
        """Return blocks back to KV pool."""
        request = gpu_service_pb2.KVPoolFreeRequest(model_name=self.model_name,
                                                    blocks=blocks)
        response = self.stub.KVPoolFree(request)
        return response.success

    def close(self):
        """Close the connection to the server."""
        for name in list(self.anon_tensors):
            self.free_tensor(name)
        self.channel.close()

    def __del__(self):
        self.close()
