import logging
import uuid
from typing import Dict, List, Optional
import pickle
import torch

import grpc

from ..proto import gpu_service_pb2
from ..proto import gpu_service_pb2_grpc

from green_ctx import GreenContext, get_sms_by_spec, init

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPUClient:
    def __init__(self, host: str = "localhost", port: int = 50051):
        self.channel = grpc.insecure_channel(f"{host}:{port}")
        self.stub = gpu_service_pb2_grpc.GPUServiceStub(self.channel)
        self.client_id = str(uuid.uuid4())
        logger.info(f"Connected to GPU server at {host}:{port}")

        init()
        logger.info("Initialized GreenContext")

        self.anon_tensors = set()

        self.block_buffer: List[int] = []
        self.block_buffer_size = 16  # TODO: find optimal buffer size

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
        request = gpu_service_pb2.RequestSMsRequest(
            num_sms=num_sms,
            client_id=self.client_id
        )
        response = self.stub.RequestExclusiveSMs(request)
        return response.alloc_uuid, get_sms_by_spec(
            num_groups=response.num_groups,
            min_size=response.min_size,
            indices=response.indices,
            get_remainder=response.get_remainder
        )

    def free_SMs(self, alloc_uuid: str) -> bool:
        """Free previously allocated SMs."""
        request = gpu_service_pb2.FreeSMsRequest(
            alloc_uuid=alloc_uuid,
        )
        response = self.stub.FreeSMs(request)
        return response.success

    def alloc_tensor(self, shape: List[int], dtype: str, name: Optional[str] = None, get_if_exists: bool = False) -> torch.Tensor:
        """Allocate a new tensor in GPU memory."""
        if name is None:
            name = f"tensor_{uuid.uuid4().hex[:8]}"
            self.anon_tensors.add(name)
        request = gpu_service_pb2.AllocTensorRequest(
            shape=shape,
            dtype=dtype,
            name=name,
            get_if_exists=get_if_exists
        )
        response = self.stub.AllocTensor(request)
        deserializer, payload = pickle.loads(response.serialized_info)

        return deserializer(*payload)

    def free_tensor(self, name: str) -> bool:
        """Free a tensor from GPU memory."""
        request = gpu_service_pb2.FreeTensorRequest(name=name)
        response = self.stub.FreeTensor(request)
        if name in self.anon_tensors:
            self.anon_tensors.remove(name)
        return response.success

    def get_tensor(self, name: str) -> torch.Tensor:
        """Get tensor information by name."""
        request = gpu_service_pb2.GetTensorRequest(name=name)
        response = self.stub.GetTensor(request)
        deserializer, payload = pickle.loads(response.serialized_info)
        return deserializer(*payload)

    def exists_tensor(self, name: str) -> bool:
        """Check if a tensor exists."""
        request = gpu_service_pb2.ExistTensorRequest(name=name)
        response = self.stub.ExistTensor(request)
        return response.exists

    def lock_tensor(self, name: str) -> bool:
        """Acquire exclusive access for a tensor."""
        request = gpu_service_pb2.LockTensorRequest(name=name)
        response = self.stub.LockTensor(request)
        return response.success

    def unlock_tensor(self, name: str) -> bool:
        """Release exclusive access for a tensor."""
        request = gpu_service_pb2.UnlockTensorRequest(name=name)
        response = self.stub.UnlockTensor(request)
        return response.success

    def kv_pool_init(self, total_num_blocks: int) -> bool:
        """Initialize KV block pool."""
        request = gpu_service_pb2.KVPoolInitRequest(total_num_blocks=total_num_blocks)
        response = self.stub.KVPoolInit(request)
        return response.success

    def kv_pool_alloc(self, num_blocks: int) -> List[int]:
        """Allocate blocks from KV pool."""
        # cache blocks to reduce number of RPC calls for decode allocations
        if num_blocks == 1:
            if len(self.block_buffer) == 0:
                # server tries to allocate buffer_size blocks or 1 block
                request = gpu_service_pb2.KVPoolAllocRequest(num_blocks=1)
                response = self.stub.KVPoolAlloc(request)
                if len(response.blocks) == 0:
                    return []
                self.block_buffer = list(response.blocks)
            return [self.block_buffer.pop()]

        request = gpu_service_pb2.KVPoolAllocRequest(num_blocks=num_blocks)
        response = self.stub.KVPoolAlloc(request)
        return list(response.blocks)

    def kv_pool_free(self, blocks: List[int]) -> bool:
        """Return blocks back to KV pool."""
        request = gpu_service_pb2.KVPoolFreeRequest(blocks=blocks)
        response = self.stub.KVPoolFree(request)
        return response.success

    def close(self):
        """Close the connection to the server."""
        for name in list(self.anon_tensors):
            self.free_tensor(name)
        self.channel.close()

    def __del__(self):
        self.close()
