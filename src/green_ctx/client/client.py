import logging
import uuid
from typing import Dict, List, Optional
import pickle

import grpc

from ..proto import gpu_service_pb2
from ..proto import gpu_service_pb2_grpc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPUClient:
    def __init__(self, host: str = "localhost", port: int = 50051):
        self.channel = grpc.insecure_channel(f"{host}:{port}")
        self.stub = gpu_service_pb2_grpc.GPUServiceStub(self.channel)
        self.client_id = str(uuid.uuid4())
        logger.info(f"Connected to GPU server at {host}:{port}")

    def health_check(self) -> Dict[str, any]:
        """Check server health status."""
        response = self.stub.HealthCheck(gpu_service_pb2.HealthCheckRequest())
        return {
            "status": response.status,
            "total_sms": response.total_sms,
            "available_sms": response.available_sms,
            "num_tensors": response.num_tensors
        }

    def request_exclusive_SMs(self, num_sms: int) -> List[int]:
        """Request exclusive access to a number of SMs."""
        request = gpu_service_pb2.RequestSMsRequest(
            num_sms=num_sms,
            client_id=self.client_id
        )
        response = self.stub.RequestExclusiveSMs(request)
        return list(response.allocated_sms)

    def free_SMs(self, allocated_sms: List[int]) -> bool:
        """Free previously allocated SMs."""
        request = gpu_service_pb2.FreeSMsRequest(
            allocated_sms=allocated_sms,
            client_id=self.client_id
        )
        response = self.stub.FreeSMs(request)
        return response.success

    def alloc_tensor(self, shape: List[int], dtype: str, name: Optional[str] = None) -> Dict[str, any]:
        """Allocate a new tensor in GPU memory."""
        if name is None:
            name = f"tensor_{uuid.uuid4().hex[:8]}"

        request = gpu_service_pb2.AllocTensorRequest(
            shape=shape,
            dtype=dtype,
            name=name
        )
        response = self.stub.AllocTensor(request)
        deserializer, payload = pickle.loads(response.serialized_info)

        return deserializer(*payload)

    def free_tensor(self, name: str) -> bool:
        """Free a tensor from GPU memory."""
        request = gpu_service_pb2.FreeTensorRequest(name=name)
        response = self.stub.FreeTensor(request)
        return response.success

    def get_tensor(self, name: str) -> Dict[str, any]:
        """Get tensor information by name."""
        request = gpu_service_pb2.GetTensorRequest(name=name)
        response = self.stub.GetTensor(request)
        deserializer, payload = pickle.loads(response.serialized_info)
        return deserializer(*payload)

    def close(self):
        """Close the connection to the server."""
        self.channel.close()