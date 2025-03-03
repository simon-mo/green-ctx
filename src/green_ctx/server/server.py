import logging
import pickle
from concurrent import futures
from dataclasses import dataclass
from typing import Dict, List

import grpc
import torch
from torch.multiprocessing.reductions import reduce_tensor

from ..proto import gpu_service_pb2
from ..proto import gpu_service_pb2_grpc

logger = logging.getLogger(__name__)

# Default server configuration
DEFAULT_PORT = 50051

DTYPE_MAP = {
    "bfoat16": torch.bfloat16,
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
        self.total_sms = 132
        self.available_sms = list(range(self.total_sms))
        self.allocated_sms: Dict[str, List[int]] = {}  # client_id -> sm_indices
        self.tensors: Dict[str, TensorMetadata] = {}

        torch.cuda.set_device(0)
        logger.info(f"Initializing GPU server with {self.total_sms} SMs")

    def HealthCheck(self, request, context):
        """Return server health status."""
        return gpu_service_pb2.HealthCheckResponse(
            status="healthy",
            total_sms=self.total_sms,
            available_sms=len(self.available_sms),
            num_tensors=len(self.tensors)
        )

    def RequestExclusiveSMs(self, request, context):
        """Allocate exclusive SMs to a client."""
        num_sms = request.num_sms
        client_id = request.client_id

        if num_sms > len(self.available_sms):
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED,
                        f"Requested {num_sms} SMs but only {len(self.available_sms)} available")
        if num_sms % 8 != 0:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT,
                        f"Requested {num_sms} SMs but must be a multiple of 8")

        allocated = self.available_sms[:num_sms]
        self.available_sms = self.available_sms[num_sms:]
        self.allocated_sms[client_id] = allocated

        logger.info(f"Allocated {num_sms} SMs to client {client_id}: {allocated}")
        return gpu_service_pb2.RequestSMsResponse(allocated_sms=allocated)

    def FreeSMs(self, request, context):
        """Free previously allocated SMs."""
        client_id = request.client_id
        allocated_sms = request.allocated_sms

        if client_id not in self.allocated_sms:
            context.abort(grpc.StatusCode.NOT_FOUND,
                        f"Client {client_id} has no allocated SMs")

        self.available_sms.extend(allocated_sms)
        del self.allocated_sms[client_id]

        logger.info(f"Freed {len(allocated_sms)} SMs from client {client_id}: {allocated_sms}")
        return gpu_service_pb2.FreeSMsResponse(success=True)

    def AllocTensor(self, request, context):
        """Allocate a new tensor in GPU memory."""
        name = request.name
        shape = tuple(map(int, request.shape))
        dtype = request.dtype

        if name in self.tensors:
            context.abort(grpc.StatusCode.ALREADY_EXISTS,
                        f"Tensor {name} already exists")

        value = torch.empty(shape, dtype= DTYPE_MAP[dtype])
        tensor = TensorMetadata(
            shape=tuple(shape),
            dtype=dtype,
            name=name,
            serialized_info=pickle.dumps(reduce_tensor(value)),
            value=value,
        )
        self.tensors[name] = tensor

        logger.info(f"Allocated tensor {name} ({shape}, {dtype})")
        return gpu_service_pb2.TensorInfo(
            name=name,
            shape=shape,
            dtype=dtype,
            serialized_info=tensor.serialized_info
        )

    def FreeTensor(self, request, context):
        """Free a tensor from GPU memory."""
        name = request.name

        if name not in self.tensors:
            context.abort(grpc.StatusCode.NOT_FOUND,
                        f"Tensor {name} does not exist")

        del self.tensors[name]
        logger.info(f"Freed tensor {name}")
        return gpu_service_pb2.FreeTensorResponse(success=True)

    def GetTensor(self, request, context):
        """Get tensor information by name."""
        name = request.name

        if name not in self.tensors:
            context.abort(grpc.StatusCode.NOT_FOUND,
                        f"Tensor {name} does not exist")

        tensor = self.tensors[name]
        logger.info(f"Got tensor {name} ({tensor.shape}, {tensor.dtype})")
        return gpu_service_pb2.TensorInfo(
            name=tensor.name,
            shape=list(tensor.shape),
            dtype=tensor.dtype,
            serialized_info=tensor.serialized_info
        )

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