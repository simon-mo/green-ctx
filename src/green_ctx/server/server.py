import zmq
from typing import List, Dict, Any
from pydantic import BaseModel
import json
import logging
from dataclasses import dataclass
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default server configuration
DEFAULT_PORT = 5555

# Message types for client-server communication
class RequestMessage(BaseModel):
    command: str
    params: Dict[str, Any]

class ResponseMessage(BaseModel):
    success: bool
    data: Any
    error: str = ""

@dataclass
class Tensor:
    shape: tuple
    dtype: str
    name: str
    handle: int  # Simulated GPU memory handle

class GPUServer:
    def __init__(self, port: int = DEFAULT_PORT):
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)

        # Initialize resource tracking
        self.total_sms = 132
        self.available_sms = list(range(self.total_sms))
        self.allocated_sms: Dict[int, List[int]] = {}  # client_id -> sm_indices
        self.tensors: Dict[str, Tensor] = {}

        logger.info(f"Initializing GPU server with {self.total_sms} SMs")

    def start(self):
        """Start the server and listen for requests."""
        self.socket.bind(f"tcp://*:{self.port}")
        logger.info(f"Server listening on port {self.port}")

        while True:
            try:
                message = self.socket.recv_string()
                request = RequestMessage.model_validate_json(message)
                response = self.handle_request(request)
                self.socket.send_string(response.model_dump_json())
            except Exception as e:
                logger.error(f"Error handling request: {e}")
                error_response = ResponseMessage(success=False, data=None, error=str(e))
                self.socket.send_string(error_response.model_dump_json())

    def handle_request(self, request: RequestMessage) -> ResponseMessage:
        """Handle incoming client requests."""
        command = request.command
        params = request.params

        handlers = {
            "health_check": self.health_check,
            "request_exclusive_SMs": self.request_exclusive_SMs,
            "free_SMs": self.free_SMs,
            "alloc_tensor": self.alloc_tensor,
            "free_tensor": self.free_tensor,
            "get_tensor": self.get_tensor,
        }

        handler = handlers.get(command)
        if not handler:
            return ResponseMessage(success=False, data=None, error=f"Unknown command: {command}")

        try:
            result = handler(**params)
            return ResponseMessage(success=True, data=result)
        except Exception as e:
            return ResponseMessage(success=False, data=None, error=str(e))

    def health_check(self) -> Dict[str, Any]:
        """Return server health status."""
        return {
            "status": "healthy",
            "total_sms": self.total_sms,
            "available_sms": len(self.available_sms),
            "num_tensors": len(self.tensors)
        }

    def request_exclusive_SMs(self, num_sms: int, client_id: int) -> List[int]:
        """Allocate exclusive SMs to a client."""
        if num_sms > len(self.available_sms):
            raise ValueError(f"Requested {num_sms} SMs but only {len(self.available_sms)} available")

        allocated = self.available_sms[:num_sms]
        self.available_sms = self.available_sms[num_sms:]
        self.allocated_sms[client_id] = allocated
        return allocated

    def free_SMs(self, allocated_sms: List[int], client_id: int) -> bool:
        """Free previously allocated SMs."""
        if client_id not in self.allocated_sms:
            raise ValueError(f"Client {client_id} has no allocated SMs")

        self.available_sms.extend(allocated_sms)
        del self.allocated_sms[client_id]
        return True

    def alloc_tensor(self, shape: List[int], dtype: str, name: str) -> Dict[str, Any]:
        """Allocate a new tensor in GPU memory."""
        if name in self.tensors:
            raise ValueError(f"Tensor {name} already exists")

        handle = len(self.tensors) + 1  # Simulate a memory handle
        tensor = Tensor(
            shape=tuple(shape),
            dtype=dtype,
            name=name,
            handle=handle
        )
        self.tensors[name] = tensor
        return {
            "handle": handle,
            "name": name,
            "shape": shape,
            "dtype": dtype
        }

    def free_tensor(self, name: str) -> bool:
        """Free a tensor from GPU memory."""
        if name not in self.tensors:
            raise ValueError(f"Tensor {name} does not exist")

        del self.tensors[name]
        return True

    def get_tensor(self, name: str) -> Dict[str, Any]:
        """Get tensor information by name."""
        if name not in self.tensors:
            raise ValueError(f"Tensor {name} does not exist")

        tensor = self.tensors[name]
        return {
            "handle": tensor.handle,
            "name": tensor.name,
            "shape": tensor.shape,
            "dtype": tensor.dtype
        }