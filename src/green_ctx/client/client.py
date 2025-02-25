import zmq
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import json
import logging
from dataclasses import dataclass
import numpy as np
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RequestMessage(BaseModel):
    command: str
    params: Dict[str, Any]

class ResponseMessage(BaseModel):
    success: bool
    data: Any
    error: str = ""

class GPUClient:
    def __init__(self, host: str = "localhost", port: int = 5555):
        self.host = host
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.client_id = hash(uuid.uuid4())  # Generate a unique client ID

        # Connect to the server
        self.socket.connect(f"tcp://{host}:{port}")
        logger.info(f"Connected to GPU server at {host}:{port}")

    def _send_request(self, command: str, **params) -> Any:
        """Send a request to the server and return the response data."""
        request = RequestMessage(command=command, params=params)
        self.socket.send_string(request.model_dump_json())

        response = ResponseMessage.model_validate_json(self.socket.recv_string())
        if not response.success:
            raise RuntimeError(f"Server error: {response.error}")

        return response.data

    def health_check(self) -> Dict[str, Any]:
        """Check server health status."""
        return self._send_request("health_check")

    def request_exclusive_SMs(self, num_sms: int) -> List[int]:
        """Request exclusive access to a number of SMs."""
        return self._send_request("request_exclusive_SMs", num_sms=num_sms, client_id=self.client_id)

    def free_SMs(self, allocated_sms: List[int]) -> bool:
        """Free previously allocated SMs."""
        return self._send_request("free_SMs", allocated_sms=allocated_sms, client_id=self.client_id)

    def alloc_tensor(self, shape: List[int], dtype: str, name: Optional[str] = None) -> Dict[str, Any]:
        """Allocate a new tensor in GPU memory."""
        if name is None:
            name = f"tensor_{uuid.uuid4().hex[:8]}"
        return self._send_request("alloc_tensor", shape=shape, dtype=dtype, name=name)

    def free_tensor(self, name: str) -> bool:
        """Free a tensor from GPU memory."""
        return self._send_request("free_tensor", name=name)

    def get_tensor(self, name: str) -> Dict[str, Any]:
        """Get tensor information by name."""
        return self._send_request("get_tensor", name=name)

    def close(self):
        """Close the connection to the server."""
        self.socket.close()
        self.context.term()