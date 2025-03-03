#!/bin/bash

# Exit on error
set -e

# Create the output directory if it doesn't exist
mkdir -p src/green_ctx/proto

# Generate Python code from proto definition
python -m grpc_tools.protoc \
    --proto_path=src/green_ctx/proto \
    --python_out=src/green_ctx/proto \
    --grpc_python_out=src/green_ctx/proto \
    src/green_ctx/proto/gpu_service.proto

# Make the generated files importable
touch src/green_ctx/proto/__init__.py

# In src/green_ctx/proto/gpu_service_pb2_grpc.py, change the import from:
# import gpu_service_pb2 as gpu__service__pb2
# import green_ctx.proto.gpu_service_pb2 as gpu__service__pb2
sed -i 's/import gpu_service_pb2 as gpu__service__pb2/import green_ctx.proto.gpu_service_pb2 as gpu__service__pb2/' src/green_ctx/proto/gpu_service_pb2_grpc.py


echo "Proto files generated successfully!"