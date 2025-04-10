list:
    just --list

start-mps:
    sudo nvidia-cuda-mps-control -d

stop-mps:
    echo quit | sudo nvidia-cuda-mps-control

build-vllm:
    VLLM_USE_PRECOMPILED=1 uv pip install -e . --verbose --no-build-isolation