list:
    just --list

start-mps:
    sudo nvidia-cuda-mps-control -d

stop-mps:
    echo quit | sudo nvidia-cuda-mps-control
