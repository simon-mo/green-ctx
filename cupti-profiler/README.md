From scratch implementation of kernel profiler using CUPTI.


## Context
It was available in PyTorch with Kineto but need to build from source.
* https://pytorch.org/docs/main/profiler.html
* https://github.com/facebookresearch/HolisticTraceAnalysis/blob/main/examples/cupti_profiler_demo.py
* https://github.com/pytorch/kineto/blob/3c3fa42e3d02dc9dc7a50827fa3ba9915642ff8e/libkineto/src/CuptiRangeProfiler.cpp#L50
* https://github.com/pytorch/pytorch/pull/125685 (the PR is reverted)


