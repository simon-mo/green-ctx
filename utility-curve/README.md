
The utility curve experiments. 03-25

Plot the utility curves of vLLM in general, and the kernels. We want to study the interference. For vLLM, we want to use the data to construct the case.

X axis: SM
Y axis: Tput

Kernel wise: compare mix and match [decode, gemm, layernorm]

1. Figure out their perforamnce for a fair when they are exclusively allocated.
2. MPS strategy as baseline, either 100% or 50/50.

Fundamentally, we are going to do speculative 3D scheduling: the SM allocation, the time/expected runtime, and the overlap dimension (i.e. density).

Pretty hard problems. Especially the CPU's scheduling can happen at T0 while the kernel actually starts at T1/2/3 which is unknown. By T2, some other kernels might be running already.