#!/usr/bin/env python3

import pm_sampling

def main():
    # Create a sampler on device index 0
    sampler = pm_sampling.PmSampler(device_index=0)

    # Query some metrics
    base_metrics = sampler.query_base_metrics()
    print("Number of base metrics:", len(base_metrics))
    print("First 5 base metrics:", base_metrics[:5])

    # Query the property (description and type) of a couple metrics
    metric_info = sampler.query_metric_properties(["gr__cycles_active.avg",
                                                   "gr__cycles_elapsed.max"])
    print("Metric info:")
    for metric, props in metric_info.items():
        print(f"  {metric} -> {props}")

    # Now enable sampling for a subset of metrics
    sampler.enable_sampling(
        metrics=["gr__cycles_active.avg",
                 "gr__cycles_elapsed.max",
                 "gpu__time_duration.sum"],
        sampling_interval=200000,           # 100us
        hardware_buffer_size=512*1024*1024, # 512MB
        max_samples=10
    )

    # Start sampling
    sampler.start_sampling()

    # -----------------------------------------------
    import torch
    N = 50000
    a = torch.arange(N, dtype=torch.int32, device='cuda')
    b = torch.arange(N, dtype=torch.int32, device='cuda')
    c = a + b  # vector add
    torch.cuda.synchronize()
    # -----------------------------------------------

    # Stop sampling
    sampler.stop_sampling()

    # Fetch the collected samples
    all_samples = sampler.get_samples()
    print(f"Collected {len(all_samples)} samples.")
    print("Printing up to first 5 samples:")
    for i, sample in enumerate(all_samples):
        print(f"Sample {i}:")
        print(f"  Start timestamp: {sample['startTimestamp']}")
        print(f"  End   timestamp: {sample['endTimestamp']}")
        print("  Metrics:")
        for m, val in sample["metrics"].items():
            print(f"    {m} = {val}")

    # Finally, disable sampling and cleanup
    sampler.disable_sampling()

if __name__ == "__main__":
    main()
