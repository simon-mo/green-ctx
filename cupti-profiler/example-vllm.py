import pm_sampling
from vllm import EngineArgs, LLMEngine, SamplingParams
import pandas as pd


def main():
    # Initialize the engine
    engine_args = EngineArgs(
        model="neuralmagic/Meta-Llama-3.1-8B-FP8",
        load_format="dummy",
        max_num_seqs=128,
    )
    engine = LLMEngine.from_engine_args(engine_args)

    # Initialize the sampler
    sampler = pm_sampling.PmSampler(device_index=0)

    metrics = [
        "gr__cycles_active.avg",
        "gr__cycles_elapsed.max",
        "sm__cycles_active.avg.pct_of_peak_sustained_elapsed",
        "tpc__warps_inactive_sm_active_realtime.avg.pct_of_peak_sustained_elapsed",
        "tpc__warps_inactive_sm_idle_realtime.avg.pct_of_peak_sustained_elapsed",
        "dramc__read_throughput.avg.pct_of_peak_sustained_elapsed",
        "dramc__write_throughput.avg.pct_of_peak_sustained_elapsed",
    ]
    metric_info = sampler.query_metric_properties(metrics)
    print("Metric info:")
    for metric, props in metric_info.items():
        print(f"  {metric} -> {props}")

    sampler.enable_sampling(
        metrics=metrics,
        sampling_interval=200000,  # about 100us
        hardware_buffer_size=512 * 1024 * 1024,  # 512MB
        max_samples=10000,
    )

    # Create requests
    sampling_params = SamplingParams(max_tokens=100)
    prompts = ["hi" * 1000] * 500
    all_samples = []

    # Add all requests to the engine
    for i, prompt in enumerate(prompts):
        engine.add_request(str(i), prompt, sampling_params)

    # Process requests and collect samples every N steps
    N_STEPS = 10  # Collect samples every 10 steps
    step_count = 0

    sampler.start_sampling()

    while engine.has_unfinished_requests():
        request_outputs = engine.step()
        step_count += 1

        # Get samples every N steps
        if step_count % N_STEPS == 0:
            samples = sampler.get_samples()
            all_samples.extend(samples)
        #     print(f"Step {step_count}: Collected {len(samples)} new samples")

        # # Print finished requests
        # for output in request_outputs:
        #     if output.finished:
        #         print(f"Finished request {output.request_id}")

    # Stop sampling and get final samples
    sampler.stop_sampling()
    final_samples = sampler.get_samples()
    all_samples.extend(final_samples)

    print(f"Total collected {len(all_samples)} samples.")
    print("Printing 10 samples, evenly spaced across the samples.")
    for i in range(0, len(all_samples), len(all_samples) // 10):
        sample = all_samples[i]
        print(f"Sample {i}:")
        print(f"  Start timestamp: {sample['startTimestamp']}")
        print(f"  End   timestamp: {sample['endTimestamp']}")
        print("  Metrics:")
        for m, val in sample["metrics"].items():
            print(f"    {m} = {val}")

    # Export the data in tidy format
    data = []
    for i, sample in enumerate(all_samples):
        for m, val in sample["metrics"].items():
            data.append({
                "sample": i,
                "metric": m,
                "value": val,
            })
    df = pd.DataFrame(data)
    df.to_csv("vllm-pm-samples.csv", index=False)

    # Finally, disable sampling and cleanup
    sampler.disable_sampling()


if __name__ == "__main__":
    main()
