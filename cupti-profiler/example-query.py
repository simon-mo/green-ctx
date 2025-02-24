#!/usr/bin/env python3

import pm_sampling


def main():
    # Create a sampler on device index 0
    sampler = pm_sampling.PmSampler(device_index=0)

    # Query some metrics
    base_metrics = sampler.query_base_metrics()
    print("Number of base metrics:", len(base_metrics))

    suffix = {
        "Counter": ".avg",
        "Ratio": ".pct",
        "Throughput": ".avg.pct_of_peak_sustained_elapsed",
    }
    full_metrics = []
    for metric in base_metrics:
        metric_type = sampler.get_metric_type(metric)
        full_name = f"{metric}{suffix[metric_type]}"
        full_metrics.append(full_name)

    props = sampler.query_metric_properties(full_metrics)
    data = []
    for i in range(len(full_metrics)):
        data.append({
            "name": base_metrics[i],
            "type": metric_type,
            "description": props[full_metrics[i]]["description"],
        })
    import pandas as pd
    pd.DataFrame(data).to_csv("metrics.csv", index=False)


if __name__ == "__main__":
    main()
