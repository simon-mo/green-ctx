from vllm import LLM, SamplingParams
from time import perf_counter

SMs = [16, 32, 64, 128]  # 1/8, 1/4, 1/2, 1

PREFILL_TOKENS = [128, 512, 2048, 8192]
# PREFILL_TOKENS = [8192]

llm = LLM(
    model="RedHatAI/Meta-Llama-3-8B-Instruct-FP8",
    load_format="dummy",
    max_model_len=max(PREFILL_TOKENS),
    enable_prefix_caching=False,
)

# warmup
# llm.generate([{
#     "prompt_token_ids": [1] * (max(PREFILL_TOKENS) - 1)
# }],
#              sampling_params,
#              use_tqdm=False)


def make_input(prefill_tokens: int):
    return [{"prompt_token_ids": [1] * (prefill_tokens - 1)}]


data = []

for _ in range(3):
    for prefill_tokens in PREFILL_TOKENS:
        for sm in SMs:
            start = perf_counter()
            out = llm.generate(make_input(prefill_tokens),
                               SamplingParams(detokenize=False,
                                              max_tokens=1,
                                              extra_args={"num_sms": sm}),
                               use_tqdm=False)
            end = perf_counter()

            data.append({
                "prefill_tokens": prefill_tokens,
                "num_sms": sm,
                "gpu_execution_time_ms": out[0].gpu_execution_time_ms,
                "wall_time_ms": (end - start) * 1000,
            })

            print(
                f"Prefilling {prefill_tokens} tokens with {sm} SMs: {(end - start) * 1000:.2f} ms, {out[0].gpu_execution_time_ms=}"
            )

import pandas as pd

df = pd.DataFrame(data)
df.to_csv("prefill-data.csv", index=False)
