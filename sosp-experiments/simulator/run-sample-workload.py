from vllm import LLM, SamplingParams
from time import perf_counter
import os
import time
import numpy as np
import rich

SMs = [32, 64, 96, 128]  # 1/8, 1/4, 1/2, 1
# SMs = [32, 64]  # 1/8, 1/4, 1/2, 1
SMs.append(132)  # without this the 128 trace will be empty, strange.

# PREFILL_TOKENS = [1024, 2048, 4096]
PREFILL_TOKENS = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
DECODE_BATCH_SIZE = [1]

llm = LLM(
    # model="RedHatAI/Meta-Llama-3-8B-Instruct-FP8",
    model="RedHatAI/QwQ-32B-FP8-dynamic",
    load_format="dummy",
    max_model_len=max(PREFILL_TOKENS) + 10,
    max_num_seqs=max(DECODE_BATCH_SIZE) * 2,
    max_num_batched_tokens=max(DECODE_BATCH_SIZE) * max(PREFILL_TOKENS),
    # enforce_eager=True,  # otherwise the graph replay uses full SMs.
)

# warmup
llm.generate([{
    "prompt_token_ids": [1] * (max(PREFILL_TOKENS))
}],
             SamplingParams(detokenize=False,
                            max_tokens=1,
                            extra_args={"num_sms": 32}),
             use_tqdm=False)


def make_input(prefill_tokens: int, batch_size: int):
    return [{
        "prompt_token_ids":
        np.random.randint(0, 10000, prefill_tokens).astype(int).tolist()
    } for _ in range(batch_size)]


for sm in SMs:
    for prefill_token in PREFILL_TOKENS:
        input_data = make_input(prefill_token, 1)

        prefill_start = perf_counter()
        out = llm.generate(
            input_data,
            SamplingParams(detokenize=False,
                           max_tokens=1,
                           extra_args={
                               "num_sms": sm,
                               "nvtx_tag": f"prefill_{prefill_token}-sm_{sm}"
                           }),
            use_tqdm=False)
        gpu_time_prefill = out[0].gpu_execution_time_ms
        prefill_end = perf_counter()

        rich.print(
            f"Processing {prefill_token} tokens with {sm} SMs: "
            f"prefill: [green]{(prefill_end - prefill_start) * 1000:.2f} ms[/green], "
            f"gpu_time_prefill: [blue]{gpu_time_prefill:.2f} ms[/blue]")

        if len(DECODE_BATCH_SIZE) == 0:
            continue

        decode_batch_input = make_input(prefill_token, max(DECODE_BATCH_SIZE))
        # prefill this batch first
        out = llm.generate(decode_batch_input,
                           SamplingParams(detokenize=False,
                                          max_tokens=1,
                                          extra_args={
                                              "num_sms": sm,
                                          }),
                           use_tqdm=False)
        # append the last token to the input
        for i in range(max(DECODE_BATCH_SIZE)):
            decode_batch_input[i]["prompt_token_ids"].append(
                out[i].outputs[0].token_ids[-1])

        for decode_batch_size in DECODE_BATCH_SIZE:
            decode_batch_input_local = decode_batch_input[:decode_batch_size]

            # decode the batch
            decode_start = perf_counter()
            out = llm.generate(
                decode_batch_input_local,
                SamplingParams(
                    detokenize=False,
                    max_tokens=1,
                    extra_args={
                        "num_sms":
                        sm,
                        "nvtx_tag":
                        f"decode_{decode_batch_size}-context_{prefill_token}-sm_{sm}"
                    }),
                use_tqdm=False)
            gpu_time_decode = out[0].gpu_execution_time_ms
            decode_end = perf_counter()

            rich.print(
                f"Processing decode batch of size {decode_batch_size} (context length {prefill_token}) with {sm} SMs: "
                f"decode: [green]{(decode_end - decode_start) * 1000:.2f} ms[/green], "
                f"gpu_time_decode: [blue]{gpu_time_decode:.2f} ms[/blue]")

time.sleep(10)
