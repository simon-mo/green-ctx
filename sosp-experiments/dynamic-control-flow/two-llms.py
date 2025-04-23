from vllm import LLM, SamplingParams
from multiprocessing import Process, Barrier, Event
import torch


def worker(barrier_before_generate, barrier_after_generate, before_init_event,
           after_init_event, gpu_memory_utilization: float, rank: int):
    before_init_event.wait()

    llm = LLM(
        model="RedHatAI/Meta-Llama-3-8B-Instruct-FP8",
        load_format="dummy",
        max_model_len=8192,
        enable_prefix_caching=False,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    after_init_event.set()
    print(f"Worker {rank} initialized")

    llm.start_profile()
    barrier_before_generate.wait()

    out = llm.generate(
        [{
            "prompt_token_ids": [1] * 256
        }],
        SamplingParams(max_tokens=10, ignore_eos=True),
        use_tqdm=False,
    )
    print(out)


if __name__ == "__main__":
    barrier_before_generate = Barrier(2)
    barrier_after_generate = Barrier(2)

    events = [Event() for _ in range(3)]
    events[0].set()

    p1 = Process(target=worker,
                 args=(barrier_before_generate, barrier_after_generate,
                       events[0], events[1], 0.4, 0))
    p2 = Process(target=worker,
                 args=(barrier_before_generate, barrier_after_generate,
                       events[1], events[2], 0.8, 1))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
