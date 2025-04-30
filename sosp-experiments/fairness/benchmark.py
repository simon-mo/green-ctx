import aiohttp
import asyncio
import numpy as np
from time import perf_counter_ns

model_name = "RedHatAI/Meta-Llama-3-8B-Instruct-FP8"
ports = [8100, 8101, 8102]


def make_random_tokens(num_tokens: int):
    return np.random.randint(0, 10000, num_tokens).astype(int).tolist()


async def send_completion_request(num_tokens: int,
                                  num_output_tokens: int,
                                  model_name: str,
                                  port: int = 8100):
    async with aiohttp.ClientSession() as session:
        start_time = perf_counter_ns()
        async with session.post(
                f"http://localhost:{port}/v1/completions",
                json={
                    "model": model_name,
                    "prompt": make_random_tokens(num_tokens),
                    "max_tokens": num_output_tokens,
                },
        ) as response:
            end_time = perf_counter_ns()
            return await response.json(), (end_time - start_time) / 1e9


async def main():
    for port in ports:
        results, duration = await send_completion_request(
            100, 10, model_name, port)
        print(results)
        print(f"Duration: {duration} s")


if __name__ == "__main__":
    asyncio.run(main())
