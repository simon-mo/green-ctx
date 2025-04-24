import torch
from threading import Barrier, Event, Thread
import time
import numpy as np
from itertools import cycle
from green_ctx import get_sms_by_spec, GreenContext
from contextlib import nullcontext

torch.set_default_device("cuda")
torch.set_default_dtype(torch.bfloat16)

a = torch.randn((1024, 8192))
b = torch.randn((8192, 1024))


def worker_thread(operation, name, barrier, stop_event, gtx=None):
    wait_time_iter = cycle(np.random.poisson(lam=1, size=1000).tolist())

    barrier.wait()
    stream = torch.cuda.Stream()

    cnt = 0
    if gtx is not None:
        ctx = gtx.with_context()
    else:
        ctx = nullcontext()

    with ctx, torch.cuda.stream(stream):
        while not stop_event.is_set():
            wait_time = next(wait_time_iter)
            if wait_time > 0:
                time.sleep(wait_time / 1e6)  # sleep 1-5 us possion distributed
            operation()
            stream.synchronize()
            cnt += 1
        stream.synchronize()

    print(f"{name}: {cnt}")


def compute_operation():
    return a @ b


def memory_operation():
    return a.flatten() + a.flatten()


def main():
    # init and warm up
    a @ b
    torch.cuda.synchronize()

    barrier = Barrier(3)
    stop_event = Event()

    compute_ctx = get_sms_by_spec(4, 32, (0, 1, 2),
                                  get_remainder=False)  # 96 SMs
    memory_ctx = get_sms_by_spec(4, 32, (3, ), get_remainder=False)  # 32 SMs

    compute_thread = Thread(target=worker_thread,
                            args=(compute_operation, "compute_bound_thread",
                                  barrier, stop_event, compute_ctx))
    memory_thread = Thread(target=worker_thread,
                           args=(memory_operation, "memory_bound_thread",
                                 barrier, stop_event, memory_ctx))
    compute_thread.start()
    memory_thread.start()

    barrier.wait()

    time.sleep(0.1)
    stop_event.set()

    torch.cuda.synchronize()

    compute_thread.join()
    memory_thread.join()


if __name__ == "__main__":
    main()
