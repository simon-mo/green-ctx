import torch
from threading import Barrier, Event, Thread
import time
import numpy as np
from itertools import cycle
from green_ctx import get_sms_by_spec, GreenContext, make_shard
from contextlib import nullcontext

from switch_lib import wrap_op_with_switch

torch.set_default_device("cuda")
torch.set_default_dtype(torch.bfloat16)

a = torch.randn((1024, 8192))
b = torch.randn((8192, 1024))


def worker_thread(operation, name, barrier, stop_event, gtx=None):
    wait_time_iter = cycle(np.random.poisson(lam=1, size=1000).tolist())

    stream = torch.cuda.Stream()

    cnt = 0
    if gtx is not None:
        gtx.with_context().__enter__()
    else:
        torch.cuda.stream(stream).__enter__()

    barrier.wait()

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
    with make_shard(132).with_context():
        a @ b
    torch.cuda.synchronize()

    barrier = Barrier(3)
    stop_event = Event()

    # compute_ctx = get_sms_by_spec(4, 32, (0, 1, 2),
    #                               get_remainder=False)  # 96 SMs
    # memory_ctx = get_sms_by_spec(4, 32, (3, ), get_remainder=False)  # 32 SMs

    compute_graph = wrap_op_with_switch(compute_operation, 8)
    memory_graph = wrap_op_with_switch(memory_operation, 8)

    compute_graph.debug_dump("./compute_graph.txt")
    memory_graph.debug_dump("./memory_graph.txt")

    def compute_op_graph():
        compute_graph.replay()

    def memory_op_graph():
        memory_graph.replay()

    torch.cuda.cudart().cudaProfilerStart()

    compute_thread = Thread(target=worker_thread,
                            args=(compute_op_graph, "compute_bound_thread",
                                  barrier, stop_event, None))
    memory_thread = Thread(target=worker_thread,
                           args=(memory_op_graph, "memory_bound_thread",
                                 barrier, stop_event, None))
    compute_thread.start()
    memory_thread.start()

    barrier.wait()

    time.sleep(0.1)
    stop_event.set()

    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStop()

    compute_thread.join()
    memory_thread.join()


if __name__ == "__main__":
    main()
