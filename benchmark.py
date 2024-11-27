import torch
from green_ctx import make_shard, init, partition
import time
import multiprocessing


def warm_up_gpu():
    a, b = initialize_data(1024)
    with torch.no_grad():
        for _ in range(10):
            c = torch.matmul(a, b)
    torch.cuda.synchronize()

def initialize_data(matrix_size):
    return torch.randn(matrix_size, matrix_size).to("cuda"), torch.randn(matrix_size, matrix_size).to("cuda")

def worker_matrix_multiply(use_first_partition, result_dict, process_id, barrier):
    init()  
    green_ctx_1, green_ctx_2 = partition(8, 8)
    context = green_ctx_1 if use_first_partition else green_ctx_2
    a, b = initialize_data(2048)

    with context.with_context():
        stream = context.make_stream()
        torch_stream = torch.cuda.ExternalStream(int(stream))
        with torch.cuda.stream(torch_stream):
            barrier.wait()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(100):
                c = torch.matmul(a, b)
            end_event.record()
        torch.cuda.synchronize()

    elapsed_time = start_event.elapsed_time(end_event)
    result_dict[process_id] = elapsed_time

def verify_two_processes():
    print("---\nVerifying two processes assigned to exclusive SMs")

    manager = multiprocessing.Manager()
    result_dict = manager.dict()
    barrier = manager.Barrier(2)

    p1 = multiprocessing.Process(target=worker_matrix_multiply, args=(True, result_dict, 1, barrier))
    p2 = multiprocessing.Process(target=worker_matrix_multiply, args=(False, result_dict, 2, barrier))

    p1.start()
    p2.start()
    p1.join()
    p2.join()

    time_1 = result_dict.get(1, 0)
    time_2 = result_dict.get(2, 0)
    total_experiment_time = max(time_1, time_2)

    print(f"Process 1 computation time: {time_1:.2f} ms")
    print(f"Process 2 computation time: {time_2:.2f} ms")
    print(f"Total experiment time: {total_experiment_time:.2f} ms")


def verify_single_process():
    print("---\nVerifying single process running workload on 8 SMs")
    init() 
    green_ctx = make_shard(8)

    matrix_size = 2048
    a, b = initialize_data(matrix_size)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_time = time.perf_counter()

    with green_ctx.with_context():  
        start_event.record()
        for _ in range(100):
            c = torch.matmul(a, b)
        end_event.record()
    torch.cuda.synchronize()

    end_time = time.perf_counter()

    elapsed_time = start_event.elapsed_time(end_event)
    total_time = (end_time - start_time) * 1000
    print(f"Single thread computation time: {elapsed_time:.2f} ms")
    print(f"Total experiment time (excluding initialization): {total_time:.2f} ms")

def main():
    warm_up_gpu()
    verify_single_process()
    verify_two_processes()

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
