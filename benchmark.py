import torch
from green_ctx import make_shard, init, partition
import time
import threading

def initialize_data(matrix_size):
    return torch.randn(matrix_size, matrix_size).cuda(), torch.randn(matrix_size, matrix_size).cuda()

def matrix_multiply(context, a, b, start_event, end_event):
    with context.with_context():
        stream = context.make_stream()
        torch_stream = torch.cuda.ExternalStream(int(stream))
        with torch.cuda.stream(torch_stream):
            start_event.record()
            for _ in range(100):
                c = torch.matmul(a, b)
            end_event.record()
        torch.cuda.synchronize()

def verify_two_processes():
    print("---\nVerifying two processes assigned to exclusive SMs")

    green_ctx_1, green_ctx_2 = partition(8, 8)
    print(f"Created {green_ctx_1.sm_count} SMs for process 1: {green_ctx_1.sm_ids}")
    print(f"Created {green_ctx_2.sm_count} SMs for process 2: {green_ctx_2.sm_ids}")

    matrix_size = 4096
    a1, b1 = initialize_data(matrix_size)
    a2, b2 = initialize_data(matrix_size)

    e_start_1, e_end_1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e_start_2, e_end_2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    t1 = threading.Thread(target=matrix_multiply, args=(green_ctx_1, a1, b1, e_start_1, e_end_1))
    t2 = threading.Thread(target=matrix_multiply, args=(green_ctx_2, a2, b2, e_start_2, e_end_2))

    start_time = time.perf_counter()
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    end_time = time.perf_counter()

    time_1 = e_start_1.elapsed_time(e_end_1)
    time_2 = e_start_2.elapsed_time(e_end_2)
    total_time = (end_time - start_time) * 1000 

    print(f"Process 1 elapsed time: {time_1:.2f} ms")
    print(f"Process 2 elapsed time: {time_2:.2f} ms")
    print(f"Total experiment time: {total_time:.2f} ms")

def verify_single_thread():
    print("---\nVerifying single thread running workload on 8 SMs")

    green_ctx = make_shard(8)

    matrix_size = 4096
    a, b = initialize_data(matrix_size)

    e_start, e_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    start_time = time.perf_counter()
    matrix_multiply(green_ctx, a, b, e_start, e_end)
    end_time = time.perf_counter()

    total_time = (end_time - start_time) * 1000
    print(f"Total experiment time: {total_time:.2f} ms")

def main():
    verify_two_processes()
    verify_single_thread()

if __name__ == "__main__":
    init()
    main()
