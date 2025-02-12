from green_ctx import init, partition
from green_ctx.kernels import count_sm_ids
import torch
import os
import torch.multiprocessing as mp

PERC_TO_TEST = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

def run_test(percentage: float, shared_dict):
    # Set thread percentage for this process
    os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(int(percentage * 100))

    init()
    num_sms = len(count_sm_ids(500))
    shared_dict[percentage] = num_sms

def main():
    mp.set_start_method('spawn')

    # Create shared memory manager
    manager = mp.Manager()
    shared_dict = manager.dict()

    # Create and start processes for each percentage
    processes = []
    for perc in PERC_TO_TEST:
        p = mp.Process(target=run_test, args=(perc, shared_dict))
        processes.append(p)
        p.start()

    # Wait for all processes to complete
    for p in processes:
        p.join()

    # Print final summary
    print("\nFinal Results Summary:")
    print("-" * 50)
    for perc in sorted(shared_dict.keys()):
        print(f"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE {perc*100:3.0f}%: {shared_dict[perc]} SMs")

if __name__ == "__main__":
    main()
