import os
import pandas as pd
import math
from dataclasses import dataclass
from copy import deepcopy
import simpy


@dataclass
class KernelSpec:
    # Metadata
    name: str
    num_sms: int
    duration_ns: float

    # Scheduling data
    num_blocks: int
    ns_per_block: float

    # runtime variables
    is_running: bool = False
    remaining_blocks: int = 0
    num_sms_running: int = 0

    def __post_init__(self):
        self.remaining_blocks = self.num_blocks

    def __repr__(self):
        return f"K({self.num_blocks} bls, {self.duration_ns} ns)"

    @property
    def can_run(self):
        return self.num_sms_running <= self.num_sms and self.remaining_blocks > 0

    def start_block(self):
        self.is_running = True
        self.remaining_blocks -= 1
        self.num_sms_running += 1

    def finish_block(self):
        self.num_sms_running -= 1


@dataclass
class StreamSpec:
    name: str
    kernels: list[KernelSpec]


class Scheduler:

    def __init__(self, stream_specs: list[StreamSpec]):
        self.stream_specs = deepcopy(stream_specs)
        self.tidy()

        # self.print_scheduler_state()

    def print_scheduler_state(self):
        for stream in self.stream_specs:
            print(f"Stream {stream.name}:")
            for kernel in stream.kernels:
                print(
                    f"  Kernel {kernel.name}: {kernel.remaining_blocks} blocks remaining"
                )

    def get_next_block_ns(self):
        self.tidy()

        if self.is_completed:
            return None, None

        queue = [
            stream.kernels[0] for stream in self.stream_specs if stream.kernels
        ]
        # sort by is_running
        queue.sort(key=lambda x: x.is_running, reverse=True)

        for kernel in queue:
            if kernel.can_run:
                kernel.start_block()
                return (kernel.ns_per_block, kernel)
        else:
            # None of the kernels can run, the SM should try later
            return None, None

    @property
    def is_completed(self):
        return len(self.stream_specs) == 0

    def tidy(self):
        # remove empty kernels
        for stream in self.stream_specs:
            if stream.kernels[0].remaining_blocks == 0:
                stream.kernels.pop(0)

        # remove empty streams
        to_remove = []
        for stream in self.stream_specs:
            if len(stream.kernels) == 0:
                to_remove.append(stream)
        for stream in to_remove:
            self.stream_specs.remove(stream)


def run_scheduler(stream_specs: list[StreamSpec], total_sms: int):
    env = simpy.Environment()

    scheduler = Scheduler(stream_specs)

    def run_sm_process(i):
        while True:
            next_block_ns, kernel = scheduler.get_next_block_ns()
            if next_block_ns is None:
                break
            yield env.timeout(next_block_ns)

    for i in range(total_sms):
        env.process(run_sm_process(i))

    env.run()

    print(f"Total time: {env.now}")


def main():
    import rich

    data = []
    specs = {}
    for path in os.listdir("profile-dir"):
        name = path.split("=")[1].replace(".csv", "")

        if "sm_132" in name:
            continue

        if "prefill" in name:
            workload, num_sms = name.split("-")[:2]
            num_sms = int(num_sms.split("_")[1])
            ctx_size = 0
        elif "decode" in name:
            workload, ctx_size, num_sms = name.split("-")
            num_sms = int(num_sms.split("_")[1])
            ctx_size = int(ctx_size.split("_")[1])
        else:
            raise ValueError(f"Unknown workload: {name}")

        df = pd.read_csv(os.path.join("profile-dir", path))
        df["num_blocks"] = df["BlockXYZ"].map(lambda x: math.prod(
            map(int, (filter(lambda y: y != "",
                             x.strip().split(" "))))))
        total_time_ms = df['Kernel Dur (ns)'].sum() / 1e6

        kernels = []
        for _, row in df.iterrows():
            kernel_name = row["Kernel Name"]
            num_blocks = row["num_blocks"]
            duration_ns = row["Kernel Dur (ns)"]
            ns_per_block = duration_ns / num_blocks
            kernels.append(
                KernelSpec(name=kernel_name,
                           num_sms=num_sms,
                           num_blocks=num_blocks,
                           ns_per_block=ns_per_block,
                           duration_ns=duration_ns))
        stream_spec = StreamSpec(name=workload, kernels=kernels)

        specs[(workload, num_sms, ctx_size)] = stream_spec
        data.append((workload, num_sms, ctx_size, total_time_ms))

    df = pd.DataFrame(
        data, columns=["workload", "num_sms", "ctx_size", "total_time_ms"])
    df = df.sort_values(["workload", "num_sms", "ctx_size"])

    df.to_csv("simulator-data.csv", index=False)

    rich.print(df)

    # workload_1 = ("prefill_4096", 64)
    # workload_2 = ("decode_64", 64)
    # parallel_streams = [specs[workload_1], specs[workload_2]]
    # print(f"Runing {workload_1} & {workload_2}")

    # run_scheduler(parallel_streams, 128)


def test():
    # this should take total of 2ns for 2 sms
    specs = [
        StreamSpec(name="stream-1",
                   kernels=[
                       KernelSpec(name="a",
                                  num_sms=1,
                                  num_blocks=2,
                                  ns_per_block=1,
                                  duration_ns=1),
                       KernelSpec(name="b",
                                  num_sms=1,
                                  num_blocks=1,
                                  ns_per_block=1,
                                  duration_ns=1),
                   ]),
        StreamSpec(name="stream-2",
                   kernels=[
                       KernelSpec(name="c",
                                  num_sms=1,
                                  num_blocks=1,
                                  ns_per_block=1,
                                  duration_ns=1),
                   ])
    ]
    run_scheduler(specs, 2)


if __name__ == "__main__":
    main()
    # test()
