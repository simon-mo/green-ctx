import torch

def cuda_timing_decorator(func):
    """Wrap the function with timing code, return the elapsed time in milliseconds.

    Note: It performs cuda sychronization! However, it does use events for timing.

    """
    def wrapper(*args, **kwargs):
        # Create start and stop events for timing
        start_event = torch.cuda.Event(enable_timing=True)
        stop_event = torch.cuda.Event(enable_timing=True)

        # Record the start event
        start_event.record()

        # Execute the function (e.g., a kernel or operation)
        func(*args, **kwargs)

        # Record the stop event and synchronize
        stop_event.record()
        torch.cuda.synchronize()

        # Measure elapsed time
        milliseconds = start_event.elapsed_time(stop_event)

        return milliseconds

    return wrapper
