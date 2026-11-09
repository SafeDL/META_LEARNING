"""Sample process memory while timing construction and online inference."""

import functools
import threading
import time

import psutil
import torch


def synchronized_time(device="cpu"):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


class PeakMemory:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)

    def __enter__(self):
        self.process = psutil.Process()
        self.baseline = self.process.memory_info().rss
        self.peak = self.baseline
        self.stop = threading.Event()
        self.started = synchronized_time(self.device)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def _sample(self):
        while not self.stop.wait(0.01):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __exit__(self, *exception):
        elapsed = synchronized_time(self.device) - self.started
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.stop.set()
        self.thread.join()
        self.result = {
            "wall_s": elapsed,
            "sampled_peak_cpu_bytes": self.peak,
            "sampled_cpu_increase_bytes": max(0, self.peak - self.baseline),
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(self.device)
            if self.device.type == "cuda"
            else 0,
            "device": str(self.device),
            "cpu_sampling_interval_s": 0.01,
        }


def measure_resources(function):
    @functools.wraps(function)
    def measured(*args, **kwargs):
        with PeakMemory(device=kwargs.get("device", "cpu")) as memory:
            result = function(*args, **kwargs)
        result["resources"] = memory.result
        return result

    return measured
