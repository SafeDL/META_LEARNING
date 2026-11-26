"""Process budgets and thread limits; model/data protocols are unchanged."""

import os

import torch
from threadpoolctl import threadpool_limits

COLLECTION_WORKERS = min(16, os.cpu_count() or 1)
METHOD_WORKERS = min(8, os.cpu_count() or 1)
TRAINING_WORKERS = min(4, os.cpu_count() or 1)
# Explicitly choose before constructing models; never auto-select a device on resume.
TRAINING_DEVICE = "cpu"
POOLED_DEVICE = "cuda"
_THREAD_LIMITER = None


def prepare_processes():
    """Set inherited limits before Windows spawn imports NumPy/PyTorch."""
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"


def initialize_worker():
    global _THREAD_LIMITER
    prepare_processes()
    torch.set_num_threads(1)
    _THREAD_LIMITER = threadpool_limits(limits=1)


def training_workers(device=TRAINING_DEVICE):
    if torch.device(device).type == "cpu":
        return TRAINING_WORKERS
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise ValueError("The registered training device is unavailable")
    return 1


def prediction_device(method):
    return POOLED_DEVICE if method == "pooled_history" else "cpu"


def prediction_workers(method, cpu_workers=METHOD_WORKERS):
    if torch.device(prediction_device(method)).type == "cpu":
        return cpu_workers
    if not torch.cuda.is_available():
        raise ValueError("The registered Pooled GP CUDA device is unavailable")
    return 1
