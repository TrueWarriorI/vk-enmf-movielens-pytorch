"""Select one device per process and initialize DDP when launched by torchrun."""

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
import os

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class Runtime:
    device: torch.device
    rank: int
    world_size: int
    backend: str | None


@contextmanager
def distributed_runtime(device_type: str, threads: int):
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.set_num_threads(threads)
    if device_type == "cuda":
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device {local_rank} is unavailable. Use one process per visible GPU.")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    backend = None
    if world_size > 1:
        backend = "nccl" if device.type == "cuda" else "gloo"
        if backend == "nccl" and not dist.is_nccl_available():
            raise RuntimeError("Multi-GPU DDP needs a PyTorch build with NCCL. CPU DDP uses Gloo.")
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=15))

    try:
        yield Runtime(device, rank, world_size, backend)
    finally:
        if backend is not None:
            dist.destroy_process_group()
