# Minimal shim for BEiT-3 inference-only usage.
#
# The upstream microsoft/unilm utils.py pulls in training-only dependencies
# (torch._six, deepspeed, torch.distributed setup) that no longer work on
# modern PyTorch and are unnecessary for embedding inference. This file keeps
# only what modeling_finetune.py needs to construct BEiT3ForRetrieval:
# ClipLoss (only used to build the loss module in __init__; never called
# during only_infer=True forward passes) and the rank/world-size helpers.

import torch.distributed as dist
import torch.nn as nn


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


class ClipLoss(nn.Module):
    """Unused at inference time; kept only so BEiT3ForRetrieval.__init__ works."""

    def __init__(self, cache_labels: bool = False, rank: int = 0, world_size: int = 1) -> None:
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size

    def forward(self, *args, **kwargs):
        raise NotImplementedError("ClipLoss is a training-only component; not used for inference.")
