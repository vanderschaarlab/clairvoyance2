"""
Useful reusable interfaces for PyTorch models.
"""
from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class CustomizableLossModelBase(ABC):

    loss_fn: nn.Module

    @abstractmethod
    def process_output(self, out: torch.Tensor, **kwargs) -> torch.Tensor:
        ...

    def compute_loss(
        self, out: torch.Tensor, target: torch.Tensor, **kwargs  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        assert self.loss_fn is not None
        return self.loss_fn(self.process_output(out), target)
