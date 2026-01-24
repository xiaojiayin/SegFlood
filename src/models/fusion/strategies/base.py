from abc import ABC, abstractmethod
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseFusionStrategy(nn.Module, ABC):
    def __init__(self, optical_channels: List[int], sar_channels: List[int]):
        super().__init__()
        self.optical_channels = optical_channels
        self.sar_channels = sar_channels
        self.num_levels = len(optical_channels)
        self.output_channels = self._calculate_output_channels()

    @abstractmethod
    def _calculate_output_channels(self) -> List[int]:
        raise NotImplementedError

    @abstractmethod
    def forward(self, optical_features: List[torch.Tensor], sar_features: List[torch.Tensor]) -> List[torch.Tensor]:
        raise NotImplementedError


