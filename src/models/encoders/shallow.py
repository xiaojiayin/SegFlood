import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(norm: str, num_channels: int) -> nn.Module:
    if norm == "gn":
        groups = min(16, num_channels)
        groups = max(1, groups)
        return nn.GroupNorm(groups, num_channels)
    return nn.BatchNorm2d(num_channels)


class ShallowDetailEncoder(nn.Module):
    """
    Lightweight spatial-detail encoder.

    Produces only 1/4 and 1/8 resolution features to complement single-scale backbones (e.g. ViT).
    Returns [feat4x, feat8x].
    """

    def __init__(self, in_channels: int, c2: int, c4: int, c8: int, norm: str = "bn"):
        super().__init__()
        # 2x feature channels
        c2 = int(c2)
        mid = max(16, c2)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c2, kernel_size=3, stride=2, padding=1, bias=False),
            _norm(norm, c2),
            nn.ReLU(inplace=True),
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(c2, c4, kernel_size=3, stride=2, padding=1, bias=False),
            _norm(norm, c4),
            nn.ReLU(inplace=True),
            nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1, bias=False),
            _norm(norm, c4),
            nn.ReLU(inplace=True),
        )
        self.stage8 = nn.Sequential(
            nn.Conv2d(c4, c8, kernel_size=3, stride=2, padding=1, bias=False),
            _norm(norm, c8),
            nn.ReLU(inplace=True),
            nn.Conv2d(c8, c8, kernel_size=3, stride=1, padding=1, bias=False),
            _norm(norm, c8),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor):
        f2 = self.stem(x)
        f4 = self.stage4(f2)
        f8 = self.stage8(f4)
        return [f2, f4, f8]


