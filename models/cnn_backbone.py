"""Shared CNN backbone for all models. (See project history for full design notes.)"""

import torch
import torch.nn as nn


class CNNBackbone(nn.Module):
    def __init__(self, in_channels: int = 1, reduced_dim: int = 4, angle_scale: bool = True):
        super().__init__()
        self.reduced_dim = reduced_dim
        self.angle_scale = angle_scale

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.flatten_dim = 16 * 7 * 7

        self.reduce = nn.Sequential(
            nn.Linear(self.flatten_dim, 32),
            nn.ReLU(),
            nn.Linear(32, reduced_dim),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return x.flatten(start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.extract_features(x)
        reduced = self.reduce(feats)
        if self.angle_scale:
            reduced = torch.tanh(reduced) * torch.pi
        return reduced

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())