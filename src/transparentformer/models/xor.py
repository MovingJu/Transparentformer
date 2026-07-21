"""Xor Classifier with basic MLP"""

import torch
import torch.nn as nn


class XorClassifierLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2, 1)
        self.out = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.out(x)
        return x


class XorClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2, 16)
        self.act = nn.Sigmoid()
        self.fc2 = nn.Linear(16, 1)
        self.out = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.out(x)
        return x
