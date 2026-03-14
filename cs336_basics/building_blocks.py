import math

import jaxtyping
import torch
import torch.nn as nn
from einops import einsum, rearrange


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        weight_tensor = torch.empty(size=(out_features, in_features), dtype=dtype, device=device)
        self.weight = nn.Parameter(weight_tensor)
        std = math.sqrt(2 / (in_features**2 + out_features**2))
        nn.init.trunc_normal_(tensor=self.weight, mean=0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = einsum(self.weight, x, "out_features in_features, ... in_features -> ... out_features")
        return x
