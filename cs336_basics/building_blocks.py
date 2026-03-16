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


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        weight_tensor = torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        self.weight = nn.Parameter(weight_tensor)
        nn.init.trunc_normal_(tensor=self.weight, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        output = self.weight[token_ids]
        return output


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        weight = torch.ones(size=[d_model], device=device, dtype=dtype)
        self.weight = nn.Parameter(weight)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(dtype=torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        x = x.to(dtype=in_dtype)
        x = self.weight * x
        return x


class FFN(nn.Module):
    def __init__(self, d_model: int, device=None, dtype=None):
        super().__init__()
        d_ff = d_model * 8 // 3
        d_ff = 64 - (d_ff % 64) + d_ff
        self.linear1 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.linear2 = Linear(in_features=d_ff, out_features=d_model, device=device, dtype=dtype)
        self.linear3 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.linear1(x)
        activation = result * torch.sigmoid(result)
        gate = self.linear3(x)
        gated_result = activation * gate
        output = self.linear(gated_result)
        return output
