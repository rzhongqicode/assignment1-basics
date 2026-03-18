import math

from jaxtyping import Float, Bool
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
    def __init__(self, d_model: int, d_ff:int = None, device=None, dtype=None):
        super().__init__()
        if not d_ff:
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
        output = self.linear2(gated_result)
        return output
    
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta:float, d_k:int, max_seq_len:int, device=None):
        super().__init__()
        k = torch.arange(start=1,end=d_k//2 + 1, step=1,device=device)
        power = (2*k -2)/d_k
        freq_array = torch.pow(theta, power)
        freq_array.reciprocal_()
        position_array = torch.arange(start=0, end=max_seq_len, step=1,device=device,dtype=torch.float32)
        angle = torch.outer(position_array, freq_array)# [max_seq_len, d_k/2]
        angle = torch.repeat_interleave(angle,2,dim=-1)# [max_seq_len, d_k]
        sin_tensor = torch.sin(angle)
        cos_tensor = torch.cos(angle)

        # put them to buffer
        self.register_buffer(name="sin_tensor", tensor=sin_tensor, persistent=False)
        self.register_buffer(name="cos_tensor", tensor=cos_tensor, persistent=False)


    def forward(self, x:torch.Tensor, token_positions:torch.Tensor)->torch.Tensor:
        # x:"... sequence_len, d_k", token_positions:"..., sequence_len"
        sin_term = self.sin_tensor[token_positions] #[... sequence_len, d_k]
        cos_term = self.cos_tensor[token_positions] #[... sequence_len, d_k]
        x_transformed = rearrange(x, "... seq_len (half two) -> ... seq_len half two", two=2)
        first_part = x_transformed[..., 0]
        second_part = x_transformed[..., 1]
        x_rotated = torch.stack([-second_part,first_part],dim=-1)
        x_final = rearrange(x_rotated, "... seq_len half two -> ... seq_len (half two)", two=2)#[... sequence_len, d_k]
        output = x * cos_term + x_final * sin_term
        return output

def softmax(in_features:torch.Tensor, dim:int)->torch.Tensor:
    max_values, _ = torch.max(input=in_features, dim=dim, keepdim=True)
    in_features = in_features - max_values
    in_features = torch.exp(in_features)
    sum_values = torch.sum(input=in_features, dim=dim, keepdim=True)
    output = in_features / sum_values
    return output

def scaled_dot_product_attention(
    Q: Float[torch.Tensor, " ... query_len d_k"],
    K: Float[torch.Tensor, " ... kv_len d_k"],
    V: Float[torch.Tensor, " ... kv_len d_v"],
    mask: Bool[torch.Tensor, "query_len kv_len"] | None = None,
) -> Float[torch.Tensor, " ... query d_v"]:
    
    score = einsum(Q,K," ... query_len d_k, ... kv_len d_k -> ... query_len kv_len")

    d_k = Q.size(-1)
    scale_value = 1 / math.sqrt(d_k)
    score = score * scale_value

    if mask is not None: 
        score = score.masked_fill(~mask, float('-inf'))

    score = softmax(in_features=score, dim=-1)
    output = einsum(score, V, "... query_len kv_len, ... kv_len d_v -> ... query_len d_v")
    return output

class Multihead_self_attention(nn.Module):
    def __init__(self, d_model:int, num_heads:int, rope_module=None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.rope = rope_module
        # d_k = d_v = d_model // num_heads
        self.W_QKV = Linear(in_features=d_model, out_features=3*d_model, device=device, dtype=dtype)
        self.Wo = Linear(in_features=d_model, out_features=d_model, device=device, dtype=dtype)

    # in this case, Q, K, V have the same 'seq_len'
    def forward(self, input:torch.Tensor, token_positions=None)->torch.Tensor:
        QKV = self.W_QKV(input)
        Q, K, V = torch.chunk(input=QKV, chunks=3, dim=-1)
        seq_len = input.shape[-2]
        Q = rearrange(Q, "... seq_len (num_heads qk_dim) -> ... num_heads seq_len qk_dim", num_heads = self.num_heads)
        K = rearrange(K, "... seq_len (num_heads qk_dim) -> ... num_heads seq_len qk_dim", num_heads = self.num_heads)
        if self.rope is not None and token_positions is not None:
            Q = self.rope(Q,token_positions)
            K = self.rope(K,token_positions)

        V = rearrange(V, "... seq_len (num_heads v_dim) -> ... num_heads seq_len v_dim", num_heads = self.num_heads)
        causal_msak = torch.ones(size=(seq_len,seq_len), dtype=torch.bool, device=input.device)
        causal_msak = torch.tril(causal_msak)
        output = scaled_dot_product_attention(Q,K,V,mask=causal_msak)
        # rearrange function is not in-place operation!
        output = rearrange(output, "... num_heads seq_len v_dim -> ... seq_len (num_heads v_dim)", num_heads=self.num_heads)
        output = self.Wo(output)
        return output
    
class Tranformer_block(nn.Module):
    def __init__(self, d_model:int, num_heads:int, d_ff:int, rope_module=None, eps1=1e-5, eps2=1e-5, device=None, dtype=None):
        super().__init__()
        self.rope_module = rope_module
        self.attention_norm = RMSNorm(d_model=d_model,eps=eps1, device=device, dtype=dtype)
        self.attention_layer = Multihead_self_attention(d_model=d_model, num_heads=num_heads, rope_module=rope_module, device=device, dtype=dtype)
        self.ffn_norm = RMSNorm(d_model=d_model, eps=eps2, device=device, dtype=dtype)
        self.ffn_layer = FFN(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(self, input:torch.Tensor, token_positions=None):
        variable1 = self.attention_norm(input)
        variable1 = self.attention_layer(input=variable1, token_positions=token_positions)
        result_first_layer = input + variable1
        variable2 = self.ffn_norm(result_first_layer)
        variable2 = self.ffn_layer(variable2)
        output = result_first_layer + variable2
        return output
