import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import termcolor
import numpy as np
from typing import Optional, Dict, Tuple
from wenet.net.transformer.attention import MultiHeadedAttention
from wenet.net.transformer.embedding import PositionalEncoding
from wenet.utils.rotary_embedding_torch import RotaryEmbedding, RotaryEmbedding_libtorch
from tqdm import tqdm


class ResidualConnect(nn.Module):
    def __init__(self):
        super(ResidualConnect, self).__init__()

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                pos_emb: torch.Tensor = torch.empty(0)) -> torch.Tensor:
        return query


class MultiHeadedAttention_cosformer_official_degenerate(nn.Module):
    """Multi-Head Attention layer derivate from official version cosformer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, n_head: int, n_feat: int, dropout_rate: float,
                 deploy: bool = False, act_fun: str = 'relu',
                 causal: bool = False, cosformer_version: int = 0,
                 rope_type: str = 'lang'):
        """Construct an MultiHeadedAttention object.
            Args:
                enable_was (bool): Whether to enable Weak-Attention Suppression on self attention module
        """

        super().__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.act_fun = act_fun.lower()
        self.deploy = deploy
        self.causal = causal
        self.cosformer_version = cosformer_version
        self.denom_eps = 1e-6
        self.eps = 1e-6
        self.max_pos = 10000
        if self.cosformer_version == 2:
            self.random_index = nn.Parameter(
                np.pi / 2 * (torch.rand(1, self.max_pos, self.d_k, requires_grad=True) * 2 - 1))
        elif self.cosformer_version == 5:
            self.exp_proj = nn.Linear(1, self.d_k)
            self.init_weights(self.exp_proj)
        elif self.cosformer_version == 7:
            self.rotary_emb = RotaryEmbedding(dim=self.d_k // 2, freqs_for=rope_type, learned_freq=False)
        elif self.cosformer_version == 8:
            self.rotary_emb = RotaryEmbedding_libtorch(dim=self.d_k // 2, freqs_for=rope_type)
        else:
            self.random_index = nn.Parameter(np.pi / 2 * (torch.rand(1, self.max_pos, 1, requires_grad=True) * 2 - 1))

    def init_weights(self, m):
        nn.init.uniform_(m.weight)

    def get_div_term(self, hidden_dim: int):
        return torch.arange(1, hidden_dim + 1).reshape(1, 1, -1) / hidden_dim

    @torch.jit.export
    def get_index(self, seq_len: int):
        index = np.pi / 2 * torch.arange(1, seq_len + 1).reshape(1, -1, 1)
        return index

    @torch.jit.export
    def kernel(self, x: torch.Tensor) -> torch.Tensor:
        if self.act_fun == "relu":
            return F.relu(x)
        elif self.act_fun == "elu":
            return 1 + F.elu(x)
        elif self.act_fun == "sigmoid":
            return F.sigmoid(x)
        elif self.act_fun == "tanh":
            return 0.5 + 0.5 * F.tanh(x)
        else:
            # no kernel
            return x

    @torch.jit.export
    def apply_mask(self, x, mask: Optional[torch.Tensor]):
        if mask is not None:
            x = x.masked_fill((~mask).transpose(1, 2), 0.0)
        return x

    @torch.jit.export
    def convert_qkv(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)

    @torch.jit.export
    def convert_mask(self, mask: torch.Tensor) -> torch.Tensor:
        mask2 = mask.transpose(-2, -1)
        mask1 = mask.repeat(1, mask.size(-1), 1)
        mask2 = mask2.repeat(1, 1, mask2.size(-2))
        mask = mask1 * mask2
        return mask.squeeze(1)

    def left_product(
            self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        num_heads = self.h
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        head_dim = embed_dim // num_heads

        # get q, k, v
        # (L, N, E)
        q = self.linear_q(query)
        # (S, N, E)
        k = self.linear_k(key)
        # (S, N, E)
        v = self.linear_v(value)

        # activation
        q = self.kernel(q)
        k = self.kernel(k)

        # multihead reshape
        assert q is not None and k is not None and v is not None
        # (N * h, L, d)
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        # cos transform
        m = max(src_len, tgt_len)
        # get index and send to cuda
        weight_index = self.random_index.to(q)
        # (N * h, L, 2 * d)
        q_ = torch.cat(
            [q * torch.sin(weight_index[:, :tgt_len, :]),
             q * torch.cos(weight_index[:, :tgt_len, :])], dim=-1)
        # (N * h, S, 2 * d)
        k_ = torch.cat(
            [k * torch.sin(weight_index[:, :src_len, :]),
             k * torch.cos(weight_index[:, :src_len, :])], dim=-1)

        # (N * h, L, d) (N * h, d, S) -> (N * h, L, S)
        weights = torch.bmm(q_, k_.transpose(1, 2))
        # mask
        if self.causal:
            # [Warning] this part is different from official implementation.
            assert mask is not None
            mask = self.convert_mask(mask).repeat(self.h, 1, 1)
            weights = weights.masked_fill(mask, 0)
        # (N * h, L, S) -> (N * h, L, S)
        denom = torch.clamp_min(weights.sum(dim=-1, keepdim=True), self.eps)
        # (N * h, L, S) (N * h, L, S) -> (N * h, L, S)
        attn_weights = weights / denom
        # (N * h, L, S) (N * h, S, d) -> (N * h, L, d)
        attn_output = torch.bmm(attn_weights, v)
        # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        # L, N, E
        attn_output = self.linear_out(attn_output).transpose(0, 1)

        return attn_output

    def left_product_single_cos(
            self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        num_heads = self.h
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        head_dim = embed_dim // num_heads

        # get q, k, v
        # (L, N, E)
        q = self.linear_q(query)
        # (S, N, E)
        k = self.linear_k(key)
        # (S, N, E)
        v = self.linear_v(value)

        # activation
        q = self.kernel(q)
        k = self.kernel(k)

        # multihead reshape
        assert q is not None and k is not None and v is not None
        # (N * h, L, d)
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        # cos transform
        m = max(src_len, tgt_len)
        # get index and send to cuda
        weight_index = self.random_index.to(q)
        # # (N * h, L, 2 * d)
        q_ = q
        k_ = k * torch.cos(weight_index[:, :src_len, :] / m)

        # (N * h, L, d) (N * h, d, S) -> (N * h, L, S)
        weights = torch.bmm(q_, k_.transpose(1, 2))
        # mask
        if self.causal:
            # [Warning] this part is different from official implementation.
            assert mask is not None
            mask = self.convert_mask(mask).repeat(self.h, 1, 1)
            weights = weights.masked_fill(mask, 0)
        # (N * h, L, S) -> (N * h, L, S)
        denom = torch.clamp_min(weights.sum(dim=-1, keepdim=True), self.eps)
        # (N * h, L, S) (N * h, L, S) -> (N * h, L, S)
        attn_weights = weights / denom
        # (N * h, L, S) (N * h, S, d) -> (N * h, L, d)
        attn_output = torch.bmm(attn_weights, v)
        # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        # L, N, E
        attn_output = self.linear_out(attn_output).transpose(0, 1)

        return attn_output

    def left_product_single_cos2(
            self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        num_heads = self.h
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        head_dim = embed_dim // num_heads

        # get q, k, v
        # (L, N, E)
        q = self.linear_q(query)
        # (S, N, E)
        k = self.linear_k(key)
        # (S, N, E)
        v = self.linear_v(value)

        # activation
        q = self.kernel(q)
        k = self.kernel(k)

        # multihead reshape
        assert q is not None and k is not None and v is not None
        # (N * h, L, d)
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        # cos transform
        # m = max(src_len, tgt_len)
        # get index and send to cuda
        weight_index = self.random_index.to(q)
        # # (N * h, L, 2 * d)
        q_ = q
        k_ = k * torch.cos(weight_index[:, :src_len, :])

        # (N * h, L, d) (N * h, d, S) -> (N * h, L, S)
        weights = torch.bmm(q_, k_.transpose(1, 2))
        # mask
        if self.causal:
            # [Warning] this part is different from official implementation.
            assert mask is not None
            mask = self.convert_mask(mask).repeat(self.h, 1, 1)
            weights = weights.masked_fill(mask, 0)
        # (N * h, L, S) -> (N * h, L, S)
        denom = torch.clamp_min(weights.sum(dim=-1, keepdim=True), self.eps)
        # (N * h, L, S) (N * h, L, S) -> (N * h, L, S)
        attn_weights = weights / denom
        # (N * h, L, S) (N * h, S, d) -> (N * h, L, d)
        attn_output = torch.bmm(attn_weights, v)
        # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        # L, N, E
        attn_output = self.linear_out(attn_output).transpose(0, 1)

        return attn_output

    def left_product_single_cos3(
            self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        num_heads = self.h
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        head_dim = embed_dim // num_heads

        # get q, k, v
        # (L, N, E)
        q = self.linear_q(query)
        # (S, N, E)
        k = self.linear_k(key)
        # (S, N, E)
        v = self.linear_v(value)

        # activation
        q = self.kernel(q)
        k = self.kernel(k)

        # multihead reshape
        assert q is not None and k is not None and v is not None
        # (N * h, L, d)
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        # cos transform
        # m = max(src_len, tgt_len)
        # get index and send to cuda
        weight_index = self.random_index.to(q)
        div_term = self.get_div_term(self.d_k).to(q)
        # # (N * h, L, 2 * d)
        q_ = q
        k_ = k * torch.cos(weight_index[:, :src_len, :] * div_term)

        # (N * h, L, d) (N * h, d, S) -> (N * h, L, S)
        weights = torch.bmm(q_, k_.transpose(1, 2))
        # mask
        if self.causal:
            # [Warning] this part is different from official implementation.
            assert mask is not None
            mask = self.convert_mask(mask).repeat(self.h, 1, 1)
            weights = weights.masked_fill(mask, 0)
        # (N * h, L, S) -> (N * h, L, S)
        denom = torch.clamp_min(weights.sum(dim=-1, keepdim=True), self.eps)
        # (N * h, L, S) (N * h, L, S) -> (N * h, L, S)
        attn_weights = weights / denom
        # (N * h, L, S) (N * h, S, d) -> (N * h, L, d)
        attn_output = torch.bmm(attn_weights, v)
        # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        # L, N, E
        attn_output = self.linear_out(attn_output).transpose(0, 1)

        return attn_output

    def left_product_single_cos4(
            self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        num_heads = self.h
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        head_dim = embed_dim // num_heads

        # get q, k, v
        # (L, N, E)
        q = self.linear_q(query)
        # (S, N, E)
        k = self.linear_k(key)
        # (S, N, E)
        v = self.linear_v(value)

        # activation
        q = self.kernel(q)
        k = self.kernel(k)

        # multihead reshape
        assert q is not None and k is not None and v is not None
        # (N * h, L, d)
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        # cos transform
        # m = max(src_len, tgt_len)
        # get index and send to cuda
        # weight_index = self.random_index.to(q)
        # div_term = self.get_div_term(self.d_k).to(q)
        # # (N * h, L, 2 * d)
        q_ = q
        k_ = k

        # (N * h, L, d) (N * h, d, S) -> (N * h, L, S)
        weights = torch.bmm(q_, k_.transpose(1, 2))
        weights = torch.sigmoid(weights)
        # mask
        if self.causal:
            # [Warning] this part is different from official implementation.
            assert mask is not None
            mask = self.convert_mask(mask).repeat(self.h, 1, 1)
            weights = weights.masked_fill(mask, 0)
        # (N * h, L, S) -> (N * h, L, S)
        denom = torch.clamp_min(weights.sum(dim=-1, keepdim=True), self.eps)
        # (N * h, L, S) (N * h, L, S) -> (N * h, L, S)
        attn_weights = weights / denom
        # (N * h, L, S) (N * h, S, d) -> (N * h, L, d)
        attn_output = torch.bmm(attn_weights, v)
        # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        # L, N, E
        attn_output = self.linear_out(attn_output).transpose(0, 1)

        return attn_output

    def left_product_single_cos5(
            self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        num_heads = self.h
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        head_dim = embed_dim // num_heads

        # get q, k, v
        # (L, N, E)
        q = self.linear_q(query)
        # (S, N, E)
        k = self.linear_k(key)
        # (S, N, E)
        v = self.linear_v(value)

        # activation
        q = self.kernel(q)
        k = self.kernel(k)

        # multihead reshape
        assert q is not None and k is not None and v is not None
        # (N * h, L, d)
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        # cos transform
        m = max(src_len, tgt_len)
        # get index and send to cuda
        weight_index = self.get_index(m).to(q)
        # # (N * h, L, 2 * d)
        q_ = q
        k_ = k * torch.cos(self.exp_proj(weight_index[:, :src_len, :] / m))

        # (N * h, L, d) (N * h, d, S) -> (N * h, L, S)
        weights = torch.bmm(q_, k_.transpose(1, 2))
        # mask
        if self.causal:
            # [Warning] this part is different from official implementation.
            assert mask is not None
            mask = self.convert_mask(mask).repeat(self.h, 1, 1)
            weights = weights.masked_fill(mask, 0)
        # (N * h, L, S) -> (N * h, L, S)
        denom = torch.clamp_min(weights.sum(dim=-1, keepdim=True), self.eps)
        # (N * h, L, S) (N * h, L, S) -> (N * h, L, S)
        attn_weights = weights / denom
        # (N * h, L, S) (N * h, S, d) -> (N * h, L, d)
        attn_output = torch.bmm(attn_weights, v)
        # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        # L, N, E
        attn_output = self.linear_out(attn_output).transpose(0, 1)

        return attn_output

    def left_product_only(
            self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        num_heads = self.h
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        head_dim = embed_dim // num_heads

        # get q, k, v
        # (L, N, E)
        q = self.linear_q(query)
        # (S, N, E)
        k = self.linear_k(key)
        # (S, N, E)
        v = self.linear_v(value)

        # activation
        q = self.kernel(q)
        k = self.kernel(k)

        # multihead reshape
        assert q is not None and k is not None and v is not None
        # (N * h, L, d)
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        # cos transform
        m = max(src_len, tgt_len)
        # get index and send to cuda
        weight_index = self.get_index(m).to(q)
        # # (N * h, L, 2 * d)
        q_ = q
        k_ = k

        # (N * h, L, d) (N * h, d, S) -> (N * h, L, S)
        weights = torch.bmm(q_, k_.transpose(1, 2))
        # mask
        if self.causal:
            # [Warning] this part is different from official implementation.
            assert mask is not None
            mask = self.convert_mask(mask).repeat(self.h, 1, 1)
            weights = weights.masked_fill(mask, 0)
        # (N * h, L, S) -> (N * h, L, S)
        denom = torch.clamp_min(weights.sum(dim=-1, keepdim=True), self.eps)
        # (N * h, L, S) (N * h, L, S) -> (N * h, L, S)
        attn_weights = weights / denom
        # (N * h, L, S) (N * h, S, d) -> (N * h, L, d)
        attn_output = torch.bmm(attn_weights, v)
        # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        # L, N, E
        attn_output = self.linear_out(attn_output).transpose(0, 1)

        return attn_output

    def left_product_rope(
            self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        num_heads = self.h
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        head_dim = embed_dim // num_heads

        # get q, k, v
        # (L, N, E)
        q = self.linear_q(query)
        # (S, N, E)
        k = self.linear_k(key)
        # (S, N, E)
        v = self.linear_v(value)

        # activation
        q = self.kernel(q)
        k = self.kernel(k)

        # multihead reshape
        assert q is not None and k is not None and v is not None
        # (N * h, L, d)
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        # # (N * h, L, 2 * d)
        q_ = self.rotary_emb.rotate_queries_or_keys(q)
        k_ = self.rotary_emb.rotate_queries_or_keys(k)

        # (N * h, L, d) (N * h, d, S) -> (N * h, L, S)
        weights = torch.bmm(q_, k_.transpose(1, 2))
        # mask
        if self.causal:
            # [Warning] this part is different from official implementation.
            assert mask is not None
            mask = self.convert_mask(mask).repeat(self.h, 1, 1)
            weights = weights.masked_fill(mask, 0)
        # (N * h, L, S) -> (N * h, L, S)
        denom = torch.clamp_min(weights.sum(dim=-1, keepdim=True), self.eps)
        # (N * h, L, S) (N * h, L, S) -> (N * h, L, S)
        attn_weights = weights / denom
        # (N * h, L, S) (N * h, S, d) -> (N * h, L, d)
        attn_output = torch.bmm(attn_weights, v)
        # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        # L, N, E
        attn_output = self.linear_out(attn_output).transpose(0, 1)

        return attn_output

    def left_product_rope2(
            self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        num_heads = self.h
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        head_dim = embed_dim // num_heads

        # get q, k, v
        # (L, N, E)
        q = self.linear_q(query)
        # (S, N, E)
        k = self.linear_k(key)
        # (S, N, E)
        v = self.linear_v(value)

        # activation
        q = self.kernel(q)
        k = self.kernel(k)

        # multihead reshape
        assert q is not None and k is not None and v is not None
        # (N * h, L, d)
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        # # (N * h, L, 2 * d)
        q_ = self.rotary_emb(q)
        k_ = self.rotary_emb(k)

        # (N * h, L, d) (N * h, d, S) -> (N * h, L, S)
        weights = torch.bmm(q_, k_.transpose(1, 2))
        # mask
        if self.causal:
            # [Warning] this part is different from official implementation.
            assert mask is not None
            mask = self.convert_mask(mask).repeat(self.h, 1, 1)
            weights = weights.masked_fill(mask, 0)
        # (N * h, L, S) -> (N * h, L, S)
        denom = torch.clamp_min(weights.sum(dim=-1, keepdim=True), self.eps)
        # (N * h, L, S) (N * h, L, S) -> (N * h, L, S)
        attn_weights = weights / denom
        # (N * h, L, S) (N * h, S, d) -> (N * h, L, d)
        attn_output = torch.bmm(attn_weights, v)
        # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        # L, N, E
        attn_output = self.linear_out(attn_output).transpose(0, 1)

        return attn_output

    def right_product(
            self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor, size
                (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor, size
                (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor, size
                (#batch, n_head, time2, d_k).

        """
        num_heads = self.h
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        head_dim = embed_dim // num_heads
        # get q, k, v
        # (L, N, E)
        q = self.linear_q(query)
        # (S, N, E)
        k = self.linear_k(key)
        # (S, N, E)
        v = self.linear_v(value)

        q = self.kernel(q)
        k = self.kernel(k)

        # multihead reshape
        assert q is not None and k is not None and v is not None
        # (N * h, L, d)
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        # cos transform
        m = max(src_len, tgt_len)
        # get index and send to cuda
        weight_index = self.random_index.to(q)
        q_ = torch.cat(
            [q * torch.sin(weight_index[:, :tgt_len, :]),
             q * torch.cos(weight_index[:, :tgt_len, :])], dim=-1)
        # (N * h, S, 2 * d)
        k_ = torch.cat(
            [k * torch.sin(weight_index[:, :src_len, :]),
             k * torch.cos(weight_index[:, :src_len, :])], dim=-1)

        if self.causal:
            ## Need to improve speed!
            # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, L, h, 2 * d, d)
            kv_ = torch.einsum("nld,nlm->nldm", k_, v)
            # (N * h, L, 2 * d, d) -> (N * h, L, 2 * d, d)
            kv_cum = torch.cumsum(kv_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
            qkv = torch.einsum("nld,nldm->nlm", q_, kv_cum)
            # (N * h, L, 2 * d) -> (N * h, L, 2 * d)
            k_cum = torch.cumsum(k_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d) -> (N * h, L)
            denom = torch.clamp_min(torch.einsum("nlm,nlm->nl", q_, k_cum), self.eps)
            # (N * h, L, d) (N * h, L, 1) -> (N * h, L, d)
            attn_output = qkv / denom.unsqueeze(-1)
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        else:
            # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, 2 * d, d)
            kv_ = torch.einsum('nld,nlm->ndm', k_, v)
            # (N * h, L, 2 * d) (N * h, 2 * d) -> (N * h, L)
            z_ = 1 / torch.clamp_min(torch.einsum('nld,nd->nl', q_, torch.sum(k_, dim=1)), self.eps)
            # (N * h, L, 2 * d) (N * h, d, 2 * d) (N * h, L) -> (N * h, L, d)
            attn_output = torch.einsum('nld,ndm,nl->nlm', q_, kv_, z_)
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        # L, N, E
        attn_output = self.linear_out(attn_output).transpose(0, 1)

        return attn_output

    def right_product_single_cos(
            self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor, size
                (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor, size
                (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor, size
                (#batch, n_head, time2, d_k).

        """
        num_heads = self.h
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        head_dim = embed_dim // num_heads
        # get q, k, v
        # (L, N, E)
        q = self.linear_q(query)
        # (S, N, E)
        k = self.linear_k(key)
        # (S, N, E)
        v = self.linear_v(value)

        q = self.kernel(q)
        k = self.kernel(k)

        # multihead reshape
        assert q is not None and k is not None and v is not None
        # (N * h, L, d)
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        # cos transform
        m = max(src_len, tgt_len)
        # get index and send to cuda
        weight_index = self.random_index.to(q)

        q_ = q
        k_ = k * torch.cos(weight_index[:, :src_len, :] / m)

        if self.causal:
            ## Need to improve speed!
            # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, L, h, 2 * d, d)
            kv_ = torch.einsum("nld,nlm->nldm", k_, v)
            # (N * h, L, 2 * d, d) -> (N * h, L, 2 * d, d)
            kv_cum = torch.cumsum(kv_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
            qkv = torch.einsum("nld,nldm->nlm", q_, kv_cum)
            # (N * h, L, 2 * d) -> (N * h, L, 2 * d)
            k_cum = torch.cumsum(k_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d) -> (N * h, L)
            denom = torch.clamp_min(torch.einsum("nlm,nlm->nl", q_, k_cum), self.eps)
            # (N * h, L, d) (N * h, L, 1) -> (N * h, L, d)
            attn_output = qkv / denom.unsqueeze(-1)
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        else:
            # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, 2 * d, d)
            kv_ = torch.einsum('nld,nlm->ndm', k_, v)
            # (N * h, L, 2 * d) (N * h, 2 * d) -> (N * h, L)
            z_ = 1 / torch.clamp_min(torch.einsum('nld,nd->nl', q_, torch.sum(k_, dim=1)), self.eps)
            # (N * h, L, 2 * d) (N * h, d, 2 * d) (N * h, L) -> (N * h, L, d)
            attn_output = torch.einsum('nld,ndm,nl->nlm', q_, kv_, z_)
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        # L, N, E
        attn_output = self.linear_out(attn_output).transpose(0, 1)

        return attn_output

    def right_product_single_cos2(
            self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor, size
                (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor, size
                (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor, size
                (#batch, n_head, time2, d_k).

        """
        num_heads = self.h
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        head_dim = embed_dim // num_heads
        # get q, k, v
        # (L, N, E)
        q = self.linear_q(query)
        # (S, N, E)
        k = self.linear_k(key)
        # (S, N, E)
        v = self.linear_v(value)

        q = self.kernel(q)
        k = self.kernel(k)

        # multihead reshape
        assert q is not None and k is not None and v is not None
        # (N * h, L, d)
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        # cos transform
        m = max(src_len, tgt_len)
        # get index and send to cuda
        weight_index = self.random_index.to(q)

        q_ = q
        k_ = k * torch.cos(weight_index[:, :src_len, :])

        if self.causal:
            ## Need to improve speed!
            # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, L, h, 2 * d, d)
            kv_ = torch.einsum("nld,nlm->nldm", k_, v)
            # (N * h, L, 2 * d, d) -> (N * h, L, 2 * d, d)
            kv_cum = torch.cumsum(kv_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
            qkv = torch.einsum("nld,nldm->nlm", q_, kv_cum)
            # (N * h, L, 2 * d) -> (N * h, L, 2 * d)
            k_cum = torch.cumsum(k_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d) -> (N * h, L)
            denom = torch.clamp_min(torch.einsum("nlm,nlm->nl", q_, k_cum), self.eps)
            # (N * h, L, d) (N * h, L, 1) -> (N * h, L, d)
            attn_output = qkv / denom.unsqueeze(-1)
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        else:
            # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, 2 * d, d)
            kv_ = torch.einsum('nld,nlm->ndm', k_, v)
            # (N * h, L, 2 * d) (N * h, 2 * d) -> (N * h, L)
            z_ = 1 / torch.clamp_min(torch.einsum('nld,nd->nl', q_, torch.sum(k_, dim=1)), self.eps)
            # (N * h, L, 2 * d) (N * h, d, 2 * d) (N * h, L) -> (N * h, L, d)
            attn_output = torch.einsum('nld,ndm,nl->nlm', q_, kv_, z_)
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        # L, N, E
        attn_output = self.linear_out(attn_output).transpose(0, 1)

        return attn_output

    def right_product_single_cos3(
            self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor, size
                (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor, size
                (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor, size
                (#batch, n_head, time2, d_k).

        """
        num_heads = self.h
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        head_dim = embed_dim // num_heads
        # get q, k, v
        # (L, N, E)
        q = self.linear_q(query)
        # (S, N, E)
        k = self.linear_k(key)
        # (S, N, E)
        v = self.linear_v(value)

        q = self.kernel(q)
        k = self.kernel(k)

        # multihead reshape
        assert q is not None and k is not None and v is not None
        # (N * h, L, d)
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        # cos transform
        m = max(src_len, tgt_len)
        # get index and send to cuda
        weight_index = self.random_index.to(q)
        div_term = self.get_div_term(self.d_k).to(q)

        q_ = q
        k_ = k * torch.cos(weight_index[:, :src_len, :] * div_term)

        if self.causal:
            ## Need to improve speed!
            # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, L, h, 2 * d, d)
            kv_ = torch.einsum("nld,nlm->nldm", k_, v)
            # (N * h, L, 2 * d, d) -> (N * h, L, 2 * d, d)
            kv_cum = torch.cumsum(kv_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
            qkv = torch.einsum("nld,nldm->nlm", q_, kv_cum)
            # (N * h, L, 2 * d) -> (N * h, L, 2 * d)
            k_cum = torch.cumsum(k_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d) -> (N * h, L)
            denom = torch.clamp_min(torch.einsum("nlm,nlm->nl", q_, k_cum), self.eps)
            # (N * h, L, d) (N * h, L, 1) -> (N * h, L, d)
            attn_output = qkv / denom.unsqueeze(-1)
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        else:
            # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, 2 * d, d)
            kv_ = torch.einsum('nld,nlm->ndm', k_, v)
            # (N * h, L, 2 * d) (N * h, 2 * d) -> (N * h, L)
            z_ = 1 / torch.clamp_min(torch.einsum('nld,nd->nl', q_, torch.sum(k_, dim=1)), self.eps)
            # (N * h, L, 2 * d) (N * h, d, 2 * d) (N * h, L) -> (N * h, L, d)
            attn_output = torch.einsum('nld,ndm,nl->nlm', q_, kv_, z_)
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        # L, N, E
        attn_output = self.linear_out(attn_output).transpose(0, 1)

        return attn_output

    def right_product_single_cos4(
            self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor, size
                (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor, size
                (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor, size
                (#batch, n_head, time2, d_k).

        """
        num_heads = self.h
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        head_dim = embed_dim // num_heads
        # get q, k, v
        # (L, N, E)
        q = self.linear_q(query)
        # (S, N, E)
        k = self.linear_k(key)
        # (S, N, E)
        v = self.linear_v(value)

        q = self.kernel(q)
        k = self.kernel(k)

        # multihead reshape
        assert q is not None and k is not None and v is not None
        # (N * h, L, d)
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        # cos transform
        m = max(src_len, tgt_len)
        # get index and send to cuda
        weight_index = self.random_index.to(q)
        # div_term = self.get_div_term(self.d_k).to(q)

        q_ = q
        k_ = k

        if self.causal:
            ## Need to improve speed!
            # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, L, h, 2 * d, d)
            kv_ = torch.einsum("nld,nlm->nldm", k_, v)
            # (N * h, L, 2 * d, d) -> (N * h, L, 2 * d, d)
            kv_cum = torch.cumsum(kv_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
            qkv = torch.einsum("nld,nldm->nlm", q_, kv_cum)
            # (N * h, L, 2 * d) -> (N * h, L, 2 * d)
            k_cum = torch.cumsum(k_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d) -> (N * h, L)
            denom = torch.clamp_min(torch.einsum("nlm,nlm->nl", q_, k_cum), self.eps)
            # (N * h, L, d) (N * h, L, 1) -> (N * h, L, d)
            attn_output = qkv / denom.unsqueeze(-1)
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        else:
            # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, 2 * d, d)
            kv_ = torch.einsum('nld,nlm->ndm', k_, v)
            # (N * h, L, 2 * d) (N * h, 2 * d) -> (N * h, L)
            z_ = 1 / torch.clamp_min(torch.einsum('nld,nd->nl', q_, torch.sum(k_, dim=1)), self.eps)
            # (N * h, L, 2 * d) (N * h, d, 2 * d) (N * h, L) -> (N * h, L, d)
            attn_output = torch.einsum('nld,ndm,nl->nlm', q_, kv_, z_)
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        # L, N, E
        attn_output = self.linear_out(attn_output).transpose(0, 1)

        return attn_output

    def right_product_single_cos5(
            self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor, size
                (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor, size
                (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor, size
                (#batch, n_head, time2, d_k).

        """
        num_heads = self.h
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        head_dim = embed_dim // num_heads
        # get q, k, v
        # (L, N, E)
        q = self.linear_q(query)
        # (S, N, E)
        k = self.linear_k(key)
        # (S, N, E)
        v = self.linear_v(value)

        q = self.kernel(q)
        k = self.kernel(k)

        # multihead reshape
        assert q is not None and k is not None and v is not None
        # (N * h, L, d)
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        # cos transform
        m = max(src_len, tgt_len)
        # get index and send to cuda
        weight_index = self.get_index(m).to(q)

        q_ = q
        k_ = k * torch.cos(self.exp_proj(weight_index[:, :src_len, :] / m))

        if self.causal:
            ## Need to improve speed!
            # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, L, h, 2 * d, d)
            kv_ = torch.einsum("nld,nlm->nldm", k_, v)
            # (N * h, L, 2 * d, d) -> (N * h, L, 2 * d, d)
            kv_cum = torch.cumsum(kv_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
            qkv = torch.einsum("nld,nldm->nlm", q_, kv_cum)
            # (N * h, L, 2 * d) -> (N * h, L, 2 * d)
            k_cum = torch.cumsum(k_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d) -> (N * h, L)
            denom = torch.clamp_min(torch.einsum("nlm,nlm->nl", q_, k_cum), self.eps)
            # (N * h, L, d) (N * h, L, 1) -> (N * h, L, d)
            attn_output = qkv / denom.unsqueeze(-1)
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        else:
            # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, 2 * d, d)
            kv_ = torch.einsum('nld,nlm->ndm', k_, v)
            # (N * h, L, 2 * d) (N * h, 2 * d) -> (N * h, L)
            z_ = 1 / torch.clamp_min(torch.einsum('nld,nd->nl', q_, torch.sum(k_, dim=1)), self.eps)
            # (N * h, L, 2 * d) (N * h, d, 2 * d) (N * h, L) -> (N * h, L, d)
            attn_output = torch.einsum('nld,ndm,nl->nlm', q_, kv_, z_)
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        # L, N, E
        attn_output = self.linear_out(attn_output).transpose(0, 1)

        return attn_output

    def right_product_only(
            self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor, size
                (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor, size
                (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor, size
                (#batch, n_head, time2, d_k).

        """
        num_heads = self.h
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        head_dim = embed_dim // num_heads
        # get q, k, v
        # (L, N, E)
        q = self.linear_q(query)
        # (S, N, E)
        k = self.linear_k(key)
        # (S, N, E)
        v = self.linear_v(value)

        q = self.kernel(q)
        k = self.kernel(k)

        # multihead reshape
        assert q is not None and k is not None and v is not None
        # (N * h, L, d)
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        # cos transform
        # m = max(src_len, tgt_len)
        # get index and send to cuda
        # weight_index = self.get_index(m).to(q)

        q_ = q
        k_ = k

        if self.causal:
            ## Need to improve speed!
            # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, L, h, 2 * d, d)
            kv_ = torch.einsum("nld,nlm->nldm", k_, v)
            # (N * h, L, 2 * d, d) -> (N * h, L, 2 * d, d)
            kv_cum = torch.cumsum(kv_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
            qkv = torch.einsum("nld,nldm->nlm", q_, kv_cum)
            # (N * h, L, 2 * d) -> (N * h, L, 2 * d)
            k_cum = torch.cumsum(k_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d) -> (N * h, L)
            denom = torch.clamp_min(torch.einsum("nlm,nlm->nl", q_, k_cum), self.eps)
            # (N * h, L, d) (N * h, L, 1) -> (N * h, L, d)
            attn_output = qkv / denom.unsqueeze(-1)
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        else:
            # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, 2 * d, d)
            kv_ = torch.einsum('nld,nlm->ndm', k_, v)
            # (N * h, L, 2 * d) (N * h, 2 * d) -> (N * h, L)
            z_ = 1 / torch.clamp_min(torch.einsum('nld,nd->nl', q_, torch.sum(k_, dim=1)), self.eps)
            # (N * h, L, 2 * d) (N * h, d, 2 * d) (N * h, L) -> (N * h, L, d)
            attn_output = torch.einsum('nld,ndm,nl->nlm', q_, kv_, z_)
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        # L, N, E
        attn_output = self.linear_out(attn_output).transpose(0, 1)

        return attn_output

    def right_product_rope(
            self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor, size
                (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor, size
                (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor, size
                (#batch, n_head, time2, d_k).

        """
        num_heads = self.h
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        head_dim = embed_dim // num_heads
        # get q, k, v
        # (L, N, E)
        q = self.linear_q(query)
        # (S, N, E)
        k = self.linear_k(key)
        # (S, N, E)
        v = self.linear_v(value)

        q = self.kernel(q)
        k = self.kernel(k)

        # multihead reshape
        assert q is not None and k is not None and v is not None
        # (N * h, L, d)
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        # cos transform
        m = max(src_len, tgt_len)

        q_ = self.rotary_emb.rotate_queries_or_keys(q)
        k_ = self.rotary_emb.rotate_queries_or_keys(k)

        if self.causal:
            ## Need to improve speed!
            # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, L, h, 2 * d, d)
            kv_ = torch.einsum("nld,nlm->nldm", k_, v)
            # (N * h, L, 2 * d, d) -> (N * h, L, 2 * d, d)
            kv_cum = torch.cumsum(kv_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
            qkv = torch.einsum("nld,nldm->nlm", q_, kv_cum)
            # (N * h, L, 2 * d) -> (N * h, L, 2 * d)
            k_cum = torch.cumsum(k_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d) -> (N * h, L)
            denom = torch.clamp_min(torch.einsum("nlm,nlm->nl", q_, k_cum), self.eps)
            # (N * h, L, d) (N * h, L, 1) -> (N * h, L, d)
            attn_output = qkv / denom.unsqueeze(-1)
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        else:
            # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, 2 * d, d)
            kv_ = torch.einsum('nld,nlm->ndm', k_, v)
            # (N * h, L, 2 * d) (N * h, 2 * d) -> (N * h, L)
            z_ = 1 / torch.clamp_min(torch.einsum('nld,nd->nl', q_, torch.sum(k_, dim=1)), self.eps)
            # (N * h, L, 2 * d) (N * h, d, 2 * d) (N * h, L) -> (N * h, L, d)
            attn_output = torch.einsum('nld,ndm,nl->nlm', q_, kv_, z_)
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        # L, N, E
        attn_output = self.linear_out(attn_output).transpose(0, 1)

        return attn_output

    def right_product_rope2(
            self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor, size
                (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor, size
                (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor, size
                (#batch, n_head, time2, d_k).

        """
        num_heads = self.h
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        head_dim = embed_dim // num_heads
        # get q, k, v
        # (L, N, E)
        q = self.linear_q(query)
        # (S, N, E)
        k = self.linear_k(key)
        # (S, N, E)
        v = self.linear_v(value)

        q = self.kernel(q)
        k = self.kernel(k)

        # multihead reshape
        assert q is not None and k is not None and v is not None
        # (N * h, L, d)
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        # cos transform
        m = max(src_len, tgt_len)

        q_ = self.rotary_emb(q)
        k_ = self.rotary_emb(k)

        if self.causal:
            ## Need to improve speed!
            # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, L, h, 2 * d, d)
            kv_ = torch.einsum("nld,nlm->nldm", k_, v)
            # (N * h, L, 2 * d, d) -> (N * h, L, 2 * d, d)
            kv_cum = torch.cumsum(kv_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
            qkv = torch.einsum("nld,nldm->nlm", q_, kv_cum)
            # (N * h, L, 2 * d) -> (N * h, L, 2 * d)
            k_cum = torch.cumsum(k_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d) -> (N * h, L)
            denom = torch.clamp_min(torch.einsum("nlm,nlm->nl", q_, k_cum), self.eps)
            # (N * h, L, d) (N * h, L, 1) -> (N * h, L, d)
            attn_output = qkv / denom.unsqueeze(-1)
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        else:
            # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, 2 * d, d)
            kv_ = torch.einsum('nld,nlm->ndm', k_, v)
            # (N * h, L, 2 * d) (N * h, 2 * d) -> (N * h, L)
            z_ = 1 / torch.clamp_min(torch.einsum('nld,nd->nl', q_, torch.sum(k_, dim=1)), self.eps)
            # (N * h, L, 2 * d) (N * h, d, 2 * d) (N * h, L) -> (N * h, L, d)
            attn_output = torch.einsum('nld,ndm,nl->nlm', q_, kv_, z_)
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        # L, N, E
        attn_output = self.linear_out(attn_output).transpose(0, 1)

        return attn_output

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                pos_emb: torch.Tensor = torch.empty(0)) -> torch.Tensor:
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
                1.When applying cross attention between decoder and encoder,
                the batch padding mask for input is in (#batch, 1, T) shape.
                2.When applying self attention of encoder,
                the mask is in (#batch, T, T)  shape.
                3.When applying self attention of decoder,
                the mask is in (#batch, L, L)  shape.
                4.If the different position in decoder see different block
                of the encoder, such as Mocha, the passed in mask could be
                in (#batch, L, T) shape. But there is no such case in current
                Wenet.


        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        query, key, value = self.convert_qkv(query, key, value)
        if self.deploy:
            if self.cosformer_version == 1:
                x = self.right_product_single_cos(query, key, value, mask)
            elif self.cosformer_version == 2:  # enlarge m
                x = self.right_product_single_cos2(query, key, value, mask)
            elif self.cosformer_version == 3:  # with div term
                x = self.right_product_single_cos3(query, key, value, mask)
            elif self.cosformer_version == 4:  # with sigmoid(q, k)
                # x = self.right_product_single_cos4(query, key, value, mask) # not correct
                raise Exception("Please check your configs!")
            elif self.cosformer_version == 5:
                x = self.right_product_single_cos5(query, key, value, mask)
            elif self.cosformer_version == 6:
                x = self.right_product_only(query, key, value, mask)
            elif self.cosformer_version == 7:
                x = self.right_product_rope(query, key, value, mask)
            elif self.cosformer_version == 8:
                x = self.right_product_rope2(query, key, value, mask)
            else:  # 0
                x = self.right_product(query, key, value, mask)
        else:
            if self.cosformer_version == 1:
                x = self.left_product_single_cos(query, key, value, mask)
            elif self.cosformer_version == 2:  # enlarge m
                x = self.left_product_single_cos2(query, key, value, mask)
            elif self.cosformer_version == 3:  # with div term
                x = self.left_product_single_cos3(query, key, value, mask)
            elif self.cosformer_version == 4:  # with sigmoid(q, k)
                x = self.left_product_single_cos4(query, key, value, mask)
            elif self.cosformer_version == 5:  # with sigmoid(q, k)
                x = self.left_product_single_cos5(query, key, value, mask)
            elif self.cosformer_version == 6:  # with sigmoid(q, k)
                x = self.left_product_only(query, key, value, mask)
            elif self.cosformer_version == 7:  # with sigmoid(q, k)
                x = self.left_product_rope(query, key, value, mask)
            elif self.cosformer_version == 8:  # with sigmoid(q, k)
                x = self.left_product_rope2(query, key, value, mask)
            else:  # 0
                x = self.left_product(query, key, value, mask)
        return x


class MultiHeadedAttention_cosformer_yyg(MultiHeadedAttention_cosformer_official_degenerate):
    def __init__(self, n_head: int, n_feat: int, dropout_rate: float,
                 deploy: bool = False, act_fun: str = 'relu', causal: bool = False, cosformer_version: int = 0):
        super(MultiHeadedAttention_cosformer_yyg, self).__init__(
            n_head, n_feat, dropout_rate, deploy, act_fun, causal, cosformer_version)
        self.random_index = nn.Parameter(
            np.pi / 2 * (torch.rand(1, self.max_pos, self.d_k, requires_grad=True) * 2 - 1))


class MultiHeadedAttention_cosformer_official(nn.Module):
    """Multi-Head Attention layer derivate from official version cosformer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, n_head: int, n_feat: int, dropout_rate: float,
                 deploy: bool = False, act_fun: str = 'relu',
                 causal: bool = False, cosformer_version: int = 0):
        """Construct an MultiHeadedAttention object.
            Args:
                enable_was (bool): Whether to enable Weak-Attention Suppression on self attention module
        """

        super().__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.act_fun = act_fun.lower()
        self.deploy = deploy
        self.causal = causal
        self.cosformer_version = cosformer_version
        self.denom_eps = 1e-6
        self.eps = 1e-6
        if cosformer_version == 3:
            self.linear_pos = nn.Linear(self.d_k // 2, self.d_k // 2, bias=False)

    @torch.jit.export
    def kernel(self, x: torch.Tensor) -> torch.Tensor:
        if self.act_fun == "relu":
            return F.relu(x)
        elif self.act_fun == "elu":
            return 1 + F.elu(x)
        elif self.act_fun == "sigmoid":
            return F.sigmoid(x)
        elif self.act_fun == "tanh":
            return 0.5 + 0.5 * F.tanh(x)
        else:
            # no kernel
            return x

    @torch.jit.export
    def get_div_term_index(self):
        return torch.exp(torch.arange(0, self.d_k, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d_k))

    @torch.jit.export
    def get_index(self, seq_len: int):
        index = np.pi / 2 * torch.arange(1, seq_len + 1).reshape(1, -1, 1)
        return index

    @torch.jit.export
    def get_index2(self, seq_len: int):
        index = torch.arange(1, seq_len + 1).reshape(1, -1, 1)
        return index

    @torch.jit.export
    def apply_mask(self, x, mask: Optional[torch.Tensor]):
        if mask is not None:
            x = x.masked_fill((~mask).transpose(1, 2), 0.0)
        return x

    @torch.jit.export
    def convert_qkv(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)

    @torch.jit.export
    def convert_mask(self, mask: torch.Tensor) -> torch.Tensor:
        mask2 = mask.transpose(-2, -1)
        mask1 = mask.repeat(1, mask.size(-1), 1)
        mask2 = mask2.repeat(1, 1, mask2.size(-2))
        mask = mask1 * mask2
        return mask.squeeze(1)

    def left_product(
            self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        query, key, value = self.convert_qkv(query, key, value)
        num_heads = self.h
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        head_dim = embed_dim // num_heads

        # get q, k, v
        # (L, N, E)
        q = self.linear_q(query)
        # (S, N, E)
        k = self.linear_k(key)
        # (S, N, E)
        v = self.linear_v(value)

        # activation
        q = self.kernel(q)
        k = self.kernel(k)

        # multihead reshape
        assert q is not None and k is not None and v is not None
        # (N * h, L, d)
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        # cos transform
        m = max(src_len, tgt_len)
        # get index and send to cuda
        weight_index = self.get_index(m).to(q)
        # (N * h, L, 2 * d)
        q_ = torch.cat(
            [q * torch.sin(weight_index[:, :tgt_len, :] / m),
             q * torch.cos(weight_index[:, :tgt_len, :] / m)], dim=-1)
        # (N * h, S, 2 * d)
        k_ = torch.cat(
            [k * torch.sin(weight_index[:, :src_len, :] / m),
             k * torch.cos(weight_index[:, :src_len, :] / m)], dim=-1)

        # (N * h, L, d) (N * h, d, S) -> (N * h, L, S)
        weights = torch.bmm(q_, k_.transpose(1, 2))
        # mask
        if self.causal:
            # [Warning] this part is different from official implementation.
            assert mask is not None
            mask = self.convert_mask(mask).repeat(self.h, 1, 1)
            weights = weights.masked_fill(mask, 0)
        # (N * h, L, S) -> (N * h, L, S)
        denom = torch.clamp_min(weights.sum(dim=-1, keepdim=True), self.eps)
        # (N * h, L, S) (N * h, L, S) -> (N * h, L, S)
        attn_weights = weights / denom
        # (N * h, L, S) (N * h, S, d) -> (N * h, L, d)
        attn_output = torch.bmm(attn_weights, v)
        # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        # L, N, E
        attn_output = self.linear_out(attn_output).transpose(0, 1)

        return attn_output

    def left_product_sin(
            self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        query, key, value = self.convert_qkv(query, key, value)
        num_heads = self.h
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        head_dim = embed_dim // num_heads

        # get q, k, v
        # (L, N, E)
        q = self.linear_q(query)
        # (S, N, E)
        k = self.linear_k(key)
        # (S, N, E)
        v = self.linear_v(value)

        # activation
        q = self.kernel(q)
        k = self.kernel(k)

        # multihead reshape
        assert q is not None and k is not None and v is not None
        # (N * h, L, d)
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        # cos transform
        m = max(src_len, tgt_len)
        # get index and send to cuda
        weight_index = self.get_index(m).to(q)
        # (N * h, L, 3 * d)
        q_ = torch.cat(
            [
                q * torch.sin(weight_index[:, :tgt_len, :] / m) * 0.5,
                q * torch.cos(weight_index[:, :tgt_len, :] / m) * -0.5,
                q * torch.ones(bsz * num_heads, tgt_len, self.d_k, device=q.device) * 0.5
            ], dim=-1)
        # (N * h, S, 3 * d)
        k_ = torch.cat(
            [
                k * torch.cos(weight_index[:, :src_len, :] / m),
                k * torch.sin(weight_index[:, :src_len, :] / m),
                k * torch.ones(bsz * num_heads, src_len, self.d_k, device=q.device)
            ], dim=-1)

        # (N * h, L, d) (N * h, d, S) -> (N * h, L, S)
        weights = torch.bmm(q_, k_.transpose(1, 2))
        # (N * h, L, S) -> (N * h, L, S)
        denom = torch.clamp_min(weights.sum(dim=-1, keepdim=True), self.eps)
        # (N * h, L, S) (N * h, L, S) -> (N * h, L, S)
        attn_weights = weights / denom
        # (N * h, L, S) (N * h, S, d) -> (N * h, L, d)
        attn_output = torch.bmm(attn_weights, v)
        # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        # L, N, E
        attn_output = self.linear_out(attn_output).transpose(0, 1)

        return attn_output

    def left_product_sin_cos(
            self,
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        query, key, value = self.convert_qkv(query, key, value)
        num_heads = self.h
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        head_dim = embed_dim // num_heads

        # get q, k, v
        # (L, N, E)
        q = self.linear_q(query)
        # (S, N, E)
        k = self.linear_k(key)
        # (S, N, E)
        v = self.linear_v(value)

        # activation
        q = self.kernel(q)
        k = self.kernel(k)

        # multihead reshape
        # (N * h, L, d)
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        # cos transformer
        m = max(src_len, tgt_len)
        # get index and send to cuda
        weight_index = self.get_index(m).to(q)
        # q_even = q[:, :, 0::2]
        # q_odd = q[:, :, 1::2]
        # (N * h, L, 2 * d)
        q_even = torch.cat(
            [q[:, :, 0::2] * torch.sin(weight_index[:, :tgt_len, :] / m) * 0.5,
             q[:, :, 0::2] * torch.cos(weight_index[:, :tgt_len, :] / m) * -1.0 * 0.5,
             q[:, :, 0::2] * torch.ones(bsz * num_heads, tgt_len, self.d_k // 2, device=q.device) * 0.5
             ], dim=-1)
        # (N * h, S, 2 * d)
        k_even = torch.cat(
            [k[:, :, 0::2] * torch.cos(weight_index[:, :src_len, :] / m),
             k[:, :, 0::2] * torch.sin(weight_index[:, :src_len, :] / m),
             k[:, :, 0::2] * torch.ones(bsz * num_heads, src_len, self.d_k // 2, device=q.device)
             ], dim=-1)
        # (N * h, L, 2 * d)
        q_odd = torch.cat(
            [q[:, :, 1::2] * torch.sin(weight_index[:, :tgt_len, :] / m),
             q[:, :, 1::2] * torch.cos(weight_index[:, :tgt_len, :] / m),
             # torch.full((bsz * num_heads, tgt_len, self.d_k), 1 / math.sqrt(self.d_k), device=q.device)
             ], dim=-1)
        # (N * h, S, 2 * d)
        k_odd = torch.cat(
            [k[:, :, 1::2] * torch.sin(weight_index[:, :src_len, :] / m),
             k[:, :, 1::2] * torch.cos(weight_index[:, :src_len, :] / m),
             # torch.full((bsz * num_heads, src_len, self.d_k), 1 / math.sqrt(self.d_k), device=q.device)
             ], dim=-1)

        # (N * h, L, d) (N * h, d, S) -> (N * h, L, S)
        weights_even = torch.bmm(q_even, k_even.transpose(1, 2))
        weights_odd = torch.bmm(q_odd, k_odd.transpose(1, 2))
        weights = (weights_even + weights_odd) * 0.5
        denom = torch.clamp_min(weights.sum(dim=-1, keepdim=True), self.eps)
        weights = weights / denom
        attn_output = torch.bmm(weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        attn_output = self.linear_out(attn_output).transpose(0, 1)

        return attn_output

    def left_product_sin_cos2(
            self,
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        query, key, value = self.convert_qkv(query, key, value)
        num_heads = self.h
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        head_dim = embed_dim // num_heads

        # (L, N, E)
        q = self.linear_q(query)
        # (S, N, E)
        k = self.linear_k(key)
        # (S, N, E)
        v = self.linear_v(value)

        # activation
        q = self.kernel(q)
        k = self.kernel(k)

        # multihead reshape
        # (N * h, L, d)
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        # cos transformer
        m = max(src_len, tgt_len)
        # get index and send to cuda
        weight_index = self.get_index2(m).to(q)
        div_term = self.linear_pos(self.get_div_term_index().to(q.device))
        # (N * h, L, 2 * d)
        q_even = torch.cat(
            [q[:, :, 0::2] * torch.sin(
                (weight_index[:, :tgt_len, :] / m).repeat(
                    bsz * num_heads, 1, self.d_k // 2) * div_term) * 0.5,
             q[:, :, 0::2] * torch.cos(
                 (weight_index[:, :tgt_len, :] / m).repeat(
                     bsz * num_heads, 1, self.d_k // 2) * div_term) * -0.5,
             q[:, :, 0::2] * torch.ones(bsz * num_heads, tgt_len, self.d_k // 2, device=q.device) * 0.5
             ], dim=-1)
        # (N * h, S, 2 * d)
        k_even = torch.cat(
            [k[:, :, 0::2] * torch.cos(
                (weight_index[:, :src_len, :] / m).repeat(bsz * num_heads, 1, self.d_k // 2) * div_term),
             k[:, :, 0::2] * torch.sin(
                 (weight_index[:, :src_len, :] / m).repeat(bsz * num_heads, 1, self.d_k // 2) * div_term),
             k[:, :, 0::2] * torch.ones(bsz * num_heads, src_len, self.d_k // 2, device=q.device)
             ], dim=-1)
        # (N * h, L, 2 * d)
        q_odd = torch.cat(
            [q[:, :, 1::2] * torch.sin(
                (weight_index[:, :tgt_len, :] / m).repeat(bsz * num_heads, 1, self.d_k // 2) * div_term),
             q[:, :, 1::2] * torch.cos(
                 (weight_index[:, :tgt_len, :] / m).repeat(bsz * num_heads, 1, self.d_k // 2) * div_term),
             ], dim=-1)
        # (N * h, S, 2 * d)
        k_odd = torch.cat(
            [k[:, :, 1::2] * torch.sin(
                (weight_index[:, :src_len, :] / m).repeat(bsz * num_heads, 1, self.d_k // 2) * div_term),
             k[:, :, 1::2] * torch.cos(
                 (weight_index[:, :src_len, :] / m).repeat(bsz * num_heads, 1, self.d_k // 2) * div_term),
             ], dim=-1)

        weights_even = torch.bmm(q_even, k_even.transpose(1, 2))
        weights_odd = torch.bmm(q_odd, k_odd.transpose(1, 2))

        weights = (weights_even + weights_odd) * 0.5
        denom = torch.clamp_min(weights.sum(dim=-1, keepdim=True), self.eps)
        weights = weights / denom
        attn_output = torch.bmm(weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        attn_output = self.linear_out(attn_output).transpose(0, 1)

        return attn_output

    def left_product_sin_cos3(
            self,
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        query, key, value = self.convert_qkv(query, key, value)
        num_heads = self.h
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        head_dim = embed_dim // num_heads

        # (L, N, E)
        q = self.linear_q(query)
        # (S, N, E)
        k = self.linear_k(key)
        # (S, N, E)
        v = self.linear_v(value)

        # activation
        q = self.kernel(q)
        k = self.kernel(k)

        # multihead reshape
        # (N * h, L, d)
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        # cos transformer
        m = max(src_len, tgt_len)
        # get index and send to cuda
        weight_index = self.get_index(m).to(q)
        q_ee_sin = torch.cat(
            [q[:, 0::2, :] * torch.sin(weight_index[:, :tgt_len, :][:, 0::2, :] / m).repeat(bsz * num_heads, 1,
                                                                                            self.d_k) * 0.5,
             q[:, 0::2, :] * torch.cos(weight_index[:, :tgt_len, :][:, 0::2, :] / m).repeat(bsz * num_heads, 1,
                                                                                            self.d_k) * -1.0 * 0.5,
             q[:, 0::2, :] * torch.ones(bsz * num_heads, tgt_len // 2 + tgt_len % 2, self.d_k, device=q.device) * 0.5
             ], dim=-1)
        k_ee_sin = torch.cat(
            [k[:, 0::2, :] * torch.cos(weight_index[:, :src_len, :][:, 0::2, :] / m).repeat(bsz * num_heads, 1,
                                                                                            self.d_k),
             k[:, 0::2, :] * torch.sin(weight_index[:, :src_len, :][:, 0::2, :] / m).repeat(bsz * num_heads, 1,
                                                                                            self.d_k),
             k[:, 0::2, :] * torch.ones(bsz * num_heads, src_len // 2 + src_len % 2, self.d_k, device=q.device)
             ], dim=-1)

        q_eo_cos = torch.cat(
            [q[:, 0::2, :] * torch.sin(weight_index[:, :tgt_len, :][:, 0::2, :] / m).repeat(bsz * num_heads, 1,
                                                                                            self.d_k),
             q[:, 0::2, :] * torch.cos(weight_index[:, :tgt_len, :][:, 0::2, :] / m).repeat(bsz * num_heads, 1,
                                                                                            self.d_k),
             ], dim=-1)
        k_eo_cos = torch.cat(
            [k[:, 1::2, :] * torch.sin(weight_index[:, :src_len, :][:, 1::2, :] / m).repeat(bsz * num_heads, 1,
                                                                                            self.d_k),
             k[:, 1::2, :] * torch.cos(weight_index[:, :src_len, :][:, 1::2, :] / m).repeat(bsz * num_heads, 1,
                                                                                            self.d_k),
             ], dim=-1)

        q_oe_cos = torch.cat(
            [q[:, 1::2, :] * torch.sin(weight_index[:, :tgt_len, :][:, 1::2, :] / m).repeat(bsz * num_heads, 1,
                                                                                            self.d_k),
             q[:, 1::2, :] * torch.cos(weight_index[:, :tgt_len, :][:, 1::2, :] / m).repeat(bsz * num_heads, 1,
                                                                                            self.d_k),
             ], dim=-1)
        k_oe_cos = torch.cat(
            [k[:, 0::2, :] * torch.sin(weight_index[:, :src_len, :][:, 0::2, :] / m).repeat(bsz * num_heads, 1,
                                                                                            self.d_k),
             k[:, 0::2, :] * torch.cos(weight_index[:, :src_len, :][:, 0::2, :] / m).repeat(bsz * num_heads, 1,
                                                                                            self.d_k),
             ], dim=-1)
        q_oo_sin = torch.cat(
            [q[:, 1::2, :] * torch.sin(weight_index[:, :tgt_len, :][:, 1::2, :] / m).repeat(bsz * num_heads, 1,
                                                                                            self.d_k) * 0.5,
             q[:, 1::2, :] * torch.cos(weight_index[:, :tgt_len, :][:, 1::2, :] / m).repeat(bsz * num_heads, 1,
                                                                                            self.d_k) * -1.0 * 0.5,
             q[:, 1::2, :] * torch.ones(bsz * num_heads, tgt_len // 2, self.d_k, device=q.device) * 0.5
             ], dim=-1)
        k_oo_sin = torch.cat(
            [k[:, 1::2, :] * torch.cos(weight_index[:, :src_len, :][:, 1::2, :] / m).repeat(bsz * num_heads, 1,
                                                                                            self.d_k),
             k[:, 1::2, :] * torch.sin(weight_index[:, :src_len, :][:, 1::2, :] / m).repeat(bsz * num_heads, 1,
                                                                                            self.d_k),
             k[:, 1::2, :] * torch.ones(bsz * num_heads, src_len // 2, self.d_k, device=q.device)
             ], dim=-1)

        qkee_sin = torch.bmm(q_ee_sin, k_ee_sin.transpose(1, 2))
        qkeo_cos = torch.bmm(q_eo_cos, k_eo_cos.transpose(1, 2))
        qkoe_cos = torch.bmm(q_oe_cos, k_oe_cos.transpose(1, 2))
        qkoo_sin = torch.bmm(q_oo_sin, k_oo_sin.transpose(1, 2))

        compensate_coef_h = qkee_sin.size(2) - qkeo_cos.size(2)
        compensate_coef_v = qkee_sin.size(1) - qkoe_cos.size(1)
        cat_h0 = torch.zeros(qkee_sin.size(0), qkee_sin.size(1), compensate_coef_h, device=q.device)
        cat_h1 = torch.zeros(qkoe_cos.size(0), qkoe_cos.size(1), compensate_coef_h, device=q.device)
        qk_up = torch.cat([qkee_sin.unsqueeze(2),
                           torch.cat([qkeo_cos, cat_h0], dim=2).unsqueeze(2),
                           ], dim=-2).transpose(-2, -1).flatten(-2, -1)
        qk_dn = torch.cat([qkoe_cos.unsqueeze(2),
                           torch.cat([qkoo_sin, cat_h1], dim=2).unsqueeze(2),
                           ], dim=-2).transpose(-2, -1).flatten(-2, -1)
        cat_v = torch.zeros(qk_dn.size(0), compensate_coef_v, qk_dn.size(2), device=q.device)
        weights = torch.cat([qk_up.unsqueeze(1),
                             torch.cat([qk_dn, cat_v], dim=1).unsqueeze(1),
                             ], dim=1).transpose(1, 2).flatten(1, 2)
        weights = weights[:, :weights.size(1) - compensate_coef_v, :weights.size(2) - compensate_coef_h]

        denom = torch.clamp_min(weights.sum(dim=-1, keepdim=True), self.eps)
        weights = weights / denom
        attn_output = torch.bmm(weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        attn_output = self.linear_out(attn_output).transpose(0, 1)

        return attn_output

    def right_product(
            self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor, size
                (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor, size
                (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor, size
                (#batch, n_head, time2, d_k).

        """
        query, key, value = self.convert_qkv(query, key, value)
        num_heads = self.h
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        head_dim = embed_dim // num_heads
        # get q, k, v
        # (L, N, E)
        q = self.linear_q(query)
        # (S, N, E)
        k = self.linear_k(key)
        # (S, N, E)
        v = self.linear_v(value)

        q = self.kernel(q)
        k = self.kernel(k)

        # multihead reshape
        assert q is not None and k is not None and v is not None
        # (N * h, L, d)
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        # cos transform
        m = max(src_len, tgt_len)
        # get index and send to cuda
        weight_index = self.get_index(m).to(q)
        # (N * h, L, 2 * d)
        q_ = torch.cat(
            [q * torch.sin(weight_index[:, :tgt_len, :] / m),
             q * torch.cos(weight_index[:, :tgt_len, :] / m)], dim=-1)
        # (N * h, S, 2 * d)
        k_ = torch.cat(
            [k * torch.sin(weight_index[:, :src_len, :] / m),
             k * torch.cos(weight_index[:, :src_len, :] / m)], dim=-1)

        if self.causal:
            ## Need to improve speed!
            # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, L, h, 2 * d, d)
            kv_ = torch.einsum("nld,nlm->nldm", k_, v)
            # (N * h, L, 2 * d, d) -> (N * h, L, 2 * d, d)
            kv_cum = torch.cumsum(kv_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
            qkv = torch.einsum("nld,nldm->nlm", q_, kv_cum)
            # (N * h, L, 2 * d) -> (N * h, L, 2 * d)
            k_cum = torch.cumsum(k_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d) -> (N * h, L)
            denom = torch.clamp_min(torch.einsum("nlm,nlm->nl", q_, k_cum), self.eps)
            # (N * h, L, d) (N * h, L, 1) -> (N * h, L, d)
            attn_output = qkv / denom.unsqueeze(-1)
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        else:
            # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, 2 * d, d)
            kv_ = torch.einsum('nld,nlm->ndm', k_, v)
            # (N * h, L, 2 * d) (N * h, 2 * d) -> (N * h, L)
            z_ = 1 / torch.clamp_min(torch.einsum('nld,nd->nl', q_, torch.sum(k_, dim=1)), self.eps)
            # (N * h, L, 2 * d) (N * h, d, 2 * d) (N * h, L) -> (N * h, L, d)
            attn_output = torch.einsum('nld,ndm,nl->nlm', q_, kv_, z_)
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        # L, N, E
        attn_output = self.linear_out(attn_output).transpose(0, 1)

        return attn_output

    def right_product_sin(
            self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor, size
                (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor, size
                (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor, size
                (#batch, n_head, time2, d_k).

        """
        query, key, value = self.convert_qkv(query, key, value)
        num_heads = self.h
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        head_dim = embed_dim // num_heads
        # get q, k, v
        # (L, N, E)
        q = self.linear_q(query)
        # (S, N, E)
        k = self.linear_k(key)
        # (S, N, E)
        v = self.linear_v(value)

        q = self.kernel(q)
        k = self.kernel(k)

        # multihead reshape
        assert q is not None and k is not None and v is not None
        # (N * h, L, d)
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        # cos transform
        m = max(src_len, tgt_len)
        # get index and send to cuda
        weight_index = self.get_index(m).to(q)
        # (N * h, L, 2 * d)
        q_ = torch.cat(
            [
                q * torch.sin(weight_index[:, :tgt_len, :] / m) * 0.5,
                q * torch.cos(weight_index[:, :tgt_len, :] / m) * -0.5,
                q * torch.ones(bsz * num_heads, tgt_len, self.d_k, device=q.device) * 0.5
            ], dim=-1)
        # (N * h, S, 2 * d)
        k_ = torch.cat(
            [
                k * torch.cos(weight_index[:, :src_len, :] / m),
                k * torch.sin(weight_index[:, :src_len, :] / m),
                k * torch.ones(bsz * num_heads, src_len, self.d_k, device=q.device)
            ], dim=-1)

        # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, 2 * d, d)
        kv_ = torch.einsum('nld,nlm->ndm', k_, v)
        # (N * h, L, 2 * d) (N * h, 2 * d) -> (N * h, L)
        z_ = 1 / torch.clamp_min(torch.einsum('nld,nd->nl', q_, torch.sum(k_, dim=1)), self.eps)
        # (N * h, L, 2 * d) (N * h, d, 2 * d) (N * h, L) -> (N * h, L, d)
        attn_output = torch.einsum('nld,ndm,nl->nlm', q_, kv_, z_)
        # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        # L, N, E
        attn_output = self.linear_out(attn_output).transpose(0, 1)

        return attn_output

    def right_product_sin_cos(
            self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
            mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        query, key, value = self.convert_qkv(query, key, value)
        num_heads = self.h
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        head_dim = embed_dim // num_heads

        # get q, k, v
        # (L, N, E)
        q = self.linear_q(query)
        # (S, N, E)
        k = self.linear_k(key)
        # (S, N, E)
        v = self.linear_v(value)

        # activation
        q = self.kernel(q)
        k = self.kernel(k)

        # multihead reshape
        # (N * h, L, d)
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        # cos transform
        m = max(src_len, tgt_len)
        # get index and send to cuda
        weight_index = self.get_index(m).to(q)
        # (N * h, L, 2 * d)
        q_even = torch.cat(
            [q[:, :, 0::2] * torch.sin(weight_index[:, :tgt_len, :] / m) * 0.5,
             q[:, :, 0::2] * torch.cos(weight_index[:, :tgt_len, :] / m) * -1.0 * 0.5,
             q[:, :, 0::2] * torch.ones(bsz * num_heads, tgt_len, 1, device=q.device) * 0.5,
             ], dim=-1)
        # (N * h, S, 2 * d)
        k_even = torch.cat(
            [k[:, :, 0::2] * torch.cos(weight_index[:, :src_len, :] / m),
             k[:, :, 0::2] * torch.sin(weight_index[:, :src_len, :] / m),
             k[:, :, 0::2] * torch.ones(bsz * num_heads, src_len, 1, device=q.device),
             ], dim=-1)
        # (N * h, L, 2 * d)
        q_odd = torch.cat(
            [q[:, :, 1::2] * torch.sin(weight_index[:, :tgt_len, :] / m),
             q[:, :, 1::2] * torch.cos(weight_index[:, :tgt_len, :] / m),
             torch.zeros(bsz * num_heads, tgt_len, self.d_k // 2, device=q.device),
             ], dim=-1)
        # (N * h, S, 2 * d)
        k_odd = torch.cat(
            [k[:, :, 1::2] * torch.sin(weight_index[:, :src_len, :] / m),
             k[:, :, 1::2] * torch.cos(weight_index[:, :src_len, :] / m),
             torch.zeros(bsz * num_heads, src_len, self.d_k // 2, device=q.device),
             ], dim=-1)
        # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, 2 * d, d)
        # q_ = torch.zeros(q_even.size(0), q_even.size(1), q_even.size(2) * 3, device=q.device)
        # q_[:, :, 0::3] = q_even
        # q_[:, :, 1::3] = q_odd
        # k_ = torch.zeros(k_even.size(0), k_even.size(1), k_even.size(2) * 3, device=q.device)
        # k_[:, :, 0::3] = k_even
        # k_[:, :, 1::3] = k_odd
        q_ = torch.cat([q_even.unsqueeze(2),
                        q_odd.unsqueeze(2),
                        torch.zeros_like(q_even).unsqueeze(2)
                        ], dim=-2).transpose(-2, -1).flatten(-2, -1)
        k_ = torch.cat([k_even.unsqueeze(2),
                        k_odd.unsqueeze(2),
                        torch.zeros_like(k_even).unsqueeze(2)
                        ], dim=-2).transpose(-2, -1).flatten(-2, -1)
        kv_ = torch.einsum('nld,nlm->ndm', k_, v)
        z_ = torch.clamp_min(2 / (torch.einsum('nld,nd->nl', q_, torch.sum(k_, dim=1))), self.eps)
        attn_output = torch.einsum('nld,ndm->nlm', q_, kv_) * 0.5
        attn_output = torch.einsum('nlm,nl->nlm', attn_output, z_)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        attn_output = self.linear_out(attn_output).transpose(0, 1)

        return attn_output

    def right_product_sin_cos2(
            self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
            mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        query, key, value = self.convert_qkv(query, key, value)
        num_heads = self.h
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        head_dim = embed_dim // num_heads

        # (L, N, E)
        q = self.linear_q(query)
        # (S, N, E)
        k = self.linear_k(key)
        # (S, N, E)
        v = self.linear_v(value)

        # activation
        q = self.kernel(q)
        k = self.kernel(k)

        # multihead reshape
        # (N * h, L, d)
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        # cos transform
        m = max(src_len, tgt_len)
        # get index and send to cuda
        weight_index = self.get_index2(m).to(q)
        div_term = self.linear_pos(self.get_div_term_index().to(q.device))
        # (N * h, L, 2 * d)
        q_even = torch.cat(
            [q[:, :, 0::2] * torch.sin((weight_index[:, :tgt_len, :] / m) * div_term) * 0.5,
             q[:, :, 0::2] * torch.cos((weight_index[:, :tgt_len, :] / m) * div_term) * -1.0 * 0.5,
             q[:, :, 0::2] * torch.ones(bsz * num_heads, tgt_len, 1, device=q.device) * 0.5
             ], dim=-1)
        # (N * h, S, 2 * d)
        k_even = torch.cat(
            [k[:, :, 0::2] * torch.cos((weight_index[:, :src_len, :] / m) * div_term),
             k[:, :, 0::2] * torch.sin((weight_index[:, :src_len, :] / m) * div_term),
             k[:, :, 0::2] * torch.ones(bsz * num_heads, src_len, 1, device=q.device)
             ], dim=-1)
        # (N * h, L, 2 * d)
        q_odd = torch.cat(
            [q[:, :, 1::2] * torch.sin((weight_index[:, :tgt_len, :] / m) * div_term),
             q[:, :, 1::2] * torch.cos((weight_index[:, :tgt_len, :] / m) * div_term),
             torch.zeros(bsz * num_heads, tgt_len, self.d_k // 2, device=q.device)
             ], dim=-1)
        # (N * h, S, 2 * d)
        k_odd = torch.cat(
            [k[:, :, 1::2] * torch.sin((weight_index[:, :src_len, :] / m) * div_term),
             k[:, :, 1::2] * torch.cos((weight_index[:, :src_len, :] / m) * div_term),
             torch.zeros(bsz * num_heads, src_len, self.d_k // 2, device=q.device)
             ], dim=-1)
        # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, 2 * d, d)
        q_ = torch.cat([q_even.unsqueeze(2),
                        q_odd.unsqueeze(2),
                        torch.zeros_like(q_even).unsqueeze(2)
                        ], dim=-2).transpose(-2, -1).flatten(-2, -1)
        k_ = torch.cat([k_even.unsqueeze(2),
                        k_odd.unsqueeze(2),
                        torch.zeros_like(k_even).unsqueeze(2)
                        ], dim=-2).transpose(-2, -1).flatten(-2, -1)
        kv_ = torch.einsum('nld,nlm->ndm', k_, v)
        z_ = torch.clamp_min(2 / (torch.einsum('nld,nd->nl', q_, torch.sum(k_, dim=1))), self.eps)
        attn_output = torch.einsum('nld,ndm->nlm', q_, kv_) * 0.5
        attn_output = torch.einsum('nlm,nl->nlm', attn_output, z_)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        attn_output = self.linear_out(attn_output).transpose(0, 1)

        return attn_output

    def right_product_sin_cos3(
            self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
            mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        query, key, value = self.convert_qkv(query, key, value)
        num_heads = self.h
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        head_dim = embed_dim // num_heads

        # (L, N, E)
        q = self.linear_q(query)
        # (S, N, E)
        k = self.linear_k(key)
        # (S, N, E)
        v = self.linear_v(value)

        # activation
        q = self.kernel(q)
        k = self.kernel(k)

        # multihead reshape
        # (N * h, L, d)
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        # cos transform
        m = max(src_len, tgt_len)
        # get index and send to cuda
        weight_index = self.get_index(m).to(q)

        q_ee_sin = torch.cat(
            [q[:, 0::2, :] * torch.sin(weight_index[:, :tgt_len, :][:, 0::2, :] / m).repeat(bsz * num_heads, 1,
                                                                                            self.d_k) * 0.5,
             q[:, 0::2, :] * torch.cos(weight_index[:, :tgt_len, :][:, 0::2, :] / m).repeat(bsz * num_heads, 1,
                                                                                            self.d_k) * -1.0 * 0.5,
             q[:, 0::2, :] * torch.ones(bsz * num_heads, tgt_len // 2 + tgt_len % 2, self.d_k, device=q.device) * 0.5
             ], dim=-1)
        k_ee_sin = torch.cat(
            [k[:, 0::2, :] * torch.cos(weight_index[:, :src_len, :][:, 0::2, :] / m).repeat(bsz * num_heads, 1,
                                                                                            self.d_k),
             k[:, 0::2, :] * torch.sin(weight_index[:, :src_len, :][:, 0::2, :] / m).repeat(bsz * num_heads, 1,
                                                                                            self.d_k),
             k[:, 0::2, :] * torch.ones(bsz * num_heads, src_len // 2 + src_len % 2, self.d_k, device=q.device)
             ], dim=-1)

        q_eo_cos = torch.cat(
            [q[:, 0::2, :] * torch.sin(weight_index[:, :tgt_len, :][:, 0::2, :] / m).repeat(bsz * num_heads, 1,
                                                                                            self.d_k),
             q[:, 0::2, :] * torch.cos(weight_index[:, :tgt_len, :][:, 0::2, :] / m).repeat(bsz * num_heads, 1,
                                                                                            self.d_k),
             ], dim=-1)
        k_eo_cos = torch.cat(
            [k[:, 1::2, :] * torch.sin(weight_index[:, :src_len, :][:, 1::2, :] / m).repeat(bsz * num_heads, 1,
                                                                                            self.d_k),
             k[:, 1::2, :] * torch.cos(weight_index[:, :src_len, :][:, 1::2, :] / m).repeat(bsz * num_heads, 1,
                                                                                            self.d_k),
             ], dim=-1)

        q_oe_cos = torch.cat(
            [q[:, 1::2, :] * torch.sin(weight_index[:, :tgt_len, :][:, 1::2, :] / m).repeat(bsz * num_heads, 1,
                                                                                            self.d_k),
             q[:, 1::2, :] * torch.cos(weight_index[:, :tgt_len, :][:, 1::2, :] / m).repeat(bsz * num_heads, 1,
                                                                                            self.d_k),
             ], dim=-1)
        k_oe_cos = torch.cat(
            [k[:, 0::2, :] * torch.sin(weight_index[:, :src_len, :][:, 0::2, :] / m).repeat(bsz * num_heads, 1,
                                                                                            self.d_k),
             k[:, 0::2, :] * torch.cos(weight_index[:, :src_len, :][:, 0::2, :] / m).repeat(bsz * num_heads, 1,
                                                                                            self.d_k),
             ], dim=-1)
        q_oo_sin = torch.cat(
            [q[:, 1::2, :] * torch.sin(weight_index[:, :tgt_len, :][:, 1::2, :] / m).repeat(bsz * num_heads, 1,
                                                                                            self.d_k) * 0.5,
             q[:, 1::2, :] * torch.cos(weight_index[:, :tgt_len, :][:, 1::2, :] / m).repeat(bsz * num_heads, 1,
                                                                                            self.d_k) * -1.0 * 0.5,
             q[:, 1::2, :] * torch.ones(bsz * num_heads, tgt_len // 2, self.d_k, device=q.device) * 0.5
             ], dim=-1)
        k_oo_sin = torch.cat(
            [k[:, 1::2, :] * torch.cos(weight_index[:, :src_len, :][:, 1::2, :] / m).repeat(bsz * num_heads, 1,
                                                                                            self.d_k),
             k[:, 1::2, :] * torch.sin(weight_index[:, :src_len, :][:, 1::2, :] / m).repeat(bsz * num_heads, 1,
                                                                                            self.d_k),
             k[:, 1::2, :] * torch.ones(bsz * num_heads, src_len // 2, self.d_k, device=q.device)
             ], dim=-1)

        kv_ee_sin = torch.einsum('nld,nlm->ndm', k_ee_sin, v[:, 0::2, :])
        kv_eo_cos = torch.einsum('nld,nlm->ndm', k_eo_cos, v[:, 1::2, :])
        kv_oe_cos = torch.einsum('nld,nlm->ndm', k_oe_cos, v[:, 0::2, :])
        kv_oo_sin = torch.einsum('nld,nlm->ndm', k_oo_sin, v[:, 1::2, :])

        attn_ee_sin = torch.einsum('nld,ndm->nlm', q_ee_sin, kv_ee_sin)
        attn_eo_cos = torch.einsum('nld,ndm->nlm', q_eo_cos, kv_eo_cos)
        attn_oe_cos = torch.einsum('nld,ndm->nlm', q_oe_cos, kv_oe_cos)
        attn_oo_sin = torch.einsum('nld,ndm->nlm', q_oo_sin, kv_oo_sin)

        attn_up = attn_ee_sin + attn_eo_cos
        attn_dn = attn_oe_cos + attn_oo_sin
        compensate_h = attn_up.size(1) - attn_dn.size(1)
        cat_h = torch.zeros(attn_dn.size(0), compensate_h, attn_dn.size(2), device=q.device)
        attn_output = torch.cat([attn_up.unsqueeze(1),
                                 torch.cat([attn_dn, cat_h], dim=1).unsqueeze(1)
                                 ], dim=1).transpose(1, 2).flatten(1, 2)
        attn_output = attn_output[:, :attn_output.size(1) - compensate_h, :]

        z_ee_sin = torch.einsum('nld,nd->nl', q_ee_sin, torch.sum(k_ee_sin, dim=1))
        z_eo_cos = torch.einsum('nld,nd->nl', q_eo_cos, torch.sum(k_eo_cos, dim=1))
        z_oe_cos = torch.einsum('nld,nd->nl', q_oe_cos, torch.sum(k_oe_cos, dim=1))
        z_oo_sin = torch.einsum('nld,nd->nl', q_oo_sin, torch.sum(k_oo_sin, dim=1))
        z_up = z_ee_sin + z_eo_cos
        z_dn = z_oe_cos + z_oo_sin
        z_cat_h = torch.zeros(z_up.size(0), compensate_h, device=q.device)
        z_ = torch.cat([z_up.unsqueeze(1),
                        torch.cat([z_dn, z_cat_h], dim=1).unsqueeze(1)], dim=1).transpose(1, 2).flatten(1, 2)
        z_ = 1 / (z_[:, :z_.size(1) - compensate_h])
        attn_output = torch.einsum('nlm,nl->nlm', attn_output, z_)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        attn_output = self.linear_out(attn_output).transpose(0, 1)

        return attn_output

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                pos_emb: torch.Tensor = torch.empty(0)) -> torch.Tensor:
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
                1.When applying cross attention between decoder and encoder,
                the batch padding mask for input is in (#batch, 1, T) shape.
                2.When applying self attention of encoder,
                the mask is in (#batch, T, T)  shape.
                3.When applying self attention of decoder,
                the mask is in (#batch, L, L)  shape.
                4.If the different position in decoder see different block
                of the encoder, such as Mocha, the passed in mask could be
                in (#batch, L, T) shape. But there is no such case in current
                Wenet.


        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        if self.deploy:
            if self.cosformer_version == 1:
                x = self.right_product_sin(query, key, value, mask)
            elif self.cosformer_version == 2:
                x = self.right_product_sin_cos(query, key, value, mask)
            elif self.cosformer_version == 3:
                x = self.right_product_sin_cos2(query, key, value, mask)
            elif self.cosformer_version == 4:
                x = self.right_product_sin_cos3(query, key, value, mask)
            else:
                x = self.right_product(query, key, value, mask)
        else:
            if self.cosformer_version == 1:
                x = self.left_product_sin(query, key, value, mask)
            elif self.cosformer_version == 2:
                x = self.left_product_sin_cos(query, key, value, mask)
            elif self.cosformer_version == 3:
                x = self.left_product_sin_cos2(query, key, value, mask)
            elif self.cosformer_version == 4:
                x = self.left_product_sin_cos3(query, key, value, mask)
            else:
                x = self.left_product(query, key, value, mask)
        return x


class Cos_RelPositionMultiHeadedAttention_rel_shift(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self, n_head, n_feat, dropout_rate,
                 deploy: bool = False, act_fun: str = 'relu',
                 causal: bool = False, cosformer_version: int = 0):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate)
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)
        self.act_fun = act_fun
        self.deploy = deploy
        self.cosformer_version = cosformer_version
        self.eps = float(1e-6)
        if cosformer_version == 4 or cosformer_version == 5:
            self.max_pos = 5000
            self.random_index = nn.Parameter(
                np.pi / 2 * torch.rand(1, self.max_pos, self.d_k, requires_grad=True))
            if cosformer_version == 5:
                self.limit = 1.3

    def get_index(self, seq_len: int):
        index = np.pi / 2 * torch.arange(1, seq_len + 1).reshape(1, -1, 1)
        return index

    def get_div_term(self, hidden_dim: int):
        return torch.arange(1, hidden_dim + 1).reshape(1, 1, -1) / hidden_dim

    @torch.jit.export
    def kernel(self, x: torch.Tensor) -> torch.Tensor:
        if self.act_fun == "relu":
            return F.relu(x)
        elif self.act_fun == "elu":
            return 1 + F.elu(x)
        elif self.act_fun == "sigmoid":
            return F.sigmoid(x)
        elif self.act_fun == "tanh":
            return 0.5 + 0.5 * F.tanh(x)
        else:
            # no kernel
            return x

    def rel_shift(self, x, zero_triu: bool = False):
        """Compute relative positinal encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, size). (batch, head, time1, time2)
            zero_triu (bool): If true, return the lower triangular part of
                the matrix.
        Returns:
            torch.Tensor: Output tensor.
        """

        zero_pad = torch.zeros((x.size()[0], x.size()[1], x.size()[2], 1),
                               device=x.device,
                               dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(x.size()[0],
                                 x.size()[1],
                                 x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)[
            :, :, :, : x.size(-1) // 2 + 1
            ]  # only keep the positions from 0 to time2

        return x

    def rel_shift2(self, x):
        """Compute relative positinal encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, size). (batch, head, time1, time2)
            zero_triu (bool): If true, return the lower triangular part of
                the matrix.
        Returns:
            torch.Tensor: Output tensor.
        """

        zero_pad = torch.zeros((x.size(0), x.size(1), 1),
                               device=x.device,
                               dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(x.size()[0],
                                 x.size(2) + 1, x.size(1))
        x = x_padded[:, 1:].view_as(x)[
            :, :, : x.size(-1) // 2 + 1
            ]  # only keep the positions from 0 to time2

        return x

    def rel_shift3(self, x, q_len):
        bsz, ktl, ktw = x.size()
        x = x.transpose(-2, -1).flatten(-2, -1)
        zero_pad = torch.zeros((x.size(0), 1),
                               device=x.device,
                               dtype=x.dtype)
        # print(x.size())
        x_padded = torch.cat([zero_pad, x], dim=-1).repeat(1, q_len)[:, q_len:]
        # print(x_padded.size())
        x_padded = x_padded.contiguous().view(x.size(0), -1, ktl)[:, :, :((ktl + 1) // 2)]
        x_padded = x_padded.contiguous().view(x_padded.size(0), q_len, -1, x_padded.size(-1))
        return x_padded

    def rel_shift4(self, x, q_len):
        bsz, ktl, ktw = x.size()  # [2, 5, 2]
        x = x.transpose(-2, -1)
        rel_len = (ktl + 1) // 2
        x_slice = torch.cat([x[:, :, (i - rel_len + 1):i + 1].unsqueeze(1)
                             for i in range(rel_len + q_len - 2, rel_len - 2, -1)], dim=1)
        return x_slice

    def rescale_pos_emb(self, x: torch.Tensor):
        return 0.5 * x + 0.5

    def left_product(self,
                     query: torch.Tensor,
                     key: torch.Tensor,
                     value: torch.Tensor,
                     mask: Optional[torch.Tensor],
                     pos_emb: torch.Tensor,
                     eps: float = 1e-6,
                     ) -> torch.Tensor:
        tgt_len, src_len, n_batch = query.size(1), key.size(1), query.size(0)
        m = max(src_len, tgt_len)
        weight_index = self.get_index(m).to(query.device)

        q = self.linear_q(query.transpose(0, 1))
        k = self.linear_k(key.transpose(0, 1))
        v = self.linear_v(value.transpose(0, 1))
        p = self.linear_pos(pos_emb.transpose(0, 1).repeat(1, n_batch, 1))

        # apply kernel
        k = self.kernel(k)
        p = self.kernel(p)
        q_with_bias_u = self.kernel((q + self.pos_bias_u.view(-1)))
        q_with_bias_v = self.kernel((q + self.pos_bias_v.view(-1)))

        # QK1
        q_with_bias_u = q_with_bias_u.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        k = k.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        # QK2
        q_with_bias_v = q_with_bias_v.view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        p = p.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        # V
        v = v.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)

        # (N * h, L, 2 * d)
        q_with_bias_u = torch.cat(
            [q_with_bias_u * torch.sin(weight_index[:, :tgt_len, :] / m),
             q_with_bias_u * torch.cos(weight_index[:, :tgt_len, :] / m)], dim=-1)
        # (N * h, S, 2 * d)
        k = torch.cat(
            [k * torch.sin(weight_index[:, :src_len, :] / m),
             k * torch.cos(weight_index[:, :src_len, :] / m)], dim=-1)

        matrix_ac = torch.bmm(q_with_bias_u, k.transpose(1, 2))
        denom_ac = torch.clamp_min(matrix_ac.sum(dim=-1, keepdim=True), eps)
        matrix_ac /= denom_ac

        # compute matrix b and matrix d
        q_with_bias_v = torch.cat(
            [q_with_bias_v * torch.sin(weight_index[:, :tgt_len, :] / m),
             q_with_bias_v * torch.cos(weight_index[:, :tgt_len, :] / m)], dim=-1)
        p = torch.cat(
            [p * torch.sin(weight_index[:, :src_len, :] / m),
             p * torch.cos(weight_index[:, :src_len, :] / m)], dim=-1)
        matrix_bd = torch.bmm(q_with_bias_v, p.transpose(1, 2))
        denom_bd = torch.clamp_min(matrix_bd.sum(dim=-1, keepdim=True), eps)
        matrix_bd /= denom_bd
        scores = (matrix_ac + matrix_bd) / math.sqrt(
            self.d_k)
        attn = torch.bmm(scores, v)
        x = attn.transpose(0, 1).contiguous().view(tgt_len, n_batch, -1)
        return self.linear_out(x).transpose(0, 1)

    def left_product_rel_shift(self,
                               query: torch.Tensor,
                               key: torch.Tensor,
                               value: torch.Tensor,
                               mask: Optional[torch.Tensor],
                               pos_emb: torch.Tensor,
                               eps: float = 1e-6,
                               ) -> torch.Tensor:
        tgt_len, src_len, n_batch = query.size(1), key.size(1), query.size(0)
        m_ac = max(src_len, tgt_len)
        m_bd = max(src_len * 2 - 1, tgt_len)
        weight_index_ac = self.get_index(m_ac).to(query.device)
        weight_index_bd = self.get_index(m_bd).to(query.device)

        q = self.linear_q(query.transpose(0, 1))
        k = self.linear_k(key.transpose(0, 1))
        v = self.linear_v(value.transpose(0, 1))
        p = self.linear_pos(pos_emb.transpose(0, 1).repeat(1, n_batch, 1))

        # apply kernel
        k = self.kernel(k)
        p = self.kernel(p)
        q_with_bias_u = self.kernel((q + self.pos_bias_u.view(-1)))
        q_with_bias_v = self.kernel((q + self.pos_bias_v.view(-1)))

        # QK1
        q_with_bias_u = q_with_bias_u.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        k = k.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        # QK2
        q_with_bias_v = q_with_bias_v.view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        p = p.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        # V
        v = v.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)

        q_with_bias_u = torch.cat(
            [q_with_bias_u * torch.sin(weight_index_ac[:, :tgt_len, :] / m_ac),
             q_with_bias_u * torch.cos(weight_index_ac[:, :tgt_len, :] / m_ac)], dim=-1)
        k = torch.cat(
            [k * torch.sin(weight_index_ac[:, :src_len, :] / m_ac),
             k * torch.cos(weight_index_ac[:, :src_len, :] / m_ac)], dim=-1)

        matrix_ac = torch.bmm(q_with_bias_u, k.transpose(-2, -1))
        denom_ac = torch.clamp_min(matrix_ac.sum(dim=-1, keepdim=True), eps)
        matrix_ac /= denom_ac

        # compute matrix b and matrix d
        q_with_bias_v = torch.cat(
            [q_with_bias_v * torch.sin(weight_index_bd[:, :tgt_len, :] / m_bd),
             q_with_bias_v * torch.cos(weight_index_bd[:, :tgt_len, :] / m_bd)], dim=-1)
        p = torch.cat(
            [p * torch.sin(weight_index_bd[:, :src_len * 2 - 1, :] / m_bd),
             p * torch.cos(weight_index_bd[:, :src_len * 2 - 1, :] / m_bd)], dim=-1)
        matrix_bd = torch.bmm(q_with_bias_v, p.transpose(-2, -1))
        denom_bd = torch.clamp_min(matrix_bd.sum(dim=-1, keepdim=True), eps)
        matrix_bd /= denom_bd
        matrix_bd = self.rel_shift2(matrix_bd)
        scores = (matrix_ac + matrix_bd) / math.sqrt(
            self.d_k)  # (batch, head, time1, time2)
        attn = torch.bmm(scores, v)
        x = attn.transpose(0, 1).contiguous().view(tgt_len, n_batch, -1)
        return self.linear_out(x).transpose(0, 1)

    def left_product_rel_shift_partial(self,
                                       query: torch.Tensor,
                                       key: torch.Tensor,
                                       value: torch.Tensor,
                                       mask: Optional[torch.Tensor],
                                       pos_emb: torch.Tensor,
                                       eps: float = 1e-6,
                                       ) -> torch.Tensor:
        tgt_len, src_len, n_batch = query.size(1), key.size(1), query.size(0)
        m_ac = max(src_len, tgt_len)
        m_bd = max(src_len * 2 - 1, tgt_len)
        weight_index_ac = self.get_index(m_ac).to(query.device)
        weight_index_bd = self.get_index(m_bd).to(query.device)

        q = self.linear_q(query.transpose(0, 1))
        k = self.linear_k(key.transpose(0, 1))
        v = self.linear_v(value.transpose(0, 1))
        p = self.linear_pos(pos_emb.transpose(0, 1).repeat(1, n_batch, 1))

        # apply kernel
        k = self.kernel(k)
        q_with_bias_u = self.kernel((q + self.pos_bias_u.view(-1)))
        q_with_bias_v = q + self.pos_bias_v.view(-1)

        # QK1
        q_with_bias_u = q_with_bias_u.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        k = k.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        # QK2
        q_with_bias_v = q_with_bias_v.view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        p = p.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        # V
        v = v.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)

        q_with_bias_u = torch.cat(
            [q_with_bias_u * torch.sin(weight_index_ac[:, :tgt_len, :] / m_ac),
             q_with_bias_u * torch.cos(weight_index_ac[:, :tgt_len, :] / m_ac)], dim=-1)
        k = torch.cat(
            [k * torch.sin(weight_index_ac[:, :src_len, :] / m_ac),
             k * torch.cos(weight_index_ac[:, :src_len, :] / m_ac)], dim=-1)

        matrix_ac = torch.bmm(q_with_bias_u, k.transpose(-2, -1))
        denom_ac = torch.clamp_min(matrix_ac.sum(dim=-1, keepdim=True), eps)
        matrix_ac /= denom_ac

        # compute matrix b and matrix d
        matrix_bd = torch.bmm(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = torch.softmax(matrix_bd, dim=-1)
        matrix_bd = self.rel_shift2(matrix_bd)
        scores = (matrix_ac + matrix_bd) / math.sqrt(
            self.d_k)  # (batch, head, time1, time2)
        attn = torch.bmm(scores, v)
        x = attn.transpose(0, 1).contiguous().view(tgt_len, n_batch, -1)
        return self.linear_out(x).transpose(0, 1)

    def left_product_fixed_rpe(self,
                               query: torch.Tensor,
                               key: torch.Tensor,
                               value: torch.Tensor,
                               mask: Optional[torch.Tensor],
                               pos_emb: torch.Tensor,
                               eps: float = 1e-6,
                               ) -> torch.Tensor:
        tgt_len, src_len, n_batch = query.size(1), key.size(1), query.size(0)
        m = max(src_len, tgt_len)
        weight_index = self.get_index(m).to(query.device)

        q = self.linear_q(query.transpose(0, 1))
        k = self.linear_k(key.transpose(0, 1))
        v = self.linear_v(value.transpose(0, 1))

        # apply kernel
        k = self.kernel(k)
        p = self.kernel(pos_emb)
        q_with_bias_u = self.kernel((q + self.pos_bias_u.view(-1)))
        q_with_bias_v = self.kernel((q + self.pos_bias_v.view(-1)))

        # QK1
        q_with_bias_u = q_with_bias_u.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        k = k.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        # QK2
        q_with_bias_v = q_with_bias_v.view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        # V
        v = v.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)

        # (N * h, L, 2 * d)
        q_with_bias_u = torch.cat(
            [q_with_bias_u * torch.sin(weight_index[:, :tgt_len, :] / m),
             q_with_bias_u * torch.cos(weight_index[:, :tgt_len, :] / m)], dim=-1)
        # (N * h, S, 2 * d)
        k = torch.cat(
            [k * torch.sin(weight_index[:, :src_len, :] / m),
             k * torch.cos(weight_index[:, :src_len, :] / m)], dim=-1)

        matrix_ac = torch.bmm(q_with_bias_u, k.transpose(1, 2))
        denom_ac = torch.clamp_min(matrix_ac.sum(dim=-1, keepdim=True), eps)
        matrix_ac /= denom_ac  # [4, 128, 256]

        # compute matrix b and matrix d
        q_with_bias_v = torch.cat(
            [q_with_bias_v * torch.sin(weight_index[:, :tgt_len, :] / m),
             q_with_bias_v * torch.cos(weight_index[:, :tgt_len, :] / m)], dim=-1)
        # print('q_with_bias_v', q_with_bias_v.size())  # (B*H, L, d)
        p = torch.cat(
            [p * torch.sin(weight_index[:, :src_len, :] / m),
             p * torch.cos(weight_index[:, :src_len, :] / m)], dim=-1)

        matrix_bd = torch.einsum('bld,lsd->bls', q_with_bias_v, p)
        denom_bd = torch.clamp_min(matrix_bd.sum(dim=-1, keepdim=True), eps)
        matrix_bd /= denom_bd

        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)
        attn = torch.bmm(scores, v)  # [4, 128, 16]
        x = attn.transpose(0, 1).contiguous().view(tgt_len, n_batch, -1)
        return self.linear_out(x).transpose(0, 1)

    def left_product_new_rpe(self,
                             query: torch.Tensor,
                             key: torch.Tensor,
                             value: torch.Tensor,
                             mask: Optional[torch.Tensor],
                             pos_emb: torch.Tensor,
                             eps: float = 1e-6,
                             ) -> torch.Tensor:
        tgt_len, src_len, n_batch = query.size(1), key.size(1), query.size(0)
        weight_index = self.random_index.to(query.device)

        q = self.linear_q(query.transpose(0, 1))
        k = self.linear_k(key.transpose(0, 1))
        v = self.linear_v(value.transpose(0, 1))
        p = self.linear_pos(self.rescale_pos_emb(pos_emb).transpose(0, 1).repeat(1, n_batch, 1))

        # apply kernel
        k = self.kernel(k)
        p = self.kernel(p)
        q_with_bias_u = self.kernel((q + self.pos_bias_u.view(-1)))
        q_with_bias_v = self.kernel((q + self.pos_bias_v.view(-1)))

        # QK1
        q_with_bias_u = q_with_bias_u.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        k = k.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        # QK2
        q_with_bias_v = q_with_bias_v.view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        p = p.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        # V
        v = v.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)

        # (N * h, L, 2 * d)
        q_with_bias_u = torch.cat(
            [q_with_bias_u * torch.sin(weight_index[:, :tgt_len, :]),
             q_with_bias_u * torch.cos(weight_index[:, :tgt_len, :])], dim=-1)
        # (N * h, S, 2 * d)
        k = torch.cat(
            [k * torch.sin(weight_index[:, :src_len, :]),
             k * torch.cos(weight_index[:, :src_len, :])], dim=-1)

        matrix_ac = torch.bmm(q_with_bias_u, k.transpose(1, 2))
        denom_ac = torch.clamp_min(matrix_ac.sum(dim=-1, keepdim=True), eps)
        matrix_ac /= denom_ac

        # compute matrix b and matrix d
        q_with_bias_v = torch.cat(
            [q_with_bias_v * torch.sin(p[:, :tgt_len, :]),
             q_with_bias_v * torch.cos(p[:, :tgt_len, :])], dim=-1)
        p = torch.cat(
            [torch.sin(p[:, :src_len, :]),
             torch.cos(p[:, :src_len, :])], dim=-1)
        matrix_bd = torch.bmm(q_with_bias_v, p.transpose(1, 2))
        denom_bd = torch.clamp_min(matrix_bd.sum(dim=-1, keepdim=True), eps)
        matrix_bd /= denom_bd
        scores = (matrix_ac + matrix_bd) / math.sqrt(
            self.d_k)
        attn = torch.bmm(scores, v)
        x = attn.transpose(0, 1).contiguous().view(tgt_len, n_batch, -1)
        return self.linear_out(x).transpose(0, 1)

    def left_product_new_rpe2(self,
                              query: torch.Tensor,
                              key: torch.Tensor,
                              value: torch.Tensor,
                              mask: Optional[torch.Tensor],
                              pos_emb: torch.Tensor,
                              eps: float = 1e-6,
                              ) -> torch.Tensor:
        tgt_len, src_len, n_batch = query.size(1), key.size(1), query.size(0)
        weight_index = self.random_index.to(query.device)

        q = self.linear_q(query.transpose(0, 1))
        k = self.linear_k(key.transpose(0, 1))
        v = self.linear_v(value.transpose(0, 1))
        p = self.linear_pos(pos_emb.transpose(0, 1).repeat(1, n_batch, 1))

        # apply kernel
        k = self.kernel(k)
        q_with_bias_u = self.kernel((q + self.pos_bias_u.view(-1)))
        q_with_bias_v = self.kernel((q + self.pos_bias_v.view(-1)))

        # QK1
        q_with_bias_u = q_with_bias_u.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        k = k.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        # QK2
        q_with_bias_v = q_with_bias_v.view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        p = p.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        # V
        v = v.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)

        # (N * h, L, 2 * d)
        q_with_bias_u = torch.cat(
            [q_with_bias_u * torch.sin(weight_index[:, :tgt_len, :]),
             q_with_bias_u * torch.cos(weight_index[:, :tgt_len, :])], dim=-1)
        # (N * h, S, 2 * d)
        k = torch.cat(
            [k * torch.sin(weight_index[:, :src_len, :]),
             k * torch.cos(weight_index[:, :src_len, :])], dim=-1)

        matrix_ac = torch.bmm(q_with_bias_u, k.transpose(1, 2))
        denom_ac = torch.clamp_min(matrix_ac.sum(dim=-1, keepdim=True), eps)
        matrix_ac /= denom_ac

        # compute matrix b and matrix d
        q_with_bias_v = torch.cat(
            [q_with_bias_v * (torch.sin(p[:, :tgt_len, :] * 0.5 * torch.pi) + self.limit),
             q_with_bias_v * torch.cos(p[:, :tgt_len, :] * 0.5 * torch.pi)], dim=-1)
        p = torch.cat(
            [torch.cos(p[:, :src_len, :] * 0.5 * torch.pi) + self.limit,
             -1 * torch.sin(p[:, :src_len, :] * 0.5 * torch.pi)], dim=-1)
        matrix_bd = torch.bmm(q_with_bias_v, p.transpose(1, 2)) / 6
        scores = (matrix_ac + matrix_bd) / math.sqrt(
            self.d_k)
        attn = torch.bmm(scores, v)
        x = attn.transpose(0, 1).contiguous().view(tgt_len, n_batch, -1)
        return self.linear_out(x).transpose(0, 1)

    def left_product_new_rpe3(self,
                              query: torch.Tensor,
                              key: torch.Tensor,
                              value: torch.Tensor,
                              mask: Optional[torch.Tensor],
                              pos_emb: torch.Tensor,
                              eps: float = 1e-6,
                              ) -> torch.Tensor:
        tgt_len, src_len, n_batch = query.size(1), key.size(1), query.size(0)
        m = max(src_len, tgt_len)
        weight_index = self.get_index(m).to(query.device)

        q = self.linear_q(query.transpose(0, 1))
        k = self.linear_k(key.transpose(0, 1))
        v = self.linear_v(value.transpose(0, 1))

        # apply kernel
        k = self.kernel(k)
        q_with_bias_u = self.kernel((q + self.pos_bias_u.view(-1)))
        q_with_bias_v = self.kernel((q + self.pos_bias_v.view(-1)))

        # QK1
        q_with_bias_u = q_with_bias_u.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        k = k.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        # QK2
        q_with_bias_v = q_with_bias_v.view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        # V
        v = v.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)

        matrix_ac = torch.bmm(q_with_bias_u, k.transpose(1, 2))
        denom_ac = torch.clamp_min(matrix_ac.sum(dim=-1, keepdim=True), eps)
        matrix_ac /= denom_ac

        # compute matrix b and matrix d
        q_with_bias_v = torch.cat(
            [q_with_bias_v * torch.sin(weight_index[:, :tgt_len, :] / m),
             q_with_bias_v * torch.cos(weight_index[:, :tgt_len, :] / m),
             ], dim=-1)
        p = torch.cat(
            [torch.sin(weight_index[:, :src_len, :] / m).repeat(n_batch * self.h, 1, self.d_k),
             torch.cos(weight_index[:, :src_len, :] / m).repeat(n_batch * self.h, 1, self.d_k),
             ], dim=-1)
        matrix_bd = torch.bmm(q_with_bias_v, p.transpose(1, 2))  # * 2 / torch.pi
        denom_bd = torch.clamp_min(matrix_bd.sum(dim=-1, keepdim=True), eps)
        matrix_bd /= denom_bd
        scores = (matrix_ac + matrix_bd) * 0.5
        attn = torch.bmm(scores, v)
        x = attn.transpose(0, 1).contiguous().view(tgt_len, n_batch, -1)
        return self.linear_out(x).transpose(0, 1)

    def left_product_new_rpe4(self,
                              query: torch.Tensor,
                              key: torch.Tensor,
                              value: torch.Tensor,
                              mask: Optional[torch.Tensor],
                              pos_emb: torch.Tensor,
                              eps: float = 1e-6,
                              ) -> torch.Tensor:
        tgt_len, src_len, n_batch = query.size(1), key.size(1), query.size(0)
        m = max(src_len, tgt_len)
        weight_index = self.get_index(m).to(query.device)

        q = self.linear_q(query.transpose(0, 1))
        k = self.linear_k(key.transpose(0, 1))
        v = self.linear_v(value.transpose(0, 1))

        # apply kernel
        k = self.kernel(k)
        q_with_bias_u = self.kernel((q + self.pos_bias_u.view(-1)))
        q_with_bias_v = self.kernel((q + self.pos_bias_v.view(-1)))

        # QK1
        q_with_bias_u = q_with_bias_u.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        k = k.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        # QK2
        q_with_bias_v = q_with_bias_v.view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        # V
        v = v.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)

        matrix_ac = torch.bmm(q_with_bias_u, k.transpose(1, 2))
        denom_ac = torch.clamp_min(matrix_ac.sum(dim=-1, keepdim=True), eps)
        matrix_ac /= denom_ac

        # compute matrix b and matrix d
        q_with_bias_v = torch.cat(
            [q_with_bias_v * torch.sin(weight_index[:, :tgt_len, :] / m) * 0.5,
             q_with_bias_v * torch.cos(weight_index[:, :tgt_len, :] / m) * -0.5,
             q_with_bias_v * torch.full((n_batch * self.h, tgt_len, 1), 0.5, device=q.device)
             ], dim=-1)
        p = torch.cat(
            [torch.cos(weight_index[:, :src_len, :] / m).repeat(n_batch * self.h, 1, self.d_k),
             torch.sin(weight_index[:, :src_len, :] / m).repeat(n_batch * self.h, 1, self.d_k),
             torch.ones(n_batch * self.h, src_len, self.d_k, device=q.device)
             ], dim=-1)
        matrix_bd = torch.bmm(q_with_bias_v, p.transpose(1, 2))  # * 2 / torch.pi
        denom_bd = torch.clamp_min(matrix_bd.sum(dim=-1, keepdim=True), eps)
        matrix_bd /= denom_bd
        scores = (matrix_ac + matrix_bd) * 0.5
        attn = torch.bmm(scores, v)
        x = attn.transpose(0, 1).contiguous().view(tgt_len, n_batch, -1)
        return self.linear_out(x).transpose(0, 1)

    def left_product_new_rpe5(self,
                              query: torch.Tensor,
                              key: torch.Tensor,
                              value: torch.Tensor,
                              mask: Optional[torch.Tensor],
                              pos_emb: torch.Tensor,
                              eps: float = 1e-6,
                              ) -> torch.Tensor:
        tgt_len, src_len, n_batch = query.size(1), key.size(1), query.size(0)
        m = max(src_len, tgt_len)
        weight_index = self.get_index(m).to(query.device)
        div_term = self.get_div_term(self.d_k).to(query.device)

        q = self.linear_q(query.transpose(0, 1))
        k = self.linear_k(key.transpose(0, 1))
        v = self.linear_v(value.transpose(0, 1))

        # apply kernel
        k = self.kernel(k)
        q_with_bias_u = self.kernel((q + self.pos_bias_u.view(-1)))
        q_with_bias_v = self.kernel((q + self.pos_bias_v.view(-1)))

        # QK1
        q_with_bias_u = q_with_bias_u.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        k = k.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        # QK2
        q_with_bias_v = q_with_bias_v.view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        # V
        v = v.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)

        matrix_ac = torch.bmm(q_with_bias_u, k.transpose(1, 2))
        denom_ac = torch.clamp_min(matrix_ac.sum(dim=-1, keepdim=True), eps)
        matrix_ac /= denom_ac

        # compute matrix b and matrix d
        q_with_bias_v = torch.cat(
            [q_with_bias_v * torch.sin(weight_index[:, :tgt_len, :] / m * div_term) * 0.5,
             q_with_bias_v * torch.cos(weight_index[:, :tgt_len, :] / m * div_term) * -0.5,
             q_with_bias_v * torch.full((n_batch * self.h, tgt_len, 1), 0.5, device=q.device)
             ], dim=-1)
        p = torch.cat(
            [torch.cos((weight_index[:, :src_len, :] / m).repeat(n_batch * self.h, 1, self.d_k) * div_term),
             torch.sin((weight_index[:, :src_len, :] / m).repeat(n_batch * self.h, 1, self.d_k) * div_term),
             torch.ones(n_batch * self.h, src_len, self.d_k, device=q.device)
             ], dim=-1)
        matrix_bd = torch.bmm(q_with_bias_v, p.transpose(1, 2))  # * 2 / torch.pi
        denom_bd = torch.clamp_min(matrix_bd.sum(dim=-1, keepdim=True), eps)
        matrix_bd /= denom_bd
        scores = (matrix_ac + matrix_bd) * 0.5
        attn = torch.bmm(scores, v)
        x = attn.transpose(0, 1).contiguous().view(tgt_len, n_batch, -1)
        return self.linear_out(x).transpose(0, 1)

    def left_product_new_rpe6(self,
                              query: torch.Tensor,
                              key: torch.Tensor,
                              value: torch.Tensor,
                              mask: Optional[torch.Tensor],
                              pos_emb: torch.Tensor,
                              eps: float = 1e-6,
                              ) -> torch.Tensor:
        tgt_len, src_len, n_batch = query.size(1), key.size(1), query.size(0)
        m = max(src_len, tgt_len)
        weight_index = self.get_index(m).to(query.device)
        div_term = self.get_div_term(self.d_k).to(query.device)

        q = self.linear_q(query.transpose(0, 1))
        k = self.linear_k(key.transpose(0, 1))
        v = self.linear_v(value.transpose(0, 1))

        # apply kernel
        k = self.kernel(k)
        q_with_bias_u = self.kernel((q + self.pos_bias_u.view(-1)))
        q_with_bias_v = self.kernel((q + self.pos_bias_v.view(-1)))

        # QK1
        q_with_bias_u = q_with_bias_u.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        k = k.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        # QK2
        q_with_bias_v = q_with_bias_v.view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        # V
        v = v.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)

        matrix_ac = torch.bmm(q_with_bias_u, k.transpose(1, 2))
        denom_ac = torch.clamp_min(matrix_ac.sum(dim=-1, keepdim=True), eps)
        matrix_ac /= denom_ac

        # compute matrix b and matrix d
        q_with_bias_v = torch.cat(
            [q_with_bias_v * torch.sin((weight_index[:, :tgt_len, :] / m) * div_term),
             q_with_bias_v * torch.cos((weight_index[:, :tgt_len, :] / m) * div_term),
             ], dim=-1)
        p = torch.cat(
            [torch.sin((weight_index[:, :src_len, :] / m).repeat(n_batch * self.h, 1, self.d_k) * div_term),
             torch.cos((weight_index[:, :src_len, :] / m).repeat(n_batch * self.h, 1, self.d_k) * div_term),
             ], dim=-1)
        matrix_bd = torch.bmm(q_with_bias_v, p.transpose(1, 2))  # * 2 / torch.pi
        denom_bd = torch.clamp_min(matrix_bd.sum(dim=-1, keepdim=True), eps)
        matrix_bd /= denom_bd
        scores = (matrix_ac + matrix_bd) * 0.5
        attn = torch.bmm(scores, v)
        x = attn.transpose(0, 1).contiguous().view(tgt_len, n_batch, -1)
        return self.linear_out(x).transpose(0, 1)

    def right_product(self,
                      query: torch.Tensor,
                      key: torch.Tensor,
                      value: torch.Tensor,
                      mask: Optional[torch.Tensor],
                      pos_emb: torch.Tensor,
                      eps: float = 1e-6,
                      ) -> torch.Tensor:
        tgt_len, src_len, n_batch = query.size(1), key.size(1), query.size(0)
        m = max(src_len, tgt_len)
        weight_index = self.get_index(m).to(query.device)

        q = self.linear_q(query.transpose(0, 1))
        k = self.linear_k(key.transpose(0, 1))
        v = self.linear_v(value.transpose(0, 1))
        p = self.linear_pos(pos_emb.transpose(0, 1).repeat(1, n_batch, 1))

        # apply kernel
        k = self.kernel(k)
        p = self.kernel(p)
        q_with_bias_u = self.kernel((q + self.pos_bias_u.view(-1)))
        q_with_bias_v = self.kernel((q + self.pos_bias_v.view(-1)))

        # QK1
        q_with_bias_u = q_with_bias_u.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        k = k.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        # QK2
        q_with_bias_v = q_with_bias_v.view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        p = p.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        # V
        v = v.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)

        q_with_bias_u = torch.cat(
            [q_with_bias_u * torch.sin(weight_index[:, :tgt_len, :] / m),
             q_with_bias_u * torch.cos(weight_index[:, :tgt_len, :] / m)], dim=-1)
        k = torch.cat(
            [k * torch.sin(weight_index[:, :src_len, :] / m),
             k * torch.cos(weight_index[:, :src_len, :] / m)], dim=-1)

        kv_ac = torch.einsum('nld,nlm->ndm', k, v)
        z_ac = 1 / torch.clamp_min(torch.einsum('nld,nd->nl', q_with_bias_u, torch.sum(k, dim=1)), eps)
        attn_ac = torch.einsum('nld,ndm,nl->nlm', q_with_bias_u, kv_ac, z_ac)

        # compute matrix b and matrix d
        q_with_bias_v = torch.cat(
            [q_with_bias_v * torch.sin(weight_index[:, :tgt_len, :] / m),
             q_with_bias_v * torch.cos(weight_index[:, :tgt_len, :] / m)], dim=-1)
        # (N * h, S, 2 * d)
        p = torch.cat(
            [p * torch.sin(weight_index[:, :src_len, :] / m),
             p * torch.cos(weight_index[:, :src_len, :] / m)], dim=-1)
        pv_bd = torch.einsum('nld,nlm->ndm', p, v)
        z_bd = 1 / torch.clamp_min(torch.einsum('nld,nd->nl', q_with_bias_v, torch.sum(p, dim=1)), eps)
        attn_bd = torch.einsum('nld,ndm,nl->nlm', q_with_bias_v, pv_bd, z_bd)
        attn = (attn_ac + attn_bd) / math.sqrt(
            self.d_k)
        x = attn.transpose(0, 1).contiguous().view(tgt_len, n_batch, -1)
        return self.linear_out(x).transpose(0, 1)

    def right_product_rel_shift(self,
                                query: torch.Tensor,
                                key: torch.Tensor,
                                value: torch.Tensor,
                                mask: Optional[torch.Tensor],
                                pos_emb: torch.Tensor,
                                eps: float = 1e-6,
                                ) -> torch.Tensor:
        tgt_len, src_len, n_batch = query.size(1), key.size(1), query.size(0)
        m_ac = max(src_len, tgt_len)
        m_bd = max(src_len * 2 - 1, tgt_len)
        weight_index_ac = self.get_index(m_ac).to(query.device)
        weight_index_bd = self.get_index(m_bd).to(query.device)

        q = self.linear_q(query.transpose(0, 1))
        k = self.linear_k(key.transpose(0, 1))
        v = self.linear_v(value.transpose(0, 1))
        p = self.linear_pos(pos_emb.transpose(0, 1).repeat(1, n_batch, 1))

        # apply kernel
        k = self.kernel(k)
        p = self.kernel(p)
        q_with_bias_u = self.kernel((q + self.pos_bias_u.view(-1)))
        q_with_bias_v = self.kernel((q + self.pos_bias_v.view(-1)))
        # return q_with_bias_u, q_with_bias_v, k, p

        # QK1
        q_with_bias_u = q_with_bias_u.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        k = k.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        # QK2
        q_with_bias_v = q_with_bias_v.view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        p = p.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        # V
        v = v.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)

        q_with_bias_u = torch.cat(
            [q_with_bias_u * torch.sin(weight_index_ac[:, :tgt_len, :] / m_ac),
             q_with_bias_u * torch.cos(weight_index_ac[:, :tgt_len, :] / m_ac)], dim=-1)

        k = torch.cat(
            [k * torch.sin(weight_index_ac[:, :src_len, :] / m_ac),
             k * torch.cos(weight_index_ac[:, :src_len, :] / m_ac)], dim=-1)

        kv_ac = torch.einsum('nld,nlm->ndm', k, v)
        z_ac = 1 / torch.clamp_min(torch.einsum('nld,nd->nl', q_with_bias_u, torch.sum(k, dim=1)), eps)
        attn_ac = torch.einsum('nld,ndm,nl->nlm', q_with_bias_u, kv_ac, z_ac)

        # compute matrix b and matrix d
        q_with_bias_v = torch.cat(
            [q_with_bias_v * torch.sin(weight_index_bd[:, :tgt_len, :] / m_bd),
             q_with_bias_v * torch.cos(weight_index_bd[:, :tgt_len, :] / m_bd)], dim=-1)
        p = torch.cat(
            [p * torch.sin(weight_index_bd[:, :src_len * 2 - 1, :] / m_bd),
             p * torch.cos(weight_index_bd[:, :src_len * 2 - 1, :] / m_bd)], dim=-1)
        # p_rel = self.rel_shift3(p, q_with_bias_v.size(-2))
        p_rel = self.rel_shift4(p, q_with_bias_v.size(-2))
        # return p_rel

        pv_bd = torch.matmul(p_rel, v.unsqueeze(1))
        z_bd = 1 / torch.clamp_min(torch.einsum('nld,nd->nl', q_with_bias_v, torch.sum(p, dim=1)), eps)
        attn_bd = torch.matmul(q_with_bias_v.unsqueeze(2), pv_bd).squeeze(2)
        attn_bd = torch.einsum('nlm,nl->nlm', attn_bd, z_bd)
        attn = (attn_ac + attn_bd) / math.sqrt(
            self.d_k)
        x = attn.transpose(0, 1).contiguous().view(tgt_len, n_batch, -1)
        return self.linear_out(x).transpose(0, 1)

    def right_product_rel_shift_partial(self,
                                        query: torch.Tensor,
                                        key: torch.Tensor,
                                        value: torch.Tensor,
                                        mask: Optional[torch.Tensor],
                                        pos_emb: torch.Tensor,
                                        eps: float = 1e-6,
                                        ) -> torch.Tensor:
        tgt_len, src_len, n_batch = query.size(1), key.size(1), query.size(0)
        m_ac = max(src_len, tgt_len)
        m_bd = max(src_len * 2 - 1, tgt_len)
        weight_index_ac = self.get_index(m_ac).to(query.device)
        weight_index_bd = self.get_index(m_bd).to(query.device)

        q = self.linear_q(query.transpose(0, 1))
        k = self.linear_k(key.transpose(0, 1))
        v = self.linear_v(value.transpose(0, 1))
        p = self.linear_pos(pos_emb.transpose(0, 1).repeat(1, n_batch, 1))

        # apply kernel
        k = self.kernel(k)
        q_with_bias_u = self.kernel((q + self.pos_bias_u.view(-1)))
        q_with_bias_v = q + self.pos_bias_v.view(-1)
        # return q_with_bias_u, q_with_bias_v, k, p

        # QK1
        q_with_bias_u = q_with_bias_u.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        k = k.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        # QK2
        q_with_bias_v = q_with_bias_v.view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        p = p.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        # V
        v = v.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)

        q_with_bias_u = torch.cat(
            [q_with_bias_u * torch.sin(weight_index_ac[:, :tgt_len, :] / m_ac),
             q_with_bias_u * torch.cos(weight_index_ac[:, :tgt_len, :] / m_ac)], dim=-1)

        k = torch.cat(
            [k * torch.sin(weight_index_ac[:, :src_len, :] / m_ac),
             k * torch.cos(weight_index_ac[:, :src_len, :] / m_ac)], dim=-1)

        kv_ac = torch.einsum('nld,nlm->ndm', k, v)
        z_ac = 1 / torch.clamp_min(torch.einsum('nld,nd->nl', q_with_bias_u, torch.sum(k, dim=1)), eps)
        attn_ac = torch.einsum('nld,ndm,nl->nlm', q_with_bias_u, kv_ac, z_ac)
        attn_ac = attn_ac / math.sqrt(self.d_k)

        matrix_bd = torch.bmm(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = torch.softmax(matrix_bd, dim=-1)
        matrix_bd = self.rel_shift2(matrix_bd)
        scores_bd = matrix_bd / math.sqrt(self.d_k)
        attn_bd = torch.bmm(scores_bd, v)

        # compute matrix b and matrix d
        attn = attn_ac + attn_bd
        x = attn.transpose(0, 1).contiguous().view(tgt_len, n_batch, -1)
        return self.linear_out(x).transpose(0, 1)

    def right_product_fixed_rpe(self,
                                query: torch.Tensor,
                                key: torch.Tensor,
                                value: torch.Tensor,
                                mask: Optional[torch.Tensor],
                                pos_emb: torch.Tensor,
                                eps: float = 1e-6,
                                ) -> torch.Tensor:
        tgt_len, src_len, n_batch = query.size(1), key.size(1), query.size(0)
        m = max(src_len, tgt_len)
        weight_index = self.get_index(m).to(query.device)

        q = self.linear_q(query.transpose(0, 1))
        k = self.linear_k(key.transpose(0, 1))
        v = self.linear_v(value.transpose(0, 1))

        # apply kernel
        k = self.kernel(k)
        p = self.kernel(pos_emb)
        q_with_bias_u = self.kernel((q + self.pos_bias_u.view(-1)))
        q_with_bias_v = self.kernel((q + self.pos_bias_v.view(-1)))

        # QK1
        q_with_bias_u = q_with_bias_u.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        k = k.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        # QK2
        q_with_bias_v = q_with_bias_v.view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        # p = p.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        # V
        v = v.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)

        q_with_bias_u = torch.cat(
            [q_with_bias_u * torch.sin(weight_index[:, :tgt_len, :] / m),
             q_with_bias_u * torch.cos(weight_index[:, :tgt_len, :] / m)], dim=-1)
        k = torch.cat(
            [k * torch.sin(weight_index[:, :src_len, :] / m),
             k * torch.cos(weight_index[:, :src_len, :] / m)], dim=-1)
        kv_ac = torch.einsum('nld,nlm->ndm', k, v)
        z_ac = 1 / torch.clamp_min(torch.einsum('nld,nd->nl', q_with_bias_u, torch.sum(k, dim=1)), eps)
        attn_ac = torch.einsum('nld,ndm,nl->nlm', q_with_bias_u, kv_ac, z_ac)

        # compute matrix b and matrix d
        q_with_bias_v = torch.cat(
            [q_with_bias_v * torch.sin(weight_index[:, :tgt_len, :] / m),
             q_with_bias_v * torch.cos(weight_index[:, :tgt_len, :] / m)], dim=-1)
        # (N * h, S, 2 * d)
        p = torch.cat(
            [p * torch.sin(weight_index[:, :src_len, :] / m),
             p * torch.cos(weight_index[:, :src_len, :] / m)], dim=-1)
        pv_bd = torch.einsum('lsg,bsd->blgd', p, v)
        z_bd = 1 / torch.clamp_min(torch.einsum('blg,lg->bl', q_with_bias_v, torch.sum(p, dim=1)), eps)
        attn_bd = torch.einsum('blg,blgd,bl->bld', q_with_bias_v, pv_bd, z_bd)
        attn = (attn_ac + attn_bd) / math.sqrt(
            self.d_k)
        x = attn.transpose(0, 1).contiguous().view(tgt_len, n_batch, -1)
        return self.linear_out(x).transpose(0, 1)

    def right_product_new_rpe(self,
                              query: torch.Tensor,
                              key: torch.Tensor,
                              value: torch.Tensor,
                              mask: Optional[torch.Tensor],
                              pos_emb: torch.Tensor,
                              eps: float = 1e-6,
                              ) -> torch.Tensor:
        tgt_len, src_len, n_batch = query.size(1), key.size(1), query.size(0)
        weight_index = self.random_index.to(query.device)

        q = self.linear_q(query.transpose(0, 1))
        k = self.linear_k(key.transpose(0, 1))
        v = self.linear_v(value.transpose(0, 1))
        p = self.linear_pos(self.rescale_pos_emb(pos_emb).transpose(0, 1).repeat(1, n_batch, 1))

        # apply kernel
        k = self.kernel(k)
        p = self.kernel(p)
        q_with_bias_u = self.kernel((q + self.pos_bias_u.view(-1)))
        q_with_bias_v = self.kernel((q + self.pos_bias_v.view(-1)))

        # QK1
        q_with_bias_u = q_with_bias_u.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        k = k.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        # QK2
        q_with_bias_v = q_with_bias_v.view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        p = p.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        # V
        v = v.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)

        q_with_bias_u = torch.cat(
            [q_with_bias_u * torch.sin(weight_index[:, :tgt_len, :]),
             q_with_bias_u * torch.cos(weight_index[:, :tgt_len, :])], dim=-1)
        k = torch.cat(
            [k * torch.sin(weight_index[:, :src_len, :]),
             k * torch.cos(weight_index[:, :src_len, :])], dim=-1)

        kv_ac = torch.einsum('nld,nlm->ndm', k, v)
        z_ac = 1 / torch.clamp_min(torch.einsum('nld,nd->nl', q_with_bias_u, torch.sum(k, dim=1)), eps)
        attn_ac = torch.einsum('nld,ndm,nl->nlm', q_with_bias_u, kv_ac, z_ac)

        # compute matrix b and matrix d
        q_with_bias_v = torch.cat(
            [q_with_bias_v * torch.sin(p[:, :tgt_len, :]),
             q_with_bias_v * torch.cos(p[:, :tgt_len, :])], dim=-1)
        # (N * h, S, 2 * d)
        p = torch.cat(
            [torch.sin(p[:, :src_len, :]),
             torch.cos(p[:, :src_len, :])], dim=-1)
        pv_bd = torch.einsum('nld,nlm->ndm', p, v)
        z_bd = 1 / torch.clamp_min(torch.einsum('nld,nd->nl', q_with_bias_v, torch.sum(p, dim=1)), eps)
        attn_bd = torch.einsum('nld,ndm,nl->nlm', q_with_bias_v, pv_bd, z_bd)
        attn = (attn_ac + attn_bd) / math.sqrt(
            self.d_k)
        x = attn.transpose(0, 1).contiguous().view(tgt_len, n_batch, -1)
        return self.linear_out(x).transpose(0, 1)

    def right_product_new_rpe2(self,
                               query: torch.Tensor,
                               key: torch.Tensor,
                               value: torch.Tensor,
                               mask: Optional[torch.Tensor],
                               pos_emb: torch.Tensor,
                               eps: float = 1e-6,
                               ) -> torch.Tensor:
        tgt_len, src_len, n_batch = query.size(1), key.size(1), query.size(0)
        weight_index = self.random_index.to(query.device)

        q = self.linear_q(query.transpose(0, 1))
        k = self.linear_k(key.transpose(0, 1))
        v = self.linear_v(value.transpose(0, 1))
        p = self.linear_pos(pos_emb.transpose(0, 1).repeat(1, n_batch, 1))

        # apply kernel
        k = self.kernel(k)
        # p = self.kernel(p)
        q_with_bias_u = self.kernel((q + self.pos_bias_u.view(-1)))
        q_with_bias_v = self.kernel((q + self.pos_bias_v.view(-1)))

        # QK1
        q_with_bias_u = q_with_bias_u.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        k = k.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        # QK2
        q_with_bias_v = q_with_bias_v.view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        p = p.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        # V
        v = v.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)

        q_with_bias_u = torch.cat(
            [q_with_bias_u * torch.sin(weight_index[:, :tgt_len, :]),
             q_with_bias_u * torch.cos(weight_index[:, :tgt_len, :])], dim=-1)
        k = torch.cat(
            [k * torch.sin(weight_index[:, :src_len, :]),
             k * torch.cos(weight_index[:, :src_len, :])], dim=-1)

        kv_ac = torch.einsum('nld,nlm->ndm', k, v)
        z_ac = 1 / torch.clamp_min(torch.einsum('nld,nd->nl', q_with_bias_u, torch.sum(k, dim=1)), eps)
        attn_ac = torch.einsum('nld,ndm,nl->nlm', q_with_bias_u, kv_ac, z_ac)

        # compute matrix b and matrix d
        q_with_bias_v = torch.cat(
            [q_with_bias_v * (torch.sin(p[:, :tgt_len, :] * 0.5 * torch.pi) + self.limit),
             q_with_bias_v * torch.cos(p[:, :tgt_len, :] * 0.5 * torch.pi)], dim=-1)
        # (N * h, S, 2 * d)
        p = torch.cat(
            [torch.cos(p[:, :src_len, :] * 0.5 * torch.pi) + self.limit,
             -1 * torch.sin(p[:, :src_len, :] * 0.5 * torch.pi)], dim=-1)
        pv_bd = torch.einsum('nld,nlm->ndm', p, v) / 6
        attn_bd = torch.einsum('nld,ndm->nlm', q_with_bias_v, pv_bd)
        attn = (attn_ac + attn_bd) / math.sqrt(
            self.d_k)
        x = attn.transpose(0, 1).contiguous().view(tgt_len, n_batch, -1)
        return self.linear_out(x).transpose(0, 1)

    def right_product_new_rpe3(self,
                               query: torch.Tensor,
                               key: torch.Tensor,
                               value: torch.Tensor,
                               mask: Optional[torch.Tensor],
                               pos_emb: torch.Tensor,
                               eps: float = 1e-6,
                               ) -> torch.Tensor:
        tgt_len, src_len, n_batch = query.size(1), key.size(1), query.size(0)
        m = max(src_len, tgt_len)
        weight_index = self.get_index(m).to(query.device)

        q = self.linear_q(query.transpose(0, 1))
        k = self.linear_k(key.transpose(0, 1))
        v = self.linear_v(value.transpose(0, 1))

        # apply kernel
        k = self.kernel(k)
        q_with_bias_u = self.kernel((q + self.pos_bias_u.view(-1)))
        q_with_bias_v = self.kernel((q + self.pos_bias_v.view(-1)))

        # QK1
        q_with_bias_u = q_with_bias_u.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        k = k.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        # QK2
        q_with_bias_v = q_with_bias_v.view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        # V
        v = v.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)

        kv_ac = torch.einsum('nld,nlm->ndm', k, v)
        z_ac = 1 / torch.clamp_min(torch.einsum('nld,nd->nl', q_with_bias_u, torch.sum(k, dim=1)), eps)
        attn_ac = torch.einsum('nld,ndm,nl->nlm', q_with_bias_u, kv_ac, z_ac)

        # compute matrix b and matrix d
        q_with_bias_v = torch.cat(
            [q_with_bias_v * torch.sin(weight_index[:, :tgt_len, :] / m),
             q_with_bias_v * torch.cos(weight_index[:, :tgt_len, :] / m),
             ], dim=-1)
        p = torch.cat(
            [torch.sin(weight_index[:, :src_len, :] / m).repeat(n_batch * self.h, 1, self.d_k),
             torch.cos(weight_index[:, :src_len, :] / m).repeat(n_batch * self.h, 1, self.d_k),
             ], dim=-1)
        pv_bd = torch.einsum('nld,nlm->ndm', p, v)
        z_bd = 1 / torch.clamp_min(torch.einsum('nld,nd->nl', q_with_bias_v, torch.sum(p, dim=1)), eps)
        attn_bd = torch.einsum('nld,ndm,nl->nlm', q_with_bias_v, pv_bd, z_bd)  # * 0.5 + 0.5
        # attn_bd = torch.einsum('nld,ndm->nlm', q_with_bias_v, pv_bd)
        attn = (attn_ac + attn_bd) * 0.5
        x = attn.transpose(0, 1).contiguous().view(tgt_len, n_batch, -1)
        return self.linear_out(x).transpose(0, 1)

    def right_product_new_rpe4(self,
                               query: torch.Tensor,
                               key: torch.Tensor,
                               value: torch.Tensor,
                               mask: Optional[torch.Tensor],
                               pos_emb: torch.Tensor,
                               eps: float = 1e-6,
                               ) -> torch.Tensor:
        tgt_len, src_len, n_batch = query.size(1), key.size(1), query.size(0)
        m = max(src_len, tgt_len)
        weight_index = self.get_index(m).to(query.device)

        q = self.linear_q(query.transpose(0, 1))
        k = self.linear_k(key.transpose(0, 1))
        v = self.linear_v(value.transpose(0, 1))

        # apply kernel
        k = self.kernel(k)
        q_with_bias_u = self.kernel((q + self.pos_bias_u.view(-1)))
        q_with_bias_v = self.kernel((q + self.pos_bias_v.view(-1)))

        # QK1
        q_with_bias_u = q_with_bias_u.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        k = k.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        # QK2
        q_with_bias_v = q_with_bias_v.view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        # V
        v = v.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)

        kv_ac = torch.einsum('nld,nlm->ndm', k, v)
        z_ac = 1 / torch.clamp_min(torch.einsum('nld,nd->nl', q_with_bias_u, torch.sum(k, dim=1)), eps)
        attn_ac = torch.einsum('nld,ndm,nl->nlm', q_with_bias_u, kv_ac, z_ac)

        # compute matrix b and matrix d
        q_with_bias_v = torch.cat(
            [q_with_bias_v * torch.sin(weight_index[:, :tgt_len, :] / m) * 0.5,
             q_with_bias_v * torch.cos(weight_index[:, :tgt_len, :] / m) * -0.5,
             q_with_bias_v * torch.full((n_batch * self.h, tgt_len, 1), 0.5, device=q.device)
             ], dim=-1)
        p = torch.cat(
            [torch.cos(weight_index[:, :src_len, :] / m).repeat(n_batch * self.h, 1, self.d_k),
             torch.sin(weight_index[:, :src_len, :] / m).repeat(n_batch * self.h, 1, self.d_k),
             torch.ones(n_batch * self.h, src_len, self.d_k, device=q.device)
             ], dim=-1)
        pv_bd = torch.einsum('nld,nlm->ndm', p, v)
        z_bd = 1 / torch.clamp_min(torch.einsum('nld,nd->nl', q_with_bias_v, torch.sum(p, dim=1)), eps)
        attn_bd = torch.einsum('nld,ndm,nl->nlm', q_with_bias_v, pv_bd, z_bd)  # * 0.5 + 0.5
        # attn_bd = torch.einsum('nld,ndm->nlm', q_with_bias_v, pv_bd)
        attn = (attn_ac + attn_bd) * 0.5
        x = attn.transpose(0, 1).contiguous().view(tgt_len, n_batch, -1)
        return self.linear_out(x).transpose(0, 1)

    def right_product_new_rpe5(self,
                               query: torch.Tensor,
                               key: torch.Tensor,
                               value: torch.Tensor,
                               mask: Optional[torch.Tensor],
                               pos_emb: torch.Tensor,
                               eps: float = 1e-6,
                               ) -> torch.Tensor:
        tgt_len, src_len, n_batch = query.size(1), key.size(1), query.size(0)
        m = max(src_len, tgt_len)
        weight_index = self.get_index(m).to(query.device)
        div_term = self.get_div_term(self.d_k).to(query.device)

        q = self.linear_q(query.transpose(0, 1))
        k = self.linear_k(key.transpose(0, 1))
        v = self.linear_v(value.transpose(0, 1))

        # apply kernel
        k = self.kernel(k)
        q_with_bias_u = self.kernel((q + self.pos_bias_u.view(-1)))
        q_with_bias_v = self.kernel((q + self.pos_bias_v.view(-1)))

        # QK1
        q_with_bias_u = q_with_bias_u.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        k = k.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        # QK2
        q_with_bias_v = q_with_bias_v.view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        # V
        v = v.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)

        kv_ac = torch.einsum('nld,nlm->ndm', k, v)
        z_ac = 1 / torch.clamp_min(torch.einsum('nld,nd->nl', q_with_bias_u, torch.sum(k, dim=1)), eps)
        attn_ac = torch.einsum('nld,ndm,nl->nlm', q_with_bias_u, kv_ac, z_ac)

        # compute matrix b and matrix d
        q_with_bias_v = torch.cat(
            [q_with_bias_v * torch.sin(weight_index[:, :tgt_len, :] / m * div_term) * 0.5,
             q_with_bias_v * torch.cos(weight_index[:, :tgt_len, :] / m * div_term) * -0.5,
             q_with_bias_v * torch.full((n_batch * self.h, tgt_len, 1), 0.5, device=q.device)
             ], dim=-1)
        p = torch.cat(
            [torch.cos((weight_index[:, :src_len, :] / m).repeat(n_batch * self.h, 1, self.d_k) * div_term),
             torch.sin((weight_index[:, :src_len, :] / m).repeat(n_batch * self.h, 1, self.d_k) * div_term),
             torch.ones(n_batch * self.h, src_len, self.d_k, device=q.device)
             ], dim=-1)
        pv_bd = torch.einsum('nld,nlm->ndm', p, v)
        z_bd = 1 / torch.clamp_min(torch.einsum('nld,nd->nl', q_with_bias_v, torch.sum(p, dim=1)), eps)
        attn_bd = torch.einsum('nld,ndm,nl->nlm', q_with_bias_v, pv_bd, z_bd)  # * 0.5 + 0.5
        # attn_bd = torch.einsum('nld,ndm->nlm', q_with_bias_v, pv_bd)
        attn = (attn_ac + attn_bd) * 0.5
        x = attn.transpose(0, 1).contiguous().view(tgt_len, n_batch, -1)
        return self.linear_out(x).transpose(0, 1)

    def right_product_new_rpe6(self,
                               query: torch.Tensor,
                               key: torch.Tensor,
                               value: torch.Tensor,
                               mask: Optional[torch.Tensor],
                               pos_emb: torch.Tensor,
                               eps: float = 1e-6,
                               ) -> torch.Tensor:
        tgt_len, src_len, n_batch = query.size(1), key.size(1), query.size(0)
        m = max(src_len, tgt_len)
        weight_index = self.get_index(m).to(query.device)
        div_term = self.get_div_term(self.d_k).to(query.device)

        q = self.linear_q(query.transpose(0, 1))
        k = self.linear_k(key.transpose(0, 1))
        v = self.linear_v(value.transpose(0, 1))

        # apply kernel
        k = self.kernel(k)
        q_with_bias_u = self.kernel((q + self.pos_bias_u.view(-1)))
        q_with_bias_v = self.kernel((q + self.pos_bias_v.view(-1)))

        # QK1
        q_with_bias_u = q_with_bias_u.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        k = k.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        # QK2
        q_with_bias_v = q_with_bias_v.view(-1, n_batch * self.h, self.d_k).transpose(0, 1)
        # V
        v = v.contiguous().view(-1, n_batch * self.h, self.d_k).transpose(0, 1)

        kv_ac = torch.einsum('nld,nlm->ndm', k, v)
        z_ac = 1 / torch.clamp_min(torch.einsum('nld,nd->nl', q_with_bias_u, torch.sum(k, dim=1)), eps)
        attn_ac = torch.einsum('nld,ndm,nl->nlm', q_with_bias_u, kv_ac, z_ac)

        # compute matrix b and matrix d
        q_with_bias_v = torch.cat(
            [q_with_bias_v * torch.sin((weight_index[:, :tgt_len, :] / m) * div_term),
             q_with_bias_v * torch.cos((weight_index[:, :tgt_len, :] / m) * div_term),
             ], dim=-1)
        p = torch.cat(
            [torch.sin((weight_index[:, :src_len, :] / m).repeat(n_batch * self.h, 1, self.d_k) * div_term),
             torch.cos((weight_index[:, :src_len, :] / m).repeat(n_batch * self.h, 1, self.d_k) * div_term),
             ], dim=-1)
        pv_bd = torch.einsum('nld,nlm->ndm', p, v)
        z_bd = 1 / torch.clamp_min(torch.einsum('nld,nd->nl', q_with_bias_v, torch.sum(p, dim=1)), eps)
        attn_bd = torch.einsum('nld,ndm,nl->nlm', q_with_bias_v, pv_bd, z_bd)  # * 0.5 + 0.5
        # attn_bd = torch.einsum('nld,ndm->nlm', q_with_bias_v, pv_bd)
        attn = (attn_ac + attn_bd) * 0.5
        x = attn.transpose(0, 1).contiguous().view(tgt_len, n_batch, -1)
        return self.linear_out(x).transpose(0, 1)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                pos_emb: torch.Tensor = torch.empty(0)) -> torch.Tensor:
        if self.deploy:
            if self.cosformer_version == 1:
                x = self.right_product_rel_shift_partial(query, key, value, mask, pos_emb, self.eps)
            elif self.cosformer_version == 2:
                x = self.right_product_rel_shift(query, key, value, mask, pos_emb, self.eps)
            elif self.cosformer_version == 3:
                x = self.right_product_fixed_rpe(query, key, value, mask, pos_emb, self.eps)
            elif self.cosformer_version == 4:
                x = self.right_product_new_rpe(query, key, value, mask, pos_emb, self.eps)
            elif self.cosformer_version == 5:
                x = self.right_product_new_rpe2(query, key, value, mask, pos_emb, self.eps)
            elif self.cosformer_version == 6:
                x = self.right_product_new_rpe3(query, key, value, mask, pos_emb, self.eps)
            elif self.cosformer_version == 7:
                x = self.right_product_new_rpe4(query, key, value, mask, pos_emb, self.eps)
            elif self.cosformer_version == 8:
                x = self.right_product_new_rpe5(query, key, value, mask, pos_emb, self.eps)
            elif self.cosformer_version == 9:
                x = self.right_product_new_rpe6(query, key, value, mask, pos_emb, self.eps)
            else:  # version 0
                x = self.right_product(query, key, value, mask, pos_emb, self.eps)
        else:
            if self.cosformer_version == 1:
                x = self.left_product_rel_shift_partial(query, key, value, mask, pos_emb, self.eps)
            elif self.cosformer_version == 2:
                x = self.left_product_rel_shift(query, key, value, mask, pos_emb, self.eps)
            elif self.cosformer_version == 3:
                x = self.left_product_fixed_rpe(query, key, value, mask, pos_emb, self.eps)
            elif self.cosformer_version == 4:
                x = self.left_product_new_rpe(query, key, value, mask, pos_emb, self.eps)
            elif self.cosformer_version == 5:
                x = self.left_product_new_rpe2(query, key, value, mask, pos_emb, self.eps)
            elif self.cosformer_version == 6:
                x = self.left_product_new_rpe3(query, key, value, mask, pos_emb, self.eps)
            elif self.cosformer_version == 7:
                x = self.left_product_new_rpe4(query, key, value, mask, pos_emb, self.eps)
            elif self.cosformer_version == 8:
                x = self.left_product_new_rpe5(query, key, value, mask, pos_emb, self.eps)
            elif self.cosformer_version == 9:
                x = self.left_product_new_rpe6(query, key, value, mask, pos_emb, self.eps)
            else:
                x = self.left_product(query, key, value, mask, pos_emb, self.eps)
        return x


def test(batch=2, tgt_len=10, src_len=20, embed_dim=128, num_heads=8, N=100, causal=False):
    # model = MultiHeadedAttention_cosformer_official(embed_dim=embed_dim, num_heads=num_heads, causal=causal)
    model = MultiHeadedAttention_cosformer_official_degenerate(
        n_head=num_heads, n_feat=embed_dim,
        causal=causal, dropout_rate=0.,
        cosformer_version=8)
    # model = MultiHeadedAttention_cosformer_official(n_head=num_heads, n_feat=embed_dim,
    #                                                 causal=causal, dropout_rate=0.,
    #                                                 cosformer_version=3)
    # model = Cos_RelPositionMultiHeadedAttention_rel_shift(
    #     num_heads, embed_dim, 0., causal=True, deploy=False, cosformer_version=7)
    pos_emb_func = PositionalEncoding(embed_dim, 0.)
    x = torch.rand(1, src_len, embed_dim)
    pos_emb = pos_emb_func(x)[1]
    # print('pos_emb', pos_emb.size())
    diff = 0
    if causal:
        mask = (torch.triu(torch.ones(tgt_len, tgt_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf'))
        print(mask.size(), mask.size())
    else:
        mask = None
    pass_num = 0
    for i in tqdm(range(N)):
        query = torch.rand(tgt_len, batch, embed_dim)
        key = torch.rand(src_len, batch, embed_dim)
        value = torch.rand(src_len, batch, embed_dim)
        # query = torch.rand(batch, tgt_len, embed_dim)
        # key = torch.rand(batch, src_len, embed_dim)
        # value = torch.rand(batch, src_len, embed_dim)
        # left_res = model.left_product(query, key, value, mask)
        # right_res = model.right_product(query, key, value, mask)
        # left_res = model.left_product_new_rpe5(query, key, value, mask, pos_emb)
        # right_res = model.right_product_new_rpe5(query, key, value, mask, pos_emb)
        # left_res = model.left_product_sin(query, key, value, mask)
        # right_res = model.right_product_sin(query, key, value, mask)
        # left_res = model.left_product_sin_cos(query, key, value, mask)
        # right_res = model.right_product_sin_cos(query, key, value, mask)
        # left_res = model.left_product_sin_cos2(query, key, value, mask)
        # right_res = model.right_product_sin_cos2(query, key, value, mask)
        # left_res = model.left_product_sin_cos3(query, key, value, mask)
        # right_res = model.right_product_sin_cos3(query, key, value, mask)
        # right_res = model(query, key, value)
        # left_res = model.left_product_single_cos5(query, key, value, mask)
        # right_res = model.right_product_single_cos5(query, key, value, mask)
        left_res = model.left_product_rope2(query, key, value, mask)
        right_res = model.right_product_rope2(query, key, value, mask)
        # print(left_res.size(), right_res.size())
        # diff += torch.norm(left_res - right_res)
        error = np.mean(np.abs(right_res.detach().numpy() - left_res.detach().numpy()))
        if error < 1e-4:
            print(termcolor.colored("PASS", color="green"), error)
            pass_num += 1
        else:
            print(termcolor.colored("FAIL", color="red"), error)
    # diff /= N
    print(termcolor.colored("pass rate is {:2f}%".format(pass_num / N * 100), color="blue"))
    # if causal:
    #     print("Test result for causal model:")
    # else:
    #     print("Test result for bidirectional model:")
    # print(f"The error of left multiplication and right multiplication is {diff}")


def main():
    test(tgt_len=10, src_len=20, causal=False, N=100)
    # test(tgt_len=10, src_len=10, causal=True)


if __name__ == '__main__':
    main()
