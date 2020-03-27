#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Decoder self-attention layer definition."""

import torch
from torch import nn
import torch.nn.functional as F

from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm


class MoEAttn(nn.Module):
    def __init__(self, size, cn_src_attn, en_src_attn, mode='linear'):
        """Construct an HANMoE src attention object."""
        super(MoEAttn, self).__init__()
        self.cn_src_attn = cn_src_attn
        self.en_src_attn = en_src_attn
        self.mode = mode
        if mode == 'linear':
            self.linear_mixer = nn.Sequential(
                nn.Linear(3*size, 1),
                nn.Sigmoid())
        elif mode == 'han_dot':
            # han_mixer using x as 'query'
            self.scaling = 2.0
            self.mlp_cn = nn.Sequential(nn.Linear(size, size),
                                        nn.Tanh())
            self.mlp_en = nn.Sequential(nn.Linear(size, size),
                                        nn.Tanh())
            self.mlp_x = nn.Sequential(nn.Linear(size, size),
                                        nn.Tanh())
        else:
            raise Exception('Not implemented method for HANMoE src_attn type {}'.format(mode))

    def forward(self, x, cn_memory, en_memory, memory_mask):
        cn_att_c = self.cn_src_attn(x, cn_memory, cn_memory, memory_mask)
        en_att_c = self.en_src_attn(x, en_memory, en_memory, memory_mask)
        # here we mix two att context
        if self.mode == "linear":
            # (B,U,1), range from (0, 1)
            lambda_ = self.linear_mixer(torch.cat((cn_att_c, en_att_c, x), dim=-1)) 
            han_att_c = lambda_ * cn_att_c + (1 - lambda_) * en_att_c
        elif self.mode == 'han_dot':
            q = self.mlp_x(x) # (B,U,D)
            k_cn = self.mlp_cn(cn_att_c)
            k_en = self.mlp_en(en_att_c)
            e = torch.cat((torch.sum(q * k_cn, dim=-1).unsqueeze(-1),
                torch.sum(q * k_en, dim=-1).unsqueeze(-1)), dim=-1) # (B,U,2)
            lambda_ = F.softmax(self.scaling * e, dim=-1).unsqueeze(-1) # (B,U,2,1)
            han_att_c = lambda_[:,:,0] * cn_att_c + lambda_[:,:,1] * en_att_c

        return han_att_c


class HANDecoderLayer(nn.Module):
    """Single decoder layer module.

    :param int size: input dim
    :param espnet.nets.pytorch_backend.transformer.attention.MultiHeadedAttention self_attn: self attention module
    :param espnet.nets.pytorch_backend.transformer.attention.MultiHeadedAttention src_attn: source attention module
    :param espnet.nets.pytorch_backend.transformer.positionwise_feed_forward.PositionwiseFeedForward feed_forward:
        feed forward layer module
    :param float dropout_rate: dropout rate
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied. i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)

    """

    def __init__(self, size, self_attn, cn_src_attn, en_src_attn, feed_forward, dropout_rate,
                 moe_att_mode='linear',
                 normalize_before=True, concat_after=False):
        """Construct an DecoderLayer object."""
        super(HANDecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.norm3 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear1 = nn.Linear(size + size, size)
            self.concat_linear2 = nn.Linear(size + size, size)

        # Hierarchical attention
        self.cn_src_attn = cn_src_attn # declare attn here for initialization
        self.en_src_attn = en_src_attn
        self.src_attn = MoEAttn(size, cn_src_attn, en_src_attn, moe_att_mode)


    def forward(self, tgt, tgt_mask, cn_memory, en_memory, memory_mask, cache=None):
        """Compute decoded features.

        Args:
            tgt (torch.Tensor): decoded previous target features (batch, max_time_out, size)
            tgt_mask (torch.Tensor): mask for x (batch, max_time_out)
            memory (torch.Tensor): encoded source features (batch, max_time_in, size)
            memory_mask (torch.Tensor): mask for memory (batch, max_time_in)
            cache (torch.Tensor): cached output (batch, max_time_out-1, size)

        """
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)

        if cache is None:
            tgt_q = tgt
            tgt_q_mask = tgt_mask
        else:
            # compute only the last frame query keeping dim: max_time_out -> 1
            assert cache.shape == (tgt.shape[0], tgt.shape[1] - 1, self.size), \
                f"{cache.shape} == {(tgt.shape[0], tgt.shape[1] - 1, self.size)}"
            tgt_q = tgt[:, -1:, :]
            residual = residual[:, -1:, :]
            tgt_q_mask = None
            if tgt_mask is not None:
                tgt_q_mask = tgt_mask[:, -1:, :]

        if self.concat_after:
            tgt_concat = torch.cat((tgt_q, self.self_attn(tgt_q, tgt, tgt, tgt_q_mask)), dim=-1)
            x = residual + self.concat_linear1(tgt_concat)
        else:
            x = residual + self.dropout(self.self_attn(tgt_q, tgt, tgt, tgt_q_mask))
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        # HANMoE here, two src_attn is computed
        if self.concat_after:
            x_concat = torch.cat((x, self.src_attn(x, cn_memory, en_memory, memory_mask)), dim=-1)
            x = residual + self.concat_linear2(x_concat)
        else:
            x = residual + self.dropout(self.src_attn(x, cn_memory, en_memory, memory_mask))
        if not self.normalize_before:
            x = self.norm2(x)

        residual = x
        if self.normalize_before:
            x = self.norm3(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm3(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        return x, tgt_mask, cn_memory, en_memory, memory_mask
