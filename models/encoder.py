#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: encoder.py
@author: YQ
@date: 2026-05-20
@desc: 

"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig

from appsep.utils.rope import RotaryEmbedding


class BertAttentionWithRoPE(nn.Module):
    """带 RoPE 的多头注意力层。

    结构：Q/K/V 投影 → RoPE → Dot-Product Attention → 输出投影

    与标准 BertAttention 的区别：
    - 不使用相对位置编码（如 T5's BiDesign）或绝对位置编码
    - 使用 RoPE 将位置信息注入 Q 和 K，复用了 rotate_half 的数学性质
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        # Q, K, V 投影
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.scale = self.head_dim**0.5

        # RoPE 模块（预计算频率）
        self.rope = RotaryEmbedding(dim=self.head_dim, device=None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S, _ = hidden_states.shape

        def _split(x: torch.Tensor) -> torch.Tensor:
            return x.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Q, K, V 投影
        q = _split(self.q_proj(hidden_states))
        k = _split(self.k_proj(hidden_states))
        v = _split(self.v_proj(hidden_states))

        # 应用 RoPE（旋转 Q 和 K）
        q, k = self.rope(q, k, seq_len=S)

        # 注意力分数
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if attention_mask is not None:
            attn = attn + attention_mask

        attn = self.dropout(F.softmax(attn, dim=-1))

        # 合并 heads
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)

        return self.out_proj(out)


class BertLayerWithRoPE(nn.Module):
    """单个 Transformer 层（Pre-Norm 结构）。

    结构：LayerNorm(1) → Attention → Add → LayerNorm(2) → FFN → Add

    使用 Pre-Norm 而非 Post-Norm：
    - 稳定性更好，训练更鲁棒
    - 与原版 BERT 的 Post-Norm 有细微差异，但不影响效果
    """

    def __init__(self, config: BertConfig):
        super().__init__()

        self.attention = BertAttentionWithRoPE(config)

        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)

        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.act = nn.GELU()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-Norm: 先 Norm 再 Attention
        hidden_states = self.ln1(
            hidden_states + self.attention(hidden_states, attention_mask)
        )

        # FFN（带残差）
        ffn_input = self.ln2(hidden_states)
        ffn_output = self.dropout(self.output(self.act(self.intermediate(ffn_input))))
        hidden_states = hidden_states + ffn_output

        return hidden_states


class BertEncoderWithRoPE(nn.Module):
    """多层 Transformer 编码器（含 RoPE）。

    结构：N × BertLayerWithRoPE

    返回格式与 HuggingFace BertEncoder 兼容（返回 tuple），
    以便于直接替换使用。
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [BertLayerWithRoPE(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return (hidden_states,)
