#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: embeddings.py
@author: YQ
@date: 2026-05-20
@desc: TODO: 在这里简单描述本文件的功能。
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers import BertConfig


class AppEmbeddings(nn.Module):
    """App 序列的嵌入层。

    结构：Embedding → LayerNorm → Dropout → （可选）Linear 投影

    Args:
        config: AppConfig 实例

    Note:
        与 BERT 不同，这里没有绝对位置编码。
        位置信息由 RoPE 在注意力层注入。
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config

        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            config.emb_size,
            padding_idx=config.pad_token_id,
        )
        self.LayerNorm = nn.LayerNorm(config.emb_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 如果词嵌入维度与隐藏层维度不同，添加投影层
        if config.emb_size != config.hidden_size:
            self.proj = nn.Linear(config.emb_size, config.hidden_size)
        else:
            self.proj = None

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            input_ids: (B, L)，token IDs

        Returns:
            (B, L, hidden_size) 或 (B, L, emb_size)
        """
        emb = self.word_embeddings(input_ids)
        emb = self.LayerNorm(emb)
        emb = self.dropout(emb)

        if self.proj is not None:
            emb = self.proj(emb)

        return emb

    @property
    def embedding_dim(self) -> int:
        return self.config.hidden_size
