#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: mlm.py
@author: YQ
@date: 2026-05-20
@desc: TODO: 在这里简单描述本文件的功能。
"""


from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from appsep.models.base import AppConfig, ModelConfig
from appsep.models.embeddings import AppEmbeddings
from appsep.models.encoder import BertEncoderWithRoPE


class AppMLM(PreTrainedModel):
    """App 序列的 MLM（Masked Language Model）预训练模型。

    预训练任务：随机 mask 一些 app token，模型预测被 mask 的 token。

    结构：
        Input → AppEmbeddings → BertEncoderWithRoPE → MLM Head → logits

    MLM Head：
        hidden_states → Linear(hidden_size, emb_size) → GELU → LayerNorm → Linear(emb_size, vocab_size)

    预测时直接复用 word_embeddings 的转置，避免额外存储一个大矩阵。

    Args:
        config: ModelConfig 实例（包含 app_config）
    """

    config_class = ModelConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.app_config = (
            AppConfig(**config.app_config)
            if isinstance(config.app_config, dict)
            else config.app_config
        )
        c = self.app_config

        # Backbone
        self.app_embeddings = AppEmbeddings(c)
        self.bert = BertEncoderWithRoPE(c)

        # MLM Head
        self.mlm_dense = nn.Linear(c.hidden_size, c.emb_size)
        self.mlm_act = nn.GELU()
        self.mlm_norm = nn.LayerNorm(c.emb_size, eps=c.layer_norm_eps)
        self.mlm_bias = nn.Parameter(torch.zeros(c.vocab_size))

        self.post_init()

    def get_extended_attention_mask(
        self, attention_mask: torch.Tensor, dtype=None
    ):
        """将 (B, L) 的 attention_mask 扩展为 (B, 1, 1, L) 的广播格式。

        扩展后的 mask：padding 位置为 -inf，正常位置为 0。
        """
        if dtype is None:
            dtype = self.app_embeddings.word_embeddings.weight.dtype
        ext = attention_mask[:, None, None, :].to(dtype=dtype)
        return (1.0 - ext) * torch.finfo(dtype).min

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # 默认 attention_mask
        if attention_mask is None:
            attention_mask = (
                input_ids != self.app_config.pad_token_id
            ).long()

        dtype = self.app_embeddings.word_embeddings.weight.dtype
        ext_mask = self.get_extended_attention_mask(attention_mask, dtype)

        # Backbone
        seq_out = self.bert(
            self.app_embeddings(input_ids), attention_mask=ext_mask
        )[0]

        # MLM Head
        x = self.mlm_norm(self.mlm_act(self.mlm_dense(seq_out)))

        # 预测层（共享词嵌入权重）
        vocab_weight = self.app_embeddings.word_embeddings.weight
        logits = x @ vocab_weight.t() + self.mlm_bias

        if labels is not None:
            # 只计算被 mask 位置（labels != -100）的 loss
            active_mask = labels.view(-1) != -100
            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)

            logits_active = logits_flat[active_mask]
            labels_active = labels_flat[active_mask]

            loss = F.cross_entropy(logits_active, labels_active)
            return {"loss": loss}

        return {"logits": logits}

    def enable_gradient_checkpointing(self):
        """启用梯度检查点，节省显存。"""
        self.bert.gradient_checkpointing_enable()
