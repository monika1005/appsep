"""
@file: cls.py
@desc: AppCLS 下游分类模型

结构：
    AppEmbeddings → BertEncoderWithRoPE → [CLS Token] → Dropout → Classifier → logits
                          ↑ 预训练权重（可选冻结）

支持：
- 加载预训练 backbone
- 冻结/解冻 backbone（分阶段微调）
- 样本加权损失
"""

import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import PreTrainedModel

from appsep.models.base import AppConfig, CLSModelConfig
from appsep.models.embeddings import AppEmbeddings
from appsep.models.encoder import BertEncoderWithRoPE


def _log(msg: str, rank: int = 0) -> None:
    """仅在指定 rank 打印日志。"""
    current_rank = int(os.environ.get("RANK", 0))
    if current_rank == rank:
        print(msg, flush=True)


class AppCLS(PreTrainedModel):
    """App 序列分类模型。

    结构：
        Input → AppEmbeddings → BertEncoderWithRoPE → [CLS] → Dropout → Classifier

    支持：
    - 加载预训练 backbone 权重（从 AppMLM）
    - 冻结/解冻 backbone（分阶段微调）
    - 样本加权损失（通过 weight 参数）
    - 自定义分类头（num_labels）

    Args:
        config: CLSModelConfig 实例（包含 app_config 和分类参数）
    """

    config_class = CLSModelConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: CLSModelConfig):
        super().__init__(config)

        self.app_config = (
            AppConfig(**config.app_config)
            if isinstance(config.app_config, dict)
            else config.app_config
        )
        c = self.app_config

        # Backbone（与 AppMLM 共享结构）
        self.app_embeddings = AppEmbeddings(c)
        self.bert = BertEncoderWithRoPE(c)

        # 分类头
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(c.hidden_size, config.num_labels)

        self.post_init()

    def get_extended_attention_mask(
        self, attention_mask: torch.Tensor, dtype=None
    ):
        if dtype is None:
            dtype = self.app_embeddings.word_embeddings.weight.dtype
        ext = attention_mask[:, None, None, :].to(dtype=dtype)
        return (1.0 - ext) * torch.finfo(dtype).min

    def load_pretrained_backbone(
        self,
        pretrained_path: str,
        strict: bool = True,
    ) -> Dict[str, int]:
        """从预训练模型加载 backbone 权重。

        Args:
            pretrained_path: 预训练模型路径（包含 model.safetensors 和 config.json）
            strict: 是否严格匹配 key

        Returns:
            {"missing": [...], "unexpected": [...]}
        """
        _log(f"加载预训练 backbone: {pretrained_path}")

        # 加载 config
        import json
        config_path = os.path.join(pretrained_path, "config.json")
        with open(config_path, "r") as f:
            _ = json.load(f)  # config 不需要重新构建

        # 加载权重
        weights_path = os.path.join(pretrained_path, "model.safetensors")
        if os.path.exists(weights_path):
            state_dict = load_file(weights_path)
        else:
            weights_path = os.path.join(pretrained_path, "pytorch_model.bin")
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)

        # 提取 embeddings 和 bert 的权重（去掉前缀）
        emb_keys = {k for k in state_dict if "app_embeddings" in k}
        bert_keys = {k for k in state_dict if "bert" in k}

        emb_dict = {k.replace("app_embeddings.", "", 1): v
                    for k in emb_keys}
        bert_dict = {k.replace("bert.", "", 1): v
                     for k in bert_keys}

        # 加载（不加载 mlm_head 相关权重）
        missing_emb, unexpected_emb = self.app_embeddings.load_state_dict(
            emb_dict, strict=False
        )
        missing_bert, unexpected_bert = self.bert.load_state_dict(
            bert_dict, strict=False
        )

        info = {
            "embeddings_missing": len(missing_emb),
            "embeddings_unexpected": len(unexpected_emb),
            "bert_missing": len(missing_bert),
            "bert_unexpected": len(unexpected_bert),
        }

        # 警告：检测潜在问题
        if missing_emb or missing_bert:
            _log(f"  ⚠️  backbone 加载有 missing key: "
                 f"emb={info['embeddings_missing']}, "
                 f"bert={info['bert_missing']}")
        if unexpected_emb or unexpected_bert:
            _log(f"  ⚠️  backbone 加载有 unexpected key（可能结构不匹配）: "
                 f"emb={info['embeddings_unexpected']}, "
                 f"bert={info['bert_unexpected']}")
        else:
            _log(f"  ✅ backbone 加载成功: "
                 f"emb={info['embeddings_missing']} missing, "
                 f"bert={info['bert_missing']} missing")

        return info

    def freeze_embeddings(self) -> None:
        """冻结词嵌入层。"""
        for p in self.app_embeddings.parameters():
            p.requires_grad = False
        _log("已冻结 app_embeddings")

    def freeze_bert(self) -> None:
        """冻结 Transformer 编码器。"""
        for p in self.bert.parameters():
            p.requires_grad = False
        _log("已冻结 bert encoder")

    def freeze_backbone(self) -> None:
        """冻结整个 backbone（embeddings + bert）。"""
        self.freeze_embeddings()
        self.freeze_bert()

    def unfreeze_backbone(self) -> None:
        """解冻整个 backbone。"""
        for p in self.app_embeddings.parameters():
            p.requires_grad = True
        for p in self.bert.parameters():
            p.requires_grad = True
        _log("已解冻 backbone（embeddings + bert）")

    def trainable_params(self) -> int:
        """返回可训练参数数量。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self) -> int:
        """返回总参数数量。"""
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        weight: Optional[torch.Tensor] = None,
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

        # 取 [CLS] token（position 0）
        cls_emb = seq_out[:, 0, :]
        cls_emb = self.dropout(cls_emb)

        # 分类
        logits = self.classifier(cls_emb)

        if labels is not None:
            if weight is not None:
                # 样本加权损失
                loss = F.cross_entropy(logits, labels, reduction="none")
                total_weight = weight.sum().clamp(min=1e-8)
                loss = (loss * weight).sum() / total_weight
            else:
                loss = F.cross_entropy(logits, labels)

            return {"loss": loss, "logits": logits}

        return {"logits": logits}

    def enable_gradient_checkpointing(self) -> None:
        """启用梯度检查点，节省显存。"""
        self.bert.gradient_checkpointing_enable()

    def extract_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """提取序列 embedding（不经过分类头）。

        用于下游任务（如聚类、相似度计算等）。

        Args:
            input_ids: (B, L)
            attention_mask: (B, L)，可选

        Returns:
            (B, hidden_size) 的 embedding
        """
        if attention_mask is None:
            attention_mask = (
                input_ids != self.app_config.pad_token_id
            ).long()

        dtype = self.app_embeddings.word_embeddings.weight.dtype
        ext_mask = self.get_extended_attention_mask(attention_mask, dtype)

        seq_out = self.bert(
            self.app_embeddings(input_ids), attention_mask=ext_mask
        )[0]

        # 取 [CLS] token
        return seq_out[:, 0, :]
