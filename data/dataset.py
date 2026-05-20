#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: dataset.py
@author: YQ
@date: 2026-05-20
@desc: 

包含：
- AppMLMDataset：MLM 预训练数据集
- AppCLSDataset：分类微调数据集
"""

import random
from typing import Dict, List, Optional

import polars as pl
import torch
from torch.utils.data import Dataset

from appsep.config import CLSConfig, PAD_ID, CLS_ID, SEP_ID, MASK_ID


# =========================================================
# MLM Dataset
# =========================================================
class AppMLMDataset(Dataset):
    """MLM 预训练数据集。

    支持动态随机 mask：每次 __getitem__ 都会重新生成 mask。

    Args:
        sequences: App ID 序列列表，每个元素是一个 App ID 列表
        max_len: 最大序列长度（包含 [CLS] 和 [SEP]）
        vocab_size: 词表大小
        pad_id: PAD token ID
        cls_id: CLS token ID
        sep_id: SEP token ID
        mask_id: MASK token ID
        mlm_prob: mask 概率（默认 15%）
        random_token_start: 随机替换的起始 token ID（通常为特殊 token 之后）
        is_train: 是否为训练集（训练集每次随机，验证集固定 seed）
        seed: 验证集使用的随机 seed
    """

    def __init__(
        self,
        sequences: List[List[int]],
        max_len: int = 200,
        vocab_size: int = 10005,
        pad_id: int = PAD_ID,
        cls_id: int = CLS_ID,
        sep_id: int = SEP_ID,
        mask_id: int = MASK_ID,
        mlm_prob: float = 0.15,
        random_token_start: int = 5,
        is_train: bool = True,
        seed: int = 42,
    ):
        self.sequences = sequences
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.cls_id = cls_id
        self.sep_id = sep_id
        self.mask_id = mask_id
        self.mlm_prob = mlm_prob
        self.random_token_start = random_token_start
        self.is_train = is_train
        self.seed = seed

    def __len__(self) -> int:
        return len(self.sequences)

    def _make_item(self, idx: int, rng: random.Random) -> Dict[str, torch.Tensor]:
        """生成单个样本。"""
        # 截断序列（预留 [CLS] 和 [SEP] 的位置）
        seq = list(self.sequences[idx])[: self.max_len - 2]
        tokens = [self.cls_id] + seq + [self.sep_id]

        input_ids = list(tokens)
        labels = [-100] * len(tokens)  # -100 表示不计算 loss

        # MLM mask
        for i in range(1, len(tokens) - 1):  # 不 mask [CLS] 和 [SEP]
            if rng.random() < self.mlm_prob:
                labels[i] = tokens[i]  # label 记录原始 token

                r = rng.random()
                if r < 0.8:
                    # 80%: 替换为 [MASK]
                    input_ids[i] = self.mask_id
                elif r < 0.9:
                    # 10%: 替换为随机 token
                    input_ids[i] = rng.randint(
                        self.random_token_start, self.vocab_size - 1
                    )
                # 10%: 保持不变（已经计算了 label，所以也会被预测）

        # Padding
        attention_mask = [1] * len(input_ids)
        pad_len = self.max_len - len(input_ids)
        input_ids += [self.pad_id] * pad_len
        attention_mask += [0] * pad_len
        labels += [-100] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.is_train:
            return self._make_item(idx, random)
        else:
            # 验证集使用确定性 mask（相同 idx 总是返回相同结果）
            return self._make_item(idx, random.Random(self.seed + idx))


# =========================================================
# CLS Dataset
# =========================================================
class AppCLSDataset(Dataset):
    """分类任务数据集。

    支持：
    - 样本权重（用于加权损失）
    - 自定义 token ID

    Args:
        sequences: App ID 序列列表
        labels: 标签列表（0/1 二分类）
        weights: 样本权重列表（可选，用于加权损失）
        max_len: 最大序列长度
        pad_id: PAD token ID
        cls_id: CLS token ID
        sep_id: SEP token ID
    """

    def __init__(
        self,
        sequences: List[List[int]],
        labels: List[int],
        weights: Optional[List[float]] = None,
        max_len: int = 200,
        pad_id: int = PAD_ID,
        cls_id: int = CLS_ID,
        sep_id: int = SEP_ID,
    ):
        assert len(sequences) == len(labels), (
            f"sequences 和 labels 长度不一致: {len(sequences)} vs {len(labels)}"
        )
        if weights is not None:
            assert len(sequences) == len(weights), (
                f"sequences 和 weights 长度不一致: {len(sequences)} vs {len(weights)}"
            )

        self.sequences = sequences
        self.labels = labels
        self.weights = weights
        self.max_len = max_len
        self.pad_id = pad_id
        self.cls_id = cls_id
        self.sep_id = sep_id

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = list(self.sequences[idx])[: self.max_len - 2]
        tokens = [self.cls_id] + seq + [self.sep_id]

        attention_mask = [1] * len(tokens)
        pad_len = self.max_len - len(tokens)
        tokens += [self.pad_id] * pad_len
        attention_mask += [0] * pad_len

        item = {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

        if self.weights is not None:
            item["weight"] = torch.tensor(
                self.weights[idx], dtype=torch.float
            )

        return item


# =========================================================
# 加权函数
# =========================================================
# TODO:后续再说吧