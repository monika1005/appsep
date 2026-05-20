#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: config.py
@author: YQ
@date: 2026-05-20
@desc: 

支持两种使用方式：
1. 直接修改默认参数
2. 从命令行传入：python pretrain.py --lr 1e-4 --epochs 5
"""

import argparse
import os
from dataclasses import dataclass, field
from typing import Literal



PAD_ID = 0
UNK_ID = 1
CLS_ID = 2
SEP_ID = 3
MASK_ID = 4
SPECIAL_TOKEN_NUM = 5

SPECIAL_TOKENS = {
    "[PAD]": PAD_ID,
    "[UNK]": UNK_ID,
    "[CLS]": CLS_ID,
    "[SEP]": SEP_ID,
    "[MASK]": MASK_ID,
}


@dataclass
class PretrainConfig:
    # 模型架构
    vocab_size: int = 10005
    max_len: int = 200
    hidden_size: int = 256
    emb_size: int = 128
    layers: int = 4
    heads: int = 4
    ffn: int = 1024

    # 训练参数
    mlm_prob: float = 0.15
    per_dev_bs: int = 64
    grad_accum: int = 2
    epochs: int = 10
    lr: float = 5e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0

    # 数据
    seed: int = 42
    train_ratio: float = 0.7

    # 路径
    data_path: str = "./df_app.parquet/df_app.parquet"
    output_dir: str = "./ckpt_app_mlm"
    cache_dir: str = "./cache"

    # 训练策略
    use_gradient_checkpointing: bool = False

    def __post_init__(self):
        self.output_dir = self.output_dir.rstrip("/")

    @property
    def cache_train_hf(self) -> str:
        return os.path.join(self.cache_dir, "train_hf")

    @property
    def cache_val_hf(self) -> str:
        return os.path.join(self.cache_dir, "val_hf")

    @property
    def effective_batch_size(self) -> int:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        return self.per_dev_bs * world_size * self.grad_accum



@dataclass
class CLSConfig:
    # 模型架构（与预训练共享）
    vocab_size: int = 10005
    max_len: int = 200
    hidden_size: int = 256
    emb_size: int = 128
    layers: int = 4
    heads: int = 4
    ffn: int = 1024
    dropout: float = 0.2

    # 分类任务
    num_labels: int = 2
    label_col: str = "dpd5_ever"

    # 训练参数
    per_dev_bs: int = 128
    grad_accum: int = 2
    epochs: int = 5
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # 数据
    seed: int = 42
    oot_cutoff: str = "2026-03-01"
    early_stopping_patience: int = 2

    # 路径
    data_path: str = "./df_app.parquet"
    label_path: str = "./df_y.parquet"
    pretrained_model: str = "./ckpt_app_mlm/final"
    output_dir: str = "./ckpt_app_cls"
    cache_dir: str = "./cache"

    # 训练策略
    use_gradient_checkpointing: bool = False
    freeze_backbone: bool = False
    freeze_epochs: int = 2

    # 加权策略
    use_sample_weight: bool = True
    use_time_weight: bool = True
    min_app_count: int = 200

    def __post_init__(self):
        self.output_dir = self.output_dir.rstrip("/")

    @property
    def cache_train(self) -> str:
        return os.path.join(self.cache_dir, "train_cls.pkl")

    @property
    def cache_oot(self) -> str:
        return os.path.join(self.cache_dir, "oot_cls.pkl")

    @property
    def effective_batch_size(self) -> int:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        return self.per_dev_bs * world_size * self.grad_accum


@dataclass
class DataProcessConfig:
    data_dir: str = "./yq1005"
    min_app_count: int = 200
    max_vocab_size: int = 10000
    vocab_save_path: str = "app2id.yaml"
    output_parquet_path: str = "./yq1005/df_app.parquet"

    input_col: str = "app_name"
    output_col: str = "app_name_encoded"
    sort_key: str = "lastUpdateTime"
    sort_desc: bool = True
    extract_key: str = "appName"

    extra_sources: list = field(default_factory=list)


def make_argparser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(description="AppSeq 预训练/微调")

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # ---------- 预训练 ----------
    p_pretrain = subparsers.add_parser("pretrain", help="MLM 预训练")
    p_pretrain.add_argument("--lr", type=float, default=None)
    p_pretrain.add_argument("--epochs", type=int, default=None)
    p_pretrain.add_argument("--bs", type=int, dest="per_dev_bs", default=None)
    p_pretrain.add_argument("--data", type=str, default=None)
    p_pretrain.add_argument("--output", type=str, default=None)

    # ---------- 微调 ----------
    p_cls = subparsers.add_parser("finetune", help="分类微调")
    p_cls.add_argument("--lr", type=float, default=None)
    p_cls.add_argument("--epochs", type=int, default=None)
    p_cls.add_argument("--bs", type=int, dest="per_dev_bs", default=None)
    p_cls.add_argument("--pretrained", type=str, default=None)
    p_cls.add_argument("--output", type=str, default=None)
    p_cls.add_argument("--oot-cutoff", type=str, default=None)

    return parser


def apply_cli_args(config, args) -> None:
    """将命令行参数应用到配置对象"""
    if args is None:
        return

    for key in ["lr", "epochs", "per_dev_bs"]:
        if getattr(args, key, None) is not None:
            setattr(config, key, getattr(args, key))

    if getattr(args, "data", None):
        config.data_path = args.data

    if getattr(args, "output", None):
        config.output_dir = args.output

    if getattr(args, "pretrained", None):
        config.pretrained_model = args.pretrained

    if getattr(args, "oot_cutoff", None):
        config.oot_cutoff = args.oot_cutoff


pretrain_cfg = PretrainConfig()
cls_cfg = CLSConfig()
data_cfg = DataProcessConfig()
