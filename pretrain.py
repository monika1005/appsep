#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: pretrain.py
@author: YQ
@date: 2026-05-20
@desc: 
"""

import datetime
import os
import random

import numpy as np
import polars as pl
import torch
import torch.distributed as dist
from datasets import Dataset, load_from_disk
from torch.utils.data import Dataset as TorchDataset
from transformers import TrainingArguments

from appsep import AppConfig, AppMLM, ModelConfig, AppMLMDataset
from appsep.config import PretrainConfig, make_argparser, apply_cli_args
from appsep.utils.trainer import create_trainer


# ============================================================
# 工具函数
# =========================================================
def is_main() -> bool:
    return int(os.environ.get("RANK", 0)) == 0


def log(msg: str) -> None:
    if is_main():
        print(msg, flush=True)


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def print_system_info() -> None:
    """打印系统信息（仅 rank 0）。"""
    if not is_main():
        return

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    print("=" * 60, flush=True)
    print(f"WORLD_SIZE={world_size}  RANK={rank}  LOCAL_RANK={local_rank}", flush=True)

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(
                f"GPU {i}: {props.name}  "
                f"({props.total_memory / 1e9:.1f} GB)",
                flush=True,
            )

    print("=" * 60, flush=True)


# ============================================================
# 数据准备
# ============================================================
def prepare_data(cfg: PretrainConfig) -> None:
    """在 rank 0 上准备数据（pickle 缓存）。"""
    cache_train = cfg.cache_train_hf
    cache_val = cfg.cache_val_hf

    if all(os.path.exists(p) for p in [cache_train, cache_val]):
        log("缓存已存在，跳过预处理")
        return

    log("rank0 开始读取并处理数据...")

    df = pl.read_parquet(cfg.data_path)
    shuffled = df.sample(fraction=1.0, seed=cfg.seed)
    n = shuffled.height
    cut = int(n * cfg.train_ratio)

    train_seqs = shuffled.head(cut).get_column("app_name_encoded").to_list()
    val_seqs = shuffled.tail(n - cut).get_column("app_name_encoded").to_list()

    # 使用 HuggingFace Dataset 格式保存（支持 memory mapping）
    train_ds = Dataset.from_dict({"sequence": train_seqs})
    val_ds = Dataset.from_dict({"sequence": val_seqs})

    os.makedirs(cfg.cache_dir, exist_ok=True)
    train_ds.save_to_disk(cache_train)
    val_ds.save_to_disk(cache_val)

    log(f"数据缓存完成: train={len(train_seqs):,}, val={len(val_seqs):,}")


def load_data(cfg: PretrainConfig) -> tuple:
    """所有 rank 加载数据。"""
    barrier()  # 等 rank 0 完成保存

    train_ds_hf = load_from_disk(cfg.cache_train_hf)
    val_ds_hf = load_from_disk(cfg.cache_val_hf)

    train_seqs = train_ds_hf["sequence"]
    val_seqs = val_ds_hf["sequence"]

    return train_seqs, val_seqs


# ============================================================
# 训练
# =========================================================
def make_training_args(cfg: PretrainConfig) -> TrainingArguments:
    """构建训练参数。"""
    # 生成带时间戳的 output_dir
    if not cfg.output_dir or cfg.output_dir == "./ckpt_app_mlm":
        ts = int(datetime.datetime.now().timestamp() * 1000)
        cfg.output_dir = f"./ckpt_app_mlm_{ts}"

    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, "logs"), exist_ok=True)

    return TrainingArguments(
        output_dir=cfg.output_dir,
        seed=cfg.seed,

        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.per_dev_bs,
        per_device_eval_batch_size=cfg.per_dev_bs,
        gradient_accumulation_steps=cfg.grad_accum,

        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type="cosine",
        max_grad_norm=cfg.max_grad_norm,

        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,

        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,

        fp16=torch.cuda.is_available(),
        bf16=False,

        logging_dir=os.path.join(cfg.output_dir, "logs"),
        logging_strategy="steps",
        logging_steps=50,
        report_to="none",
    )


def count_params(model) -> None:
    """打印模型参数统计。"""
    if not is_main():
        return

    n = sum(p.numel() for p in model.parameters())
    nt = sum(p.numel() for p in model.parameters() if p.requires_grad)

    log(f"总参数: {n / 1e6:.2f}M")
    log(f"可训练: {nt / 1e6:.2f}M")


# ============================================================
# Main
# =========================================================
def main():
    # 解析命令行参数
    parser = make_argparser()
    args_cli = parser.parse_args()

    # 加载配置
    cfg = PretrainConfig()

    # 应用命令行参数
    if args_cli.command == "pretrain":
        apply_cli_args(cfg, args_cli)

    # 初始化分布式
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # DDP 初始化
    if world_size > 1:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        dist.init_process_group(backend="nccl")

    print_system_info()

    # 数据准备（仅 rank 0）
    if rank == 0:
        os.makedirs(cfg.cache_dir, exist_ok=True)
        prepare_data(cfg)

    barrier()

    # 加载数据
    train_seqs, val_seqs = load_data(cfg)
    log(f"数据加载完成: train={len(train_seqs):,}, val={len(val_seqs):,}")

    # 构建 Dataset
    train_ds = AppMLMDataset(
        sequences=train_seqs,
        is_train=True,
        mlm_prob=cfg.mlm_prob,
    )
    val_ds = AppMLMDataset(
        sequences=val_seqs,
        is_train=False,
        seed=cfg.seed,
    )

    # 构建模型
    app_cfg = AppConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        emb_size=cfg.emb_size,
        num_hidden_layers=cfg.layers,
        num_attention_heads=cfg.heads,
        intermediate_size=cfg.ffn,
        max_position_embeddings=cfg.max_len,
        pad_token_id=0,
    )
    model_cfg = ModelConfig(app_cfg=app_cfg.to_dict())
    model = AppMLM(model_cfg)

    if is_main():
        count_params(model)
        log(f"有效 batch size = {cfg.per_dev_bs} × {world_size} × {cfg.grad_accum} "
            f"= {cfg.effective_batch_size}")

    # 梯度检查点
    if cfg.use_gradient_checkpointing:
        model.enable_gradient_checkpointing()
        log("已启用梯度检查点")

    # 训练
    train_args = make_training_args(cfg)

    trainer = create_trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    trainer.train()

    # 保存最终模型
    if is_main():
        save_path = os.path.join(cfg.output_dir, "final")
        trainer.save_model(save_path)
        log(f"模型已保存至 {save_path}")

    barrier()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
