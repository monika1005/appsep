#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@file: finetune.py
@author: YQ
@date: 2026-05-20
@desc: 
"""

import os
import pickle
from typing import Optional

import numpy as np
import polars as pl
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from transformers import TrainingArguments, EarlyStoppingCallback

from appsep import AppCLS, CLSModelConfig, AppCLSDataset, compute_cls_metrics
from appsep.config import CLSConfig, PretrainConfig, make_argparser, apply_cli_args
from appsep.utils.trainer import create_trainer
from appsep.utils.metrics import _probs_preds_from_logits, compute_ks_statistic
from appsep.data.dataset import prepare_cls_dataframe


def is_main() -> bool:
    return int(os.environ.get("RANK", 0)) == 0


def log(msg: str) -> None:
    if is_main():
        print(msg, flush=True)


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

def prepare_data(cfg: CLSConfig) -> None:
    """在 rank 0 上准备分类数据（pickle 缓存）。"""
    cache_train = cfg.cache_train
    cache_oot = cfg.cache_oot

    if all(os.path.exists(p) for p in [cache_train, cache_oot]):
        log("缓存已存在，跳过预处理")
        return

    log("rank0 开始准备分类数据...")

    # 读取数据
    df_app = pl.read_parquet(cfg.data_path)
    df_y = pl.read_parquet(cfg.label_path)

    # 合并
    df = df_y.join(
        df_app,
        left_on="order_id",
        right_on="apply_no",
        how="inner",
    )
    df = df.filter(pl.col(cfg.label_col).is_in([0, 1]))

    # 时间切分 + 加权
    train_df, oot_df = prepare_cls_dataframe(
        df,
        oot_cutoff=cfg.oot_cutoff,
        label_col=cfg.label_col,
        use_sample_weight=cfg.use_sample_weight,
        use_time_weight=cfg.use_time_weight,
    )

    # 提取数据
    train_seqs = train_df["app_name_encoded"].to_list()
    train_labels = train_df[cfg.label_col].to_list()
    train_weights = train_df["set_wgt_time"].to_list() if cfg.use_sample_weight else None

    oot_seqs = oot_df["app_name_encoded"].to_list()
    oot_labels = oot_df[cfg.label_col].to_list()

    # 保存缓存
    os.makedirs(cfg.cache_dir, exist_ok=True)

    with open(cache_train, "wb") as f:
        pickle.dump((train_seqs, train_labels, train_weights), f)
    with open(cache_oot, "wb") as f:
        pickle.dump((oot_seqs, oot_labels), f)

    pos_rate_train = np.mean(train_labels) * 100
    pos_rate_oot = np.mean(oot_labels) * 100

    log(f"数据缓存完成: train={len(train_seqs):,}, oot={len(oot_seqs):,}")
    log(f"train 正样本率: {pos_rate_train:.2f}%")
    log(f"oot 正样本率: {pos_rate_oot:.2f}%")


def load_data(cfg: CLSConfig) -> tuple:
    """加载缓存的数据。"""
    with open(cfg.cache_train, "rb") as f:
        train_seqs, train_labels, train_weights = pickle.load(f)

    with open(cfg.cache_oot, "rb") as f:
        oot_seqs, oot_labels = pickle.load(f)

    return (train_seqs, train_labels, train_weights), (oot_seqs, oot_labels)


@torch.no_grad()
def predict_oot(
    model: torch.nn.Module,
    oot_ds: TorchDataset,
    batch_size: int,
) -> tuple:
    """手动推理 OOT，避免 DDP trainer.predict() 的 NCCL 超时问题。

    Args:
        model: 微调后的模型
        oot_ds: OOT 数据集
        batch_size: 推理 batch size

    Returns:
        (probs, labels)
    """
    device = next(model.parameters()).device
    model.eval()

    loader = DataLoader(
        oot_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    all_probs = []
    all_labels = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].numpy()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"].detach().cpu().numpy()

        probs, _ = _probs_preds_from_logits(logits)

        all_probs.append(probs)
        all_labels.append(labels)

    return np.concatenate(all_probs), np.concatenate(all_labels)


def evaluate_oot(
    model: torch.nn.Module,
    oot_ds: TorchDataset,
    oot_labels: list,
    trainer,
    cfg: CLSConfig,
) -> None:
    """在 OOT 数据集上评估并打印指标。"""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")

    probs, labels = predict_oot(model, oot_ds, batch_size=cfg.per_dev_bs * 2)
    probs = np.array(probs)
    labels = np.array(labels)

    auc = compute_cls_metrics((probs, labels))["auc"]
    ks = compute_ks_statistic(probs, labels)

    log(f"\n{'='*40}")
    log(f"最终 OOT 评估:")
    log(f"  AUC = {auc:.4f}")
    log(f"  KS  = {ks:.4f}")
    log(f"{'='*40}")

    # 保存预测结果
    save_path = os.path.join(cfg.output_dir, "final")
    trainer.save_model(save_path)

    # 保存预测分数（用于后续分析）
    import pandas as pd
    score_df = pd.DataFrame({
        "probs": probs,
        "labels": labels,
    })
    score_df.to_parquet(os.path.join(save_path, "oot_scores.parquet"), index=False)
    log(f"预测分数已保存: {save_path}/oot_scores.parquet")

    log(f"\n模型已保存至 {save_path}")


def make_training_args(cfg: CLSConfig) -> TrainingArguments:
    """构建训练参数。"""
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
        metric_for_best_model="eval_auc",
        greater_is_better=True,
        save_total_limit=2,

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


def main():
    # 解析命令行参数
    parser = make_argparser()
    args_cli = parser.parse_args()

    # 加载配置
    cfg = CLSConfig()

    # 应用命令行参数
    if args_cli.command == "finetune":
        apply_cli_args(cfg, args_cli)

    # 初始化
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

    # 数据准备（仅 rank 0）
    if rank == 0:
        os.makedirs(cfg.cache_dir, exist_ok=True)
        prepare_data(cfg)

    barrier()

    # 加载数据
    (train_seqs, train_labels, train_weights), (oot_seqs, oot_labels) = load_data(cfg)
    log(f"数据加载完成: train={len(train_seqs):,}, oot={len(oot_seqs):,}")

    # 构建 Dataset
    train_ds = AppCLSDataset(
        sequences=train_seqs,
        labels=train_labels,
        weights=train_weights,
    )
    oot_ds = AppCLSDataset(
        sequences=oot_seqs,
        labels=oot_labels,
    )

    # 构建模型
    from appsep import AppConfig
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
    cls_cfg = CLSModelConfig(
        app_cfg=app_cfg.to_dict(),
        num_labels=cfg.num_labels,
        dropout=cfg.dropout,
    )
    model = AppCLS(cls_cfg)

    # 加载预训练 backbone
    if os.path.exists(cfg.pretrained_model):
        model.load_pretrained_backbone(cfg.pretrained_model)
    else:
        log(f"预训练模型不存在: {cfg.pretrained_model}，从头训练")

    if is_main():
        n = model.total_params()
        nt = model.trainable_params()
        log(f"总参数: {n / 1e6:.2f}M, 可训练: {nt / 1e6:.2f}M")

    # 梯度检查点
    if cfg.use_gradient_checkpointing:
        model.enable_gradient_checkpointing()
        log("已启用梯度检查点")

    # 分阶段微调：先冻结 backbone
    if cfg.freeze_backbone:
        model.freeze_backbone()
        log(f"冻结 backbone，训练 {cfg.freeze_epochs} 个 epoch")

    # 训练
    train_args = make_training_args(cfg)

    trainer = create_trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=oot_ds,
        compute_metrics=compute_cls_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)],
    )

    trainer.train()

    # 最终 OOT 评估
    if rank == 0:
        evaluate_oot(model, oot_ds, oot_labels, trainer, cfg)

    barrier()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
