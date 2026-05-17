#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: app_pretrain.py
@author: YQ
@date: 2026-05-17
@desc: app MLM 预训练（优化版）

优化点：
  - argparse CLI 覆盖超参，无需改源码
  - 单卡/多卡自动兼容（DDP 按需初始化）
  - Gradient Checkpointing（用重计算换显存）
  - fp16 默认，--bf16 可切换
  - EarlyStoppingCallback（默认 patience=3）
  - 自动断点续训（检测已有 checkpoint）
  - Warmup 步数精确计算并打印
  - DataLoader worker 随机状态隔离（子类覆盖 get_train_dataloader）
  - REPORT_TO 环境变量控制监控后端
  - --dry-run 快速验证链路（只跑 10 步）

用法示例：
  # 单卡
  python app_pretrain.py

  # 多卡
  torchrun --nproc_per_node=4 app_pretrain.py

  # 调参
  python app_pretrain.py --lr 1e-4 --epochs 20 --batch-size 32

  # 切换 bf16 + 禁用 gradient checkpointing
  python app_pretrain.py --bf16 --no-gc

  # 快速链路验证
  python app_pretrain.py --dry-run

  # 接入 TensorBoard
  REPORT_TO=tensorboard python app_pretrain.py
"""

import argparse
import datetime
import os
import random

import polars
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.checkpoint
from datasets import Dataset, load_from_disk
from torch import nn
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Dict, List

from transformers import (
    BertConfig,
    EarlyStoppingCallback,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.configuration_utils import PretrainedConfig
from transformers.trainer_utils import seed_worker


# ─── 特殊 Token ────────────────────────────────────────────────────────────────
PAD_ID, UNK_ID, CLS_ID, SEP_ID, MASK_ID = 0, 1, 2, 3, 4
SPECIAL_TOKEN_NUM = 5

# ─── 默认超参（CLI 可覆盖） ────────────────────────────────────────────────────
VOCAB_SIZE = 10005
MAX_LEN    = 200
HIDDEN     = 256
EMB_SIZE   = 128
LAYERS     = 4
HEADS      = 4
FFN        = 1024

MLM_PROB   = 0.15
PER_DEV_BS = 64
GRAD_ACCUM = 2
EPOCHS     = 10
LR         = 5e-4
SEED       = 42

CACHE_DIR      = "./cache"
CACHE_TRAIN_HF = os.path.join(CACHE_DIR, "train_hf")
CACHE_VAL_HF   = os.path.join(CACHE_DIR, "val_hf")
DATA_PATH      = "./df_app.parquet/df_app.parquet"
TRAIN_RATIO    = 0.7


# ─── 工具函数 ──────────────────────────────────────────────────────────────────
def is_main() -> bool:
    return int(os.environ.get("RANK", 0)) == 0


def log(msg: str) -> None:
    if is_main():
        print(msg, flush=True)


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def worker_init_fn(worker_id: int) -> None:
    """每个 DataLoader worker 独立随机状态，避免多进程 mask 相关性"""
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)


# ─── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="App MLM 预训练")
    p.add_argument("--data-path",    default=DATA_PATH,   help="Parquet 数据路径")
    p.add_argument("--output-dir",   default=None,        help="输出目录（默认带时间戳自动生成）")
    p.add_argument("--epochs",       type=int,   default=EPOCHS)
    p.add_argument("--lr",           type=float, default=LR)
    p.add_argument("--batch-size",   type=int,   default=PER_DEV_BS,  dest="batch_size")
    p.add_argument("--grad-accum",   type=int,   default=GRAD_ACCUM,  dest="grad_accum")
    p.add_argument("--max-len",      type=int,   default=MAX_LEN,     dest="max_len")
    p.add_argument("--hidden",       type=int,   default=HIDDEN)
    p.add_argument("--layers",       type=int,   default=LAYERS)
    p.add_argument("--warmup-ratio", type=float, default=0.05,        dest="warmup_ratio")
    p.add_argument("--patience",     type=int,   default=3,           help="EarlyStopping patience（epoch 数）")
    p.add_argument("--bf16",         action="store_true",             help="使用 bf16（Ampere+ GPU）；默认 fp16")
    p.add_argument("--no-gc",        action="store_true",             help="禁用 gradient checkpointing")
    p.add_argument("--dry-run",      action="store_true",             help="只跑 10 步，验证链路")
    p.add_argument("--seed",         type=int,   default=SEED)
    return p.parse_args()


# ─── Dataset ───────────────────────────────────────────────────────────────────
class AppMLMDataset(Dataset):
    def __init__(
        self,
        sequences:          List[List[int]],
        max_len:            int   = MAX_LEN,
        vocab_size:         int   = VOCAB_SIZE,
        pad_id:             int   = PAD_ID,
        cls_id:             int   = CLS_ID,
        sep_id:             int   = SEP_ID,
        mask_id:            int   = MASK_ID,
        mlm_prob:           float = MLM_PROB,
        random_token_start: int   = SPECIAL_TOKEN_NUM,
        is_train:           bool  = True,
        seed:               int   = SEED,
    ):
        self.sequences          = sequences
        self.max_len            = max_len
        self.vocab_size         = vocab_size
        self.pad_id             = pad_id
        self.cls_id             = cls_id
        self.sep_id             = sep_id
        self.mask_id            = mask_id
        self.mlm_prob           = mlm_prob
        self.random_token_start = random_token_start
        self.is_train           = is_train
        self.seed               = seed

    def __len__(self) -> int:
        return len(self.sequences)

    def _make_item(self, idx: int, rng: random.Random) -> Dict[str, torch.Tensor]:
        seq    = list(self.sequences[idx])[: self.max_len - 2]
        tokens = [self.cls_id] + seq + [self.sep_id]

        input_ids = list(tokens)
        labels    = [-100] * len(tokens)

        for i in range(1, len(tokens) - 1):
            if rng.random() < self.mlm_prob:
                labels[i] = tokens[i]
                r = rng.random()
                if r < 0.8:
                    input_ids[i] = self.mask_id
                elif r < 0.9:
                    input_ids[i] = rng.randint(self.random_token_start, self.vocab_size - 1)

        attention_mask  = [1] * len(input_ids)
        pad_len         = self.max_len - len(input_ids)
        input_ids      += [self.pad_id] * pad_len
        attention_mask += [0]           * pad_len
        labels         += [-100]        * pad_len

        return {
            "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels":         torch.tensor(labels,         dtype=torch.long),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 训练集使用 worker 本地 random（由 worker_init_fn 独立种种）
        # 验证集固定种子，保证每次评估结果完全一致
        if self.is_train:
            return self._make_item(idx, random)
        return self._make_item(idx, random.Random(self.seed + idx))


# ─── Trainer 子类：注入 worker_init_fn ────────────────────────────────────────
class AppMLMTrainer(Trainer):
    """覆盖 get_train_dataloader，为每个 worker 设置独立随机状态"""

    def get_train_dataloader(self) -> DataLoader:
        dl = super().get_train_dataloader()
        return DataLoader(
            dl.dataset,
            batch_size      = dl.batch_sampler.batch_size if hasattr(dl.batch_sampler, "batch_size") else self.args.per_device_train_batch_size,
            sampler         = dl.sampler,
            num_workers     = dl.num_workers,
            collate_fn      = dl.collate_fn,
            pin_memory      = dl.pin_memory,
            worker_init_fn  = worker_init_fn,
            persistent_workers = dl.num_workers > 0,
        )


# ─── RoPE ─────────────────────────────────────────────────────────────────────
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor):
    _batch, _head, seq_len, dim = q.shape
    inv_freq = (
        1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    ).to(q.device)
    t        = torch.arange(seq_len, device=q.device, dtype=torch.float32)
    sinusoid = torch.outer(t, inv_freq)
    sinusoid = torch.cat([sinusoid, sinusoid], dim=-1)
    cos = sinusoid.cos()[None, None].to(q.dtype)
    sin = sinusoid.sin()[None, None].to(q.dtype)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


# ─── 模型组件 ──────────────────────────────────────────────────────────────────
class AppEmbeddings(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.emb_size, padding_idx=config.pad_token_id
        )
        self.LayerNorm = nn.LayerNorm(config.emb_size, eps=config.layer_norm_eps)
        self.dropout   = nn.Dropout(config.hidden_dropout_prob)
        self.proj = (
            nn.Linear(config.emb_size, config.hidden_size)
            if config.emb_size != config.hidden_size
            else None
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        emb = self.word_embeddings(input_ids)
        emb = self.LayerNorm(emb)
        emb = self.dropout(emb)
        if self.proj is not None:
            emb = self.proj(emb)
        return emb


class BertAttentionWithRoPE(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim  = config.hidden_size // self.num_heads
        self.q_proj    = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj    = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj    = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj  = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout   = nn.Dropout(config.attention_probs_dropout_prob)
        self.scale     = self.head_dim ** 0.5

    def forward(
        self,
        hidden_states:  torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S, _ = hidden_states.shape

        def _split(x: torch.Tensor) -> torch.Tensor:
            return x.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        q = _split(self.q_proj(hidden_states))
        k = _split(self.k_proj(hidden_states))
        v = _split(self.v_proj(hidden_states))

        q, k = apply_rope(q, k)

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if attention_mask is not None:
            attn = attn + attention_mask
        attn = self.dropout(F.softmax(attn, dim=-1))

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, -1)
        return self.out_proj(out)


class BertLayerWithRoPE(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.attention    = BertAttentionWithRoPE(config)
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output       = nn.Linear(config.intermediate_size, config.hidden_size)
        self.ln1          = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ln2          = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout      = nn.Dropout(config.hidden_dropout_prob)
        self.act          = nn.GELU()

    def forward(
        self,
        hidden_states:  torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.ln1(
            hidden_states + self.attention(hidden_states, attention_mask)
        )
        hidden_states = self.ln2(
            hidden_states + self.dropout(
                self.output(self.act(self.intermediate(hidden_states)))
            )
        )
        return hidden_states


class BertEncoderWithRoPE(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.layer = nn.ModuleList(
            [BertLayerWithRoPE(config) for _ in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states:  torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple:
        for layer in self.layer:
            if self.gradient_checkpointing and self.training:
                # 用激活重计算换显存：前向不保存中间激活，反向时重新计算
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    use_reentrant=False,  # 更稳定，与 autocast/AMP 兼容
                )
            else:
                hidden_states = layer(hidden_states, attention_mask)
        return (hidden_states,)


# ─── Config ────────────────────────────────────────────────────────────────────
class AppConfig(BertConfig):
    model_type = "app"

    def __init__(
        self,
        vocab_size:                  int   = VOCAB_SIZE,
        emb_size:                    int   = EMB_SIZE,
        hidden_size:                 int   = HIDDEN,
        num_hidden_layers:           int   = LAYERS,
        num_attention_heads:         int   = HEADS,
        intermediate_size:           int   = FFN,
        max_position_embeddings:     int   = MAX_LEN,
        hidden_dropout_prob:         float = 0.1,
        attention_probs_dropout_prob:float = 0.1,
        pad_token_id:                int   = PAD_ID,
        layer_norm_eps:              float = 1e-12,
        initializer_range:           float = 0.02,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            pad_token_id=pad_token_id,
            layer_norm_eps=layer_norm_eps,
            initializer_range=initializer_range,
            **kwargs,
        )
        self.emb_size = emb_size
        self._attn_implementation = "eager"


class ModelConfig(PretrainedConfig):
    model_type = "flow"

    def __init__(
        self,
        app_cfg:      Optional[Dict] = None,
        pad_token_id: int            = PAD_ID,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.app_config   = app_cfg or {}
        self.pad_token_id = pad_token_id


# ─── 模型 ──────────────────────────────────────────────────────────────────────
class AppMLM(PreTrainedModel):
    config_class                    = ModelConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.app_config = (
            AppConfig(**config.app_config)
            if isinstance(config.app_config, dict)
            else config.app_config
        )
        c = self.app_config

        self.app_embeddings = AppEmbeddings(c)
        self.bert           = BertEncoderWithRoPE(c)
        self.mlm_dense      = nn.Linear(c.hidden_size, c.emb_size)
        self.mlm_act        = nn.GELU()
        self.mlm_norm       = nn.LayerNorm(c.emb_size, eps=c.layer_norm_eps)
        self.mlm_bias       = nn.Parameter(torch.zeros(c.vocab_size))

        self.post_init()

    def _set_gradient_checkpointing(self, module: nn.Module, value: bool = False) -> None:
        """HuggingFace Trainer 调用此方法启用/禁用 gradient checkpointing"""
        if isinstance(module, BertEncoderWithRoPE):
            module.gradient_checkpointing = value

    def get_extended_attention_mask(
        self, attention_mask: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        ext = attention_mask[:, None, None, :].to(dtype=dtype)
        return (1.0 - ext) * torch.finfo(dtype).min

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels:         Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        if attention_mask is None:
            attention_mask = (input_ids != self.app_config.pad_token_id).long()

        dtype    = self.app_embeddings.word_embeddings.weight.dtype
        ext_mask = self.get_extended_attention_mask(attention_mask, dtype)

        seq_out = self.bert(self.app_embeddings(input_ids), attention_mask=ext_mask)[0]

        x     = self.mlm_norm(self.mlm_act(self.mlm_dense(seq_out)))
        emb_w = self.app_embeddings.word_embeddings.weight

        if labels is not None:
            active        = labels.view(-1) != -100
            logits_active = x.view(-1, x.size(-1))[active] @ emb_w.t() + self.mlm_bias
            loss          = F.cross_entropy(logits_active, labels.view(-1)[active])
            return {"loss": loss}

        return {"logits": x @ emb_w.t() + self.mlm_bias}


# ─── 数据准备 ──────────────────────────────────────────────────────────────────
def prepare_on_rank0(data_path: str, seed: int) -> None:
    if all(os.path.exists(p) for p in [CACHE_TRAIN_HF, CACHE_VAL_HF]):
        log("缓存已存在，跳过预处理")
        return

    log(f"rank0 开始读取数据：{data_path}")
    df       = polars.read_parquet(data_path)
    shuffled = df.sample(fraction=1.0, seed=seed)
    n        = shuffled.height
    cut      = int(n * TRAIN_RATIO)

    train_seqs = shuffled.head(cut).get_column("app_name_encoded").to_list()
    val_seqs   = shuffled.tail(n - cut).get_column("app_name_encoded").to_list()

    Dataset.from_dict({"sequence": train_seqs}).save_to_disk(CACHE_TRAIN_HF)
    Dataset.from_dict({"sequence": val_seqs}).save_to_disk(CACHE_VAL_HF)
    log(f"数据缓存完成：train={len(train_seqs):,}  val={len(val_seqs):,}")


def load_prepared():
    barrier()
    train_ds = load_from_disk(CACHE_TRAIN_HF)
    val_ds   = load_from_disk(CACHE_VAL_HF)
    return train_ds["sequence"], val_ds["sequence"]


# ─── TrainingArguments 工厂 ────────────────────────────────────────────────────
def make_training_args(
    cfg:          argparse.Namespace,
    output_dir:   str,
    warmup_steps: int,
) -> TrainingArguments:
    use_fp16   = not cfg.bf16
    use_bf16   = cfg.bf16
    report_to  = os.environ.get("REPORT_TO", "none")  # 可设 "tensorboard" 或 "wandb"
    enable_gc  = not cfg.no_gc

    return TrainingArguments(
        output_dir                  = output_dir,
        seed                        = cfg.seed,

        num_train_epochs            = cfg.epochs,
        per_device_train_batch_size = cfg.batch_size,
        per_device_eval_batch_size  = cfg.batch_size,
        gradient_accumulation_steps = cfg.grad_accum,

        learning_rate               = cfg.lr,
        weight_decay                = 0.01,
        warmup_steps                = warmup_steps,      # 精确步数，优于 ratio
        lr_scheduler_type           = "cosine",
        max_grad_norm               = 1.0,

        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
        save_total_limit            = 3,

        gradient_checkpointing      = enable_gc,

        ddp_find_unused_parameters  = False,
        dataloader_num_workers      = 4,
        dataloader_pin_memory       = True,

        fp16                        = use_fp16,
        bf16                        = use_bf16,

        logging_dir                 = os.path.join(output_dir, "logs"),
        logging_strategy            = "steps",
        logging_steps               = 20,               # 2w 数据更细粒度
        report_to                   = report_to,

        max_steps                   = 10 if cfg.dry_run else -1,
    )


# ─── 主训练入口 ────────────────────────────────────────────────────────────────
def main() -> None:
    cfg        = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank       = int(os.environ.get("RANK",       0))
    world_size = int(os.environ.get("WORLD_SIZE",  1))

    set_seed(cfg.seed)

    output_dir = cfg.output_dir or \
        f"./ckpt_app_mlm_{int(datetime.datetime.now().timestamp() * 1000)}"

    if rank == 0:
        print("=" * 70, flush=True)
        print(f"WORLD_SIZE={world_size}  RANK={rank}  LOCAL_RANK={local_rank}", flush=True)
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {p.name}  ({p.total_memory / 1e9:.1f} GB)", flush=True)
        print(
            f"fp16={not cfg.bf16}  bf16={cfg.bf16}  "
            f"gradient_checkpointing={not cfg.no_gc}  dry_run={cfg.dry_run}",
            flush=True,
        )
        print("=" * 70, flush=True)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(CACHE_DIR,  exist_ok=True)
        prepare_on_rank0(cfg.data_path, cfg.seed)

    # 多卡才初始化进程组，单卡直接跳过，避免挂起
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    barrier()

    train_seqs, val_seqs = load_prepared()
    log(f"加载完成：train={len(train_seqs):,}  val={len(val_seqs):,}")

    train_ds = AppMLMDataset(train_seqs, max_len=cfg.max_len, is_train=True,  seed=cfg.seed)
    val_ds   = AppMLMDataset(val_seqs,   max_len=cfg.max_len, is_train=False, seed=cfg.seed)

    # 精确计算 warmup steps 并打印，方便复现
    steps_per_epoch = max(1, len(train_seqs) // (cfg.batch_size * world_size * cfg.grad_accum))
    total_steps     = steps_per_epoch * (10 if cfg.dry_run else cfg.epochs)
    warmup_steps    = int(total_steps * cfg.warmup_ratio)
    log(
        f"steps_per_epoch={steps_per_epoch}  total_steps={total_steps}  "
        f"warmup_steps={warmup_steps}  warmup_ratio={cfg.warmup_ratio}"
    )

    app_cfg   = AppConfig(
        vocab_size           = VOCAB_SIZE,
        hidden_size          = cfg.hidden,
        emb_size             = EMB_SIZE,
        num_hidden_layers    = cfg.layers,
        num_attention_heads  = HEADS,
        intermediate_size    = FFN,
        max_position_embeddings = cfg.max_len,
        pad_token_id         = PAD_ID,
    )
    model = AppMLM(ModelConfig(app_cfg=app_cfg.to_dict(), pad_token_id=PAD_ID))

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        log(f"总参数：{n_params / 1e6:.2f}M")
        log(
            f"等效 batch = {cfg.batch_size} × {world_size} × {cfg.grad_accum} "
            f"= {cfg.batch_size * world_size * cfg.grad_accum}"
        )

    # 断点续训：自动检测已有 checkpoint
    resume_from: Optional[str] = None
    if os.path.isdir(output_dir):
        ckpts = sorted(
            [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[-1]),
        )
        if ckpts:
            resume_from = os.path.join(output_dir, ckpts[-1])
            log(f"发现已有 checkpoint，从 {resume_from} 续训")

    trainer = AppMLMTrainer(
        model         = model,
        args          = make_training_args(cfg, output_dir, warmup_steps),
        train_dataset = train_ds,
        eval_dataset  = val_ds,
        callbacks     = [EarlyStoppingCallback(early_stopping_patience=cfg.patience)],
    )

    trainer.train(resume_from_checkpoint=resume_from)

    if rank == 0:
        save_path = os.path.join(output_dir, "final")
        trainer.save_model(save_path)
        log(f"模型已保存至 {save_path}")

    barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
