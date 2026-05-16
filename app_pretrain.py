#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: app_data.py
@author: YQ
@date: 2026-05-15
@desc: app MLM预训练
"""

import datetime
import os
import random
import pickle
import time
import polars



import torch
from torch import nn
from datasets import Dataset
import torch.nn.functional as F
import torch.distributed as dist
from datasets import load_from_disk
from torch.utils.data import Dataset

from typing import Optional, Dict, List

from transformers import (
    Trainer,
    TrainingArguments,
    set_seed,
    BertConfig,
    PreTrainedModel,
    EarlyStoppingCallback,
)
from transformers.configuration_utils import PretrainedConfig


PAD_ID, UNK_ID, CLS_ID, SEP_ID, MASK_ID = 0, 1, 2, 3, 4
SPECIAL_TOKEN_NUM = 5

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

OUTPUT_DIR      = f"./ckpt_app_mlm_{int(datetime.datetime.now().timestamp()*1000)}"
CACHE_DIR       = "./cache"
CACHE_TRAIN_HF  = os.path.join(CACHE_DIR, "train_hf")
CACHE_VAL_HF   = os.path.join(CACHE_DIR, "val_hf")

DATA_PATH   = "./df_app.parquet/df_app.parquet"
TRAIN_RATIO = 0.7


def is_main():
    return int(os.environ.get("RANK", 0)) == 0

def log(msg: str):
    if is_main():
        print(msg, flush=True)

def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()



class AppMLMDataset(Dataset):
    def __init__(
        self,
        sequences: List[List[int]],
        max_len: int            = MAX_LEN,
        vocab_size: int         = VOCAB_SIZE,
        pad_id: int             = PAD_ID,
        cls_id: int             = CLS_ID,
        sep_id: int             = SEP_ID,
        mask_id: int            = MASK_ID,
        mlm_prob: float         = MLM_PROB,
        random_token_start: int = SPECIAL_TOKEN_NUM,
        is_train: bool          = True,
        seed: int               = SEED,
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
                    input_ids[i] = rng.randint(
                        self.random_token_start, self.vocab_size - 1
                    )

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
        if self.is_train:
            return self._make_item(idx, random)
        
        return self._make_item(idx, random.Random(self.seed + idx))



def rotate_half(x: torch.Tensor):
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

    def forward(self, input_ids: torch.Tensor):
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
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) :
        B, S, _ = hidden_states.shape

        def _split(x):
            return x.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = _split(self.q_proj(hidden_states)), \
                  _split(self.k_proj(hidden_states)), \
                  _split(self.v_proj(hidden_states))

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
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) :
        hidden_states = self.ln1(hidden_states + self.attention(hidden_states, attention_mask))
        hidden_states = self.ln2(hidden_states + self.dropout(
            self.output(self.act(self.intermediate(hidden_states)))
        ))
        return hidden_states


class BertEncoderWithRoPE(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.layer = nn.ModuleList(
            [BertLayerWithRoPE(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple:
        for layer in self.layer:
            hidden_states = layer(hidden_states, attention_mask)
        return (hidden_states,)


class AppConfig(BertConfig):
    model_type = "app"

    def __init__(
        self,
        vocab_size: int   = VOCAB_SIZE,
        emb_size: int     = EMB_SIZE,
        hidden_size: int  = HIDDEN,
        num_hidden_layers: int       = LAYERS,
        num_attention_heads: int     = HEADS,
        intermediate_size: int       = FFN,
        max_position_embeddings: int = MAX_LEN,
        hidden_dropout_prob: float             = 0.1,
        attention_probs_dropout_prob: float    = 0.1,
        pad_token_id: int   = PAD_ID,
        layer_norm_eps: float      = 1e-12,
        initializer_range: float   = 0.02,
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
        app_cfg: Optional[Dict] = None,
        pad_token_id: int       = PAD_ID,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.app_config  = app_cfg or {}
        self.pad_token_id = pad_token_id


class AppMLM(PreTrainedModel):
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

        self.app_embeddings = AppEmbeddings(c)
        self.bert           = BertEncoderWithRoPE(c)
        self.mlm_dense      = nn.Linear(c.hidden_size, c.emb_size)
        self.mlm_act        = nn.GELU()
        self.mlm_norm       = nn.LayerNorm(c.emb_size, eps=c.layer_norm_eps)
        self.mlm_bias       = nn.Parameter(torch.zeros(c.vocab_size))

        self.post_init()

    def get_extended_attention_mask(
        self, attention_mask: torch.Tensor, dtype: torch.dtype
    ) :
        ext = attention_mask[:, None, None, :].to(dtype=dtype)
        return (1.0 - ext) * torch.finfo(dtype).min

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor]         = None,
        **kwargs,
    ):
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


def make_args():
    return TrainingArguments(
        output_dir    = OUTPUT_DIR,
        seed          = SEED,

   
        num_train_epochs            = EPOCHS,
        per_device_train_batch_size = PER_DEV_BS,
        per_device_eval_batch_size  = PER_DEV_BS,
        gradient_accumulation_steps = GRAD_ACCUM,

        # 优化器
        learning_rate     = LR,
        weight_decay      = 0.01,
        warmup_ratio      = 0.05,
        lr_scheduler_type = "cosine",
        max_grad_norm     = 1.0,

        eval_strategy          = "epoch",
        save_strategy          = "epoch",
        load_best_model_at_end = True,
        metric_for_best_model  = "eval_loss",
        greater_is_better      = False,
        save_total_limit       = 3,

        ddp_find_unused_parameters = False,
        dataloader_num_workers     = 4,
        dataloader_pin_memory      = True,

        fp16 = True,
        bf16 = False,

        logging_dir      = os.path.join(OUTPUT_DIR, "logs"),
        logging_strategy = "steps",
        logging_steps    = 50,
        report_to        = "none",
    )



def prepare_on_rank0() -> None:
    if all(os.path.exists(p) for p in [CACHE_TRAIN_HF, CACHE_VAL_HF]):
        log("缓存已存在，跳过预处理")
        return

    log("rank0 开始读取并处理数据...")
    df      = polars.read_parquet(DATA_PATH)
    shuffled = df.sample(fraction=1.0, seed=SEED)
    n        = shuffled.height
    cut      = int(n * TRAIN_RATIO)

    train_seqs = shuffled.head(cut).get_column("app_name_encoded").to_list()
    val_seqs   = shuffled.tail(n - cut).get_column("app_name_encoded").to_list()

    train_ds = Dataset.from_dict({"sequence": train_seqs})
    val_ds   = Dataset.from_dict({"sequence": val_seqs})

    train_ds.save_to_disk(CACHE_TRAIN_HF)
    val_ds.save_to_disk(CACHE_VAL_HF)

    log(f"数据缓存完成: train={len(train_seqs):,}, val={len(val_seqs):,}")


def load_prepared():
    
    # 所有进程都要等 rank0 完成数据保存后再进行
    barrier()
    train_ds = load_from_disk(CACHE_TRAIN_HF)
    val_ds   = load_from_disk(CACHE_VAL_HF)
    return train_ds["sequence"], val_ds["sequence"]


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank  = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    set_seed(SEED)

    if rank == 0:
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++", flush=True)
        print(f"WORLD_SIZE={world_size}  RANK={rank}  LOCAL_RANK={local_rank}",
              flush=True)
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {p.name}  ({p.total_memory/1e9:.1f} GB)", flush=True)
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++", flush=True)

    if rank == 0:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(CACHE_DIR,  exist_ok=True)
        prepare_on_rank0()
        
    dist.init_process_group(backend="nccl")
    barrier()


    train_seqs, val_seqs = load_prepared()
    log(f"加载完成: train={len(train_seqs):,}  val={len(val_seqs):,}")

    train_ds = AppMLMDataset(train_seqs, is_train=True)
    val_ds   = AppMLMDataset(val_seqs,   is_train=False, seed=SEED)

    app_cfg   = AppConfig(
        vocab_size=VOCAB_SIZE, hidden_size=HIDDEN, emb_size=EMB_SIZE,
        num_hidden_layers=LAYERS, num_attention_heads=HEADS,
        intermediate_size=FFN, max_position_embeddings=MAX_LEN,
        pad_token_id=PAD_ID,
    )
    model_cfg = ModelConfig(app_cfg=app_cfg.to_dict(), pad_token_id=PAD_ID)
    model     = AppMLM(model_cfg)

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        log(f"总参数: {n_params/1e6:.2f}M")
        log(f"等效 batch = {PER_DEV_BS} × {world_size} × {GRAD_ACCUM} "
            f"= {PER_DEV_BS * world_size * GRAD_ACCUM}")

    trainer = Trainer(
        model         = model,
        args          = make_args(),
        train_dataset = train_ds,
        eval_dataset  = val_ds,
    )

    trainer.train()

    if rank == 0:
        save_path = os.path.join(OUTPUT_DIR, "final")
        trainer.save_model(save_path)
        log(f"模型已保存至 {save_path}")

    barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
