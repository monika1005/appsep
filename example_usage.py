#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
appsep 详细用法示例

运行前请确保：
  pip install polars pyyaml torch transformers datasets

目录结构：
  yq1005/
    app_df_ok.parquet
    app_df_em.parquet
"""

import os
import yaml
import torch
import polars
import random


# ─────────────────────────────────────────────
# Part 1  数据预处理
# ─────────────────────────────────────────────

def demo_data_process():
    """基础用法：读取数据 → 构建词表 → 序列编码 → 保存结果"""
    from data_process import AppSeqProcess

    processor = AppSeqProcess(
        data_dir             = "./yq1005",           # 原始 Parquet 所在目录
        min_app_count        = 200,                  # 词表最低频次
        max_vocab_size       = 10000,                # 词表上限（不含特殊 Token）
        vocab_save_path      = "app2id.yaml",        # 词表输出路径
        output_parquet_path  = "./yq1005/df_app.parquet",
        input_col            = "app_name",           # 原始 JSON 列名
        output_col           = "app_name_encoded",   # 编码结果列名
        sort_key             = "lastUpdateTime",     # 序列内部排序字段
        sort_desc            = True,                 # True = 最新 App 排最前
        extract_key          = "appName",            # JSON 中提取的字段
    )

    df_result = processor.main()
    print("处理完成，列：", df_result.columns)
    print(df_result.head(3))


def demo_data_process_with_extra():
    """带额外数据源：合并多个 Parquet 文件"""
    from data_process import AppSeqProcess

    AppSeqProcess(
        data_dir            = "./yq1005",
        extra_sources       = ["./extra/supplement.parquet"],  # 追加数据源
        min_app_count       = 100,    # 放宽阈值
        max_vocab_size      = 5000,   # 缩小词表
        vocab_save_path     = "app2id_small.yaml",
        output_parquet_path = "./yq1005/df_app_small.parquet",
    ).main()


def demo_step_by_step():
    """分步调用：中途插入自定义逻辑"""
    from data_process import AppSeqProcess

    proc = AppSeqProcess(
        data_dir            = "./yq1005",
        vocab_save_path     = "app2id.yaml",
        output_parquet_path = "./yq1005/df_app.parquet",
    )

    proc.load_data()
    print("原始行数：", proc.df_app.shape[0])

    proc.to_list()
    # 查看一条用户序列
    sample = proc.df_app["app_name"][0]
    print("第一条用户序列（前5个App）：", sample[:5])

    proc.build_vocab()
    print("词表大小：", len(proc.app2id))
    print("高频 App Top10：", list(proc.app2id.keys())[5:15])  # 跳过特殊 Token

    proc.save_vocab()
    proc.encode_seq()

    # 查看编码结果
    encoded = proc.df_app["app_name_encoded"][0]
    print("编码后（前5个ID）：", list(encoded)[:5])

    proc.save_result()


# ─────────────────────────────────────────────
# Part 2  词表操作
# ─────────────────────────────────────────────

def demo_vocab_usage():
    """加载词表，实现 App 包名 ↔ ID 互转"""
    with open("app2id.yaml", encoding="utf-8") as f:
        app2id: dict[str, int] = yaml.safe_load(f)

    id2app = {v: k for k, v in app2id.items()}

    # 包名 → ID
    pkg = "com.whatsapp"
    idx = app2id.get(pkg, app2id["[UNK]"])
    print(f"{pkg} → {idx}")

    # ID → 包名
    print(f"ID 5 → {id2app.get(5, '[UNK]')}")

    # 特殊 Token ID
    print("特殊 Token：", {k: app2id[k] for k in ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]})

    # 词表统计
    total       = len(app2id)
    special_num = 5
    print(f"总词表大小: {total}（含 {special_num} 个特殊 Token，{total - special_num} 个 App）")


# ─────────────────────────────────────────────
# Part 3  Dataset 用法
# ─────────────────────────────────────────────

def demo_dataset():
    """手动构造 AppMLMDataset，查看 batch 内容"""
    from app_pretrain import AppMLMDataset

    # 伪造序列：实际应从 df_app.parquet 读取 app_name_encoded 列
    fake_seqs = [
        [5, 6, 7, 8, 9, 10, 11, 12],
        [5, 100, 200, 300, 7],
        list(range(5, 55)),          # 较长序列
    ]

    train_ds = AppMLMDataset(fake_seqs, is_train=True,  max_len=20)
    val_ds   = AppMLMDataset(fake_seqs, is_train=False, max_len=20, seed=42)

    item = train_ds[0]
    print("input_ids     :", item["input_ids"].tolist())
    print("attention_mask:", item["attention_mask"].tolist())
    print("labels        :", item["labels"].tolist())   # -100 = 不参与 loss

    # 验证集固定随机，两次结果相同
    assert (val_ds[0]["input_ids"] == val_ds[0]["input_ids"]).all()
    print("验证集随机性固定：OK")


def demo_dataset_from_parquet():
    """从真实 Parquet 文件构建 Dataset"""
    from app_pretrain import AppMLMDataset

    df = polars.read_parquet("./yq1005/df_app.parquet")
    seqs = df["app_name_encoded"].to_list()

    n     = len(seqs)
    cut   = int(n * 0.7)
    rng   = random.Random(42)
    random.shuffle(seqs)

    train_ds = AppMLMDataset(seqs[:cut],  is_train=True)
    val_ds   = AppMLMDataset(seqs[cut:],  is_train=False)

    print(f"train: {len(train_ds)}  val: {len(val_ds)}")


# ─────────────────────────────────────────────
# Part 4  模型构建与推理
# ─────────────────────────────────────────────

def demo_build_model():
    """手动构建模型，查看参数量"""
    from app_pretrain import AppConfig, ModelConfig, AppMLM

    app_cfg = AppConfig(
        vocab_size           = 10005,
        emb_size             = 128,
        hidden_size          = 256,
        num_hidden_layers    = 4,
        num_attention_heads  = 4,
        intermediate_size    = 1024,
        max_position_embeddings = 200,
        pad_token_id         = 0,
    )
    model_cfg = ModelConfig(app_cfg=app_cfg.to_dict())
    model     = AppMLM(model_cfg)

    total  = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total/1e6:.2f}M  可训练: {trainable/1e6:.2f}M")

    # 打印各子模块参数量
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        print(f"  {name:<20} {n/1e6:.3f}M")


def demo_forward_pass():
    """前向推理：给定 App ID 序列，获取 logits 或 CLS 向量"""
    from app_pretrain import AppConfig, ModelConfig, AppMLM, CLS_ID, SEP_ID, PAD_ID

    # 构建小模型（快速演示）
    app_cfg   = AppConfig(vocab_size=10005, hidden_size=256, emb_size=128,
                          num_hidden_layers=2, num_attention_heads=4,
                          intermediate_size=512, max_position_embeddings=200)
    model_cfg = ModelConfig(app_cfg=app_cfg.to_dict())
    model     = AppMLM(model_cfg).eval()

    # 构造一个 batch：2 条序列，长度 10
    # 格式：[CLS] app_ids... [SEP] [PAD]...
    input_ids = torch.tensor([
        [CLS_ID, 5, 6, 7, 8,  SEP_ID, PAD_ID, PAD_ID, PAD_ID, PAD_ID],
        [CLS_ID, 5, 9, 10, 11, 12,   SEP_ID,  PAD_ID, PAD_ID, PAD_ID],
    ])
    attention_mask = (input_ids != PAD_ID).long()

    with torch.no_grad():
        # 1. 获取 MLM logits（vocab 维度）
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out["logits"]                      # (batch=2, seq=10, vocab=10005)
        print("logits shape:", logits.shape)

        # 2. 提取 [CLS] 向量作为用户表示
        hidden = model.bert(
            model.app_embeddings(input_ids),
            attention_mask=model.get_extended_attention_mask(
                attention_mask, model.app_embeddings.word_embeddings.weight.dtype
            )
        )[0]
        cls_vec = hidden[:, 0, :]                   # (batch=2, hidden=256)
        print("CLS 向量 shape:", cls_vec.shape)

        # 3. 对 mask 位置预测 Top-5
        masked_pos = (input_ids == 4)               # MASK_ID=4（此处无mask，仅演示）
        pred_ids   = logits.argmax(dim=-1)
        print("预测 token IDs:", pred_ids.tolist())


def demo_load_checkpoint():
    """从检查点加载预训练模型"""
    from app_pretrain import AppMLM

    ckpt_dir = "./ckpt_app_mlm_xxxxx/final"     # 替换为实际路径
    if not os.path.exists(ckpt_dir):
        print(f"检查点不存在：{ckpt_dir}，跳过演示")
        return

    model = AppMLM.from_pretrained(ckpt_dir)
    model.eval()
    print("模型加载成功")

    # 在 GPU 上推理
    if torch.cuda.is_available():
        model = model.cuda()
        input_ids = torch.tensor([[2, 5, 6, 7, 3]]).cuda()
    else:
        input_ids = torch.tensor([[2, 5, 6, 7, 3]])

    with torch.no_grad():
        out = model(input_ids=input_ids)
    print("logits shape:", out["logits"].shape)


# ─────────────────────────────────────────────
# Part 5  单卡训练（简化版，不使用 DDP）
# ─────────────────────────────────────────────

def demo_single_gpu_train():
    """
    单卡训练示例（跳过 DDP 初始化）。
    适合本地调试，生产环境用 torchrun 启动 app_pretrain.py。
    """
    import polars
    from transformers import Trainer, TrainingArguments, set_seed
    from app_pretrain import (
        AppConfig, ModelConfig, AppMLM, AppMLMDataset,
        VOCAB_SIZE, MAX_LEN, HIDDEN, EMB_SIZE, LAYERS, HEADS, FFN,
        PER_DEV_BS, EPOCHS, LR, SEED, PAD_ID,
    )

    set_seed(SEED)

    parquet_path = "./yq1005/df_app.parquet"
    if not os.path.exists(parquet_path):
        print("数据文件不存在，请先运行 demo_data_process()，跳过训练演示")
        return

    df    = polars.read_parquet(parquet_path)
    seqs  = df["app_name_encoded"].to_list()
    cut   = int(len(seqs) * 0.7)
    random.shuffle(seqs)

    train_ds = AppMLMDataset(seqs[:cut], is_train=True)
    val_ds   = AppMLMDataset(seqs[cut:], is_train=False, seed=SEED)
    print(f"train: {len(train_ds)}  val: {len(val_ds)}")

    app_cfg   = AppConfig(vocab_size=VOCAB_SIZE, hidden_size=HIDDEN, emb_size=EMB_SIZE,
                          num_hidden_layers=LAYERS, num_attention_heads=HEADS,
                          intermediate_size=FFN, max_position_embeddings=MAX_LEN,
                          pad_token_id=PAD_ID)
    model     = AppMLM(ModelConfig(app_cfg=app_cfg.to_dict(), pad_token_id=PAD_ID))

    args = TrainingArguments(
        output_dir                  = "./ckpt_single_gpu",
        num_train_epochs            = 2,                # 演示只跑 2 epoch
        per_device_train_batch_size = 32,
        per_device_eval_batch_size  = 32,
        gradient_accumulation_steps = 1,
        learning_rate               = LR,
        weight_decay                = 0.01,
        warmup_ratio                = 0.05,
        lr_scheduler_type           = "cosine",
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
        save_total_limit            = 2,
        fp16                        = torch.cuda.is_available(),
        logging_steps               = 20,
        report_to                   = "none",
        seed                        = SEED,
    )

    trainer = Trainer(
        model         = model,
        args          = args,
        train_dataset = train_ds,
        eval_dataset  = val_ds,
    )
    trainer.train()
    trainer.save_model("./ckpt_single_gpu/final")
    print("训练完成，模型保存至 ./ckpt_single_gpu/final")


# ─────────────────────────────────────────────
# Part 6  用户向量提取（下游任务）
# ─────────────────────────────────────────────

def demo_extract_user_embeddings():
    """
    批量提取用户 CLS 向量，用于下游分类/聚类/推荐任务。
    """
    from app_pretrain import AppMLM, CLS_ID, SEP_ID, PAD_ID, MAX_LEN

    ckpt_dir = "./ckpt_single_gpu/final"
    if not os.path.exists(ckpt_dir):
        print(f"检查点不存在：{ckpt_dir}，跳过演示")
        return

    model  = AppMLM.from_pretrained(ckpt_dir).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    parquet_path = "./yq1005/df_app.parquet"
    df   = polars.read_parquet(parquet_path)
    seqs = df["app_name_encoded"].to_list()[:100]       # 演示取前100条

    def build_input(seq: list[int], max_len: int = MAX_LEN):
        tokens = [CLS_ID] + seq[: max_len - 2] + [SEP_ID]
        pad    = max_len - len(tokens)
        ids    = tokens + [PAD_ID] * pad
        mask   = [1] * len(tokens) + [0] * pad
        return ids, mask

    all_vecs = []
    batch_size = 64

    for start in range(0, len(seqs), batch_size):
        batch   = seqs[start : start + batch_size]
        ids, ms = zip(*[build_input(s) for s in batch])

        input_ids      = torch.tensor(ids, dtype=torch.long).to(device)
        attention_mask = torch.tensor(ms,  dtype=torch.long).to(device)

        with torch.no_grad():
            dtype    = model.app_embeddings.word_embeddings.weight.dtype
            ext_mask = model.get_extended_attention_mask(attention_mask, dtype)
            hidden   = model.bert(model.app_embeddings(input_ids), attention_mask=ext_mask)[0]
            cls_vec  = hidden[:, 0, :].cpu().float()    # (batch, 256)

        all_vecs.append(cls_vec)

    user_embeddings = torch.cat(all_vecs, dim=0)        # (100, 256)
    print("用户向量矩阵 shape：", user_embeddings.shape)

    # 保存为 pt 文件供下游使用
    torch.save(user_embeddings, "user_embeddings.pt")
    print("已保存至 user_embeddings.pt")


# ─────────────────────────────────────────────
# 主入口：按需注释/取消注释
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "="*60)
    print("Part 1  数据预处理")
    print("="*60)
    # demo_data_process()             # 基础流水线
    # demo_data_process_with_extra()  # 带额外数据源
    # demo_step_by_step()             # 分步调用

    print("\n" + "="*60)
    print("Part 2  词表操作")
    print("="*60)
    # demo_vocab_usage()

    print("\n" + "="*60)
    print("Part 3  Dataset")
    print("="*60)
    demo_dataset()                   # 不依赖文件，可直接运行
    # demo_dataset_from_parquet()

    print("\n" + "="*60)
    print("Part 4  模型构建与推理")
    print("="*60)
    demo_build_model()
    demo_forward_pass()
    # demo_load_checkpoint()

    print("\n" + "="*60)
    print("Part 5  单卡训练")
    print("="*60)
    # demo_single_gpu_train()         # 需要 df_app.parquet

    print("\n" + "="*60)
    print("Part 6  提取用户向量")
    print("="*60)
    # demo_extract_user_embeddings()  # 需要训练好的检查点
