#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: torch2onnx_single.py
@author: YQ
@date: 2026-06-08
@desc: 导出无 batch 维度的 ONNX 模型（单条推理）
      输入输出 shape 不含 batch_size 维度，开发侧直接使用单条输入

CLs 模型:
  input:  input_ids [seq], attention_mask [seq]
  output: logits [num_labels]

Embedding 模型:
  input:  input_ids [seq], attention_mask [seq]
  output: cls_emb [hidden], mean_emb [hidden], hidden_states [seq, hidden]
"""

import os
import time
import argparse
import numpy as np
import torch
import onnxruntime as ort
import torch.nn as nn
from importlib.metadata import version

print("onnxruntime:", version("onnxruntime"))
print("transformers:", version("transformers"))

from app_cls import AppCLS, MAX_LEN, PAD_ID, CLS_ID, SEP_ID
from pretrain import AppMLM


# =============== 配置 ===============
parser = argparse.ArgumentParser(description="PyTorch → ONNX（单条，无 batch 维度）")
parser.add_argument("--torch_model", type=str,
                    default="/Users/yuqing/Desktop/appseq/appsep/ckpt_app_cls/final/",
                    help="PyTorch CLS 模型目录")
parser.add_argument("--onnx_model", type=str,
                    default="/Users/yuqing/Desktop/appseq/appsep/app_cls_single.onnx",
                    help="导出的 ONNX CLS 模型路径")
parser.add_argument("--torch_model_mlm", type=str,
                    default="/Users/yuqing/Desktop/appseq/appsep/ckpt_app_mlm/final/",
                    help="PyTorch MLM 模型目录")
parser.add_argument("--onnx_model_mlm", type=str,
                    default="/Users/yuqing/Desktop/appseq/appsep/app_mlm_single.onnx",
                    help="导出的 ONNX MLM embedding 模型路径")
args = parser.parse_args()

TORCH_MODEL_PATH     = args.torch_model
ONNX_MODEL_PATH      = args.onnx_model
TORCH_MODEL_PATH_MLM = args.torch_model_mlm
ONNX_MODEL_PATH_MLM  = args.onnx_model_mlm
MAX_LEN_EXPORT       = MAX_LEN


# =============== 0. 加载模型 ===============
print(f"加载预训练模型: {TORCH_MODEL_PATH_MLM}")
model_mlm = AppMLM.from_pretrained(TORCH_MODEL_PATH_MLM)
model_mlm.eval()

print(f"加载微调模型: {TORCH_MODEL_PATH}")
model = AppCLS.from_pretrained(TORCH_MODEL_PATH)
model.eval()


# =============== 0a. 导出 MLM Embedding 为 ONNX（无 batch） ===============
class AppEmbForOnnx(nn.Module):
    """将 MLM 模型的 embedding + encoder 提取为独立 ONNX，输出 cls/mean/hidden"""
    def __init__(self, mlm_model):
        super().__init__()
        self.app_embeddings = mlm_model.app_embeddings
        self.bert           = mlm_model.bert

    def forward(self, input_ids, attention_mask):
        # input_ids: [seq], attention_mask: [seq]  → unsqueeze 加 batch 维度
        input_ids      = input_ids.unsqueeze(0)       # [1, seq]
        attention_mask = attention_mask.unsqueeze(0)   # [1, seq]

        hidden = self.app_embeddings(input_ids)

        extended_mask = attention_mask[:, None, None, :].to(hidden.dtype)
        extended_mask = (1.0 - extended_mask) * -10000.0

        out = self.bert(hidden, extended_mask)
        if isinstance(out, tuple):
            hidden = out[0]
        elif isinstance(out, dict):
            hidden = out.get("hidden_states", out.get("last_hidden_state"))
        else:
            hidden = out

        cls_emb  = hidden[:, 0, :].squeeze(0)       # [hidden]
        mask_e   = attention_mask.unsqueeze(-1).to(hidden.dtype)
        sum_emb  = (hidden * mask_e).sum(dim=1)
        sum_mask = mask_e.sum(dim=1).clamp(min=1e-9)
        mean_emb = (sum_emb / sum_mask).squeeze(0)   # [hidden]
        hidden   = hidden.squeeze(0)                  # [seq, hidden]

        return cls_emb, mean_emb, hidden


emb_model = AppEmbForOnnx(model_mlm)
emb_model.eval()
emb_model.to("cpu")

print(f"开始导出 MLM Embedding ONNX → {ONNX_MODEL_PATH_MLM}")

dummy_input_ids      = torch.randint(0, 10000, (MAX_LEN_EXPORT,), dtype=torch.long)
dummy_attention_mask = torch.ones((MAX_LEN_EXPORT,), dtype=torch.long)

torch.onnx.export(
    emb_model,
    (dummy_input_ids, dummy_attention_mask),
    ONNX_MODEL_PATH_MLM,
    input_names  = ["input_ids", "attention_mask"],
    output_names = ["cls_emb", "mean_emb", "hidden_states"],
    dynamic_axes = {
        "input_ids":      {0: "seq"},
        "attention_mask": {0: "seq"},
        "cls_emb":        {},
        "mean_emb":       {},
        "hidden_states":  {0: "seq"},
    },
    opset_version=14,
    do_constant_folding=True,
    export_params=True,
)
print(f"✅ MLM Embedding ONNX 导出完成: {ONNX_MODEL_PATH_MLM}")


# =============== 0b. 导出 CLS 分类模型为 ONNX（无 batch） ===============
class AppCLSForOnnx(nn.Module):
    """包装 CLS 模型，单条输入，只输出 logits tensor"""
    def __init__(self, m: AppCLS):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask):
        # input_ids: [seq], attention_mask: [seq]  → unsqueeze 加 batch 维度
        input_ids      = input_ids.unsqueeze(0)       # [1, seq]
        attention_mask = attention_mask.unsqueeze(0)   # [1, seq]
        out = self.m(input_ids=input_ids, attention_mask=attention_mask)
        return out["logits"].squeeze(0)  # [num_labels]


export_model = AppCLSForOnnx(model)
export_model.eval()
export_model.to("cpu")

print(f"开始导出 CLS ONNX → {ONNX_MODEL_PATH}")

torch.onnx.export(
    export_model,
    (dummy_input_ids, dummy_attention_mask),
    ONNX_MODEL_PATH,
    input_names  = ["input_ids", "attention_mask"],
    output_names = ["logits"],
    dynamic_axes = {
        "input_ids":      {0: "seq"},
        "attention_mask": {0: "seq"},
        "logits":         {},
    },
    opset_version=14,
    do_constant_folding=True,
    export_params=True,
)
print(f"✅ CLS ONNX 导出完成: {ONNX_MODEL_PATH}")


# =============== 1. 加载推理模型 ===============
print("=" * 60)
print("加载推理模型")
print("=" * 60)

torch_model = model
torch_model.eval()
print(f"✅ PyTorch CLS 模型已加载: {TORCH_MODEL_PATH}")

torch_model_mlm = model_mlm
torch_model_mlm.eval()
print(f"✅ PyTorch MLM 模型已加载: {TORCH_MODEL_PATH_MLM}")

sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 4
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

ort_sess_cls = ort.InferenceSession(
    ONNX_MODEL_PATH,
    sess_options=sess_options,
    providers=["CPUExecutionProvider"],
)
print(f"✅ ONNX CLS 模型已加载: {ONNX_MODEL_PATH}")

ort_sess_emb = ort.InferenceSession(
    ONNX_MODEL_PATH_MLM,
    sess_options=sess_options,
    providers=["CPUExecutionProvider"],
)
print(f"✅ ONNX MLM Embedding 模型已加载: {ONNX_MODEL_PATH_MLM}")

# 打印模型输入输出 shape，确认无 batch 维度
print("\n--- CLS ONNX 输入/输出 ---")
for inp in ort_sess_cls.get_inputs():
    print(f"  输入 {inp.name}: {inp.shape}  dtype={inp.type}")
for out in ort_sess_cls.get_outputs():
    print(f"  输出 {out.name}: {out.shape}  dtype={out.type}")

print("\n--- Emb ONNX 输入/输出 ---")
for inp in ort_sess_emb.get_inputs():
    print(f"  输入 {inp.name}: {inp.shape}  dtype={inp.type}")
for out in ort_sess_emb.get_outputs():
    print(f"  输出 {out.name}: {out.shape}  dtype={out.type}")


# =============== 2. 预处理 ===============
def build_single(token_ids, max_len=MAX_LEN, pad_id=PAD_ID):
    """List[int] → (input_ids, attention_mask) 1-D numpy 数组"""
    ids = token_ids[:max_len]
    pad_len = max_len - len(ids)
    input_ids      = np.array(ids + [pad_id] * pad_len, dtype=np.int64)
    attention_mask = np.array([1] * len(ids) + [0] * pad_len, dtype=np.int64)
    return input_ids, attention_mask


def softmax_np(logits):
    """数值稳定的 softmax"""
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# =============== 3. CLS 推理函数（逐条） ===============
def predict_cls_torch(token_ids_list):
    """PyTorch CLS 逐条推理"""
    all_probs = []
    for ids in token_ids_list:
        input_ids, attention_mask = build_single(ids)
        with torch.no_grad():
            out = torch_model(
                input_ids=torch.from_numpy(input_ids).unsqueeze(0),
                attention_mask=torch.from_numpy(attention_mask).unsqueeze(0),
            )
            logits = out["logits"].numpy()  # [1, num_labels]
        all_probs.append(softmax_np(logits[0]))
    return np.stack(all_probs, axis=0)


def predict_cls_onnx(token_ids_list):
    """ONNX CLS 逐条推理（无 batch 维度）"""
    all_probs = []
    for ids in token_ids_list:
        input_ids, attention_mask = build_single(ids)
        logits = ort_sess_cls.run(None, {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
        })[0]  # [num_labels]
        all_probs.append(softmax_np(logits))
    return np.stack(all_probs, axis=0)


# =============== 3b. Embedding 推理函数（逐条） ===============
def predict_emb_torch(token_ids_list):
    """PyTorch Embedding 逐条推理"""
    all_cls, all_mean, all_hidden = [], [], []
    for ids in token_ids_list:
        input_ids, attention_mask = build_single(ids)
        with torch.no_grad():
            hidden = torch_model_mlm.app_embeddings(
                torch.from_numpy(input_ids).unsqueeze(0))
            am = torch.from_numpy(attention_mask).unsqueeze(0).float()  # [1, seq]
            ext_mask = (1.0 - am[:, None, None, :]) * -10000.0         # [1, 1, 1, seq]
            out = torch_model_mlm.bert(hidden, attention_mask=ext_mask)
            if isinstance(out, tuple):
                h = out[0]
            elif isinstance(out, dict):
                h = out.get("hidden_states", out.get("last_hidden_state"))
            else:
                h = out

            cls_emb = h[0, 0, :].numpy()                                    # [hidden]
            mask_e  = torch.from_numpy(attention_mask).unsqueeze(-1).float()  # [seq, 1]
            sum_emb = (h.squeeze(0) * mask_e).sum(dim=0)
            sum_mask = mask_e.sum(dim=0).clamp(min=1e-9)
            mean_emb = (sum_emb / sum_mask).numpy()                          # [hidden]
            hidden_np = h.squeeze(0).numpy()                                 # [seq, hidden]

        all_cls.append(cls_emb)
        all_mean.append(mean_emb)
        all_hidden.append(hidden_np)
    return np.stack(all_cls), np.stack(all_mean), np.stack(all_hidden)


def predict_emb_onnx(token_ids_list):
    """ONNX Embedding 逐条推理（无 batch 维度）"""
    all_cls, all_mean, all_hidden = [], [], []
    for ids in token_ids_list:
        input_ids, attention_mask = build_single(ids)
        cls_emb, mean_emb, hidden_np = ort_sess_emb.run(None, {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
        })
        all_cls.append(cls_emb)
        all_mean.append(mean_emb)
        all_hidden.append(hidden_np)
    return np.stack(all_cls), np.stack(all_mean), np.stack(all_hidden)


# =============== 4. 准备测试数据 ===============
print("\n" + "=" * 60)
print("准备测试数据")
print("=" * 60)

np.random.seed(42)

simple_samples = [
    [100, 200, 300],
    [50, 150, 250, 350, 450],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
]

TOKEN_UPPER = max(SEP_ID + 2, 10000)

def random_sample():
    length = np.random.randint(10, MAX_LEN - 2)
    body = np.random.randint(SEP_ID + 1, TOKEN_UPPER, size=length).tolist()
    return [CLS_ID] + body + [SEP_ID]

random_samples = [random_sample() for _ in range(100)]

print(f"简单样本数: {len(simple_samples)}")
print(f"随机样本数: {len(random_samples)}")


# =============== 5. CLS 精度对比 ===============
print("\n" + "=" * 60)
print("CLS 精度对比")
print("=" * 60)

print("\n--- 简单样本 ---")
torch_probs = predict_cls_torch(simple_samples)
onnx_probs  = predict_cls_onnx(simple_samples)

for i in range(len(simple_samples)):
    print(f"\nSample {i}:")
    print(f"  PyTorch probs: {torch_probs[i]}")
    print(f"  ONNX    probs: {onnx_probs[i]}")
    print(f"  差异:          {np.abs(torch_probs[i] - onnx_probs[i])}")
    print(f"  PyTorch pred:  {torch_probs[i].argmax()}")
    print(f"  ONNX    pred:  {onnx_probs[i].argmax()}")

print("\n--- 100 条随机样本统计 ---")
torch_probs = predict_cls_torch(random_samples)
onnx_probs  = predict_cls_onnx(random_samples)

abs_diff = np.abs(torch_probs - onnx_probs)
print(f"概率最大绝对误差:   {abs_diff.max():.6e}")
print(f"概率平均绝对误差:   {abs_diff.mean():.6e}")
print(f"概率 99% 分位误差:  {np.percentile(abs_diff, 99):.6e}")

torch_pred = torch_probs.argmax(axis=1)
onnx_pred  = onnx_probs.argmax(axis=1)
agreement = (torch_pred == onnx_pred).mean()
print(f"类别预测一致率:     {agreement * 100:.2f}%")

if abs_diff.max() < 1e-4:
    print("\n✅ CLS 精度完全一致 (< 1e-4)，可放心上线")
elif abs_diff.max() < 1e-3:
    print("\n⚠️  CLS 精度差异略大 (1e-4 ~ 1e-3)，可接受但建议检查")
else:
    print("\n❌ CLS 精度差异过大 (> 1e-3)，需要排查导出问题！")


# =============== 5b. Embedding 精度对比 ===============
print("\n" + "=" * 60)
print("MLM Embedding 精度对比")
print("=" * 60)

print("\n--- 100 条随机样本统计 ---")
cls_t, mean_t, hidden_t = predict_emb_torch(random_samples)
cls_o, mean_o, hidden_o = predict_emb_onnx(random_samples)

for name, t, o in [("cls_emb", cls_t, cls_o), ("mean_emb", mean_t, mean_o), ("hidden_states", hidden_t, hidden_o)]:
    d = np.abs(t - o)
    print(f"\n  [{name}]")
    print(f"    最大绝对误差:  {d.max():.6e}")
    print(f"    平均绝对误差:  {d.mean():.6e}")
    print(f"    99%分位误差:   {np.percentile(d, 99):.6e}")

    if d.max() < 1e-4:
        print(f"    ✅ 精度完全一致")
    elif d.max() < 1e-3:
        print(f"    ⚠️  精度差异可接受")
    else:
        print(f"    ❌ 精度差异过大，需排查！")


# =============== 6. 速度对比 ===============
print("\n" + "=" * 60)
print("CLS 速度对比")
print("=" * 60)

def benchmark(predict_fn, samples, name, repeat=3, warmup=1):
    for _ in range(warmup):
        predict_fn(samples)
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        predict_fn(samples)
        times.append(time.perf_counter() - t0)
    avg = np.mean(times)
    qps = len(samples) / avg
    print(f"  {name:<10} 平均耗时: {avg*1000:>7.2f} ms   QPS: {qps:>8.1f}")
    return avg

# CLS 单条
print(f"\n--- CLS 单条推理 × 50 次 ---")
single = [random_samples[0]] * 50
t_torch = benchmark(predict_cls_torch, single, "PyTorch")
t_onnx  = benchmark(predict_cls_onnx,  single, "ONNX")
print(f"  ONNX 加速比: {t_torch / t_onnx:.2f}x")

# CLS 批量
print(f"\n--- CLS 批量推理 100 条 ---")
t_torch = benchmark(predict_cls_torch, random_samples, "PyTorch")
t_onnx  = benchmark(predict_cls_onnx,  random_samples, "ONNX")
print(f"  ONNX 加速比: {t_torch / t_onnx:.2f}x")

# Embedding 单条
print(f"\n--- Embedding 单条推理 × 50 次 ---")
t_torch = benchmark(predict_emb_torch, single, "PyTorch")
t_onnx  = benchmark(predict_emb_onnx,  single, "ONNX")
print(f"  ONNX 加速比: {t_torch / t_onnx:.2f}x")

# Embedding 批量
print(f"\n--- Embedding 批量推理 100 条 ---")
t_torch = benchmark(predict_emb_torch, random_samples, "PyTorch")
t_onnx  = benchmark(predict_emb_onnx,  random_samples, "ONNX")
print(f"  ONNX 加速比: {t_torch / t_onnx:.2f}x")


# =============== 7. 模型大小对比 ===============
print("\n" + "=" * 60)
print("模型大小对比")
print("=" * 60)

def get_size(path):
    if os.path.isfile(path):
        return os.path.getsize(path) / 1024**2
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total / 1024**2

print(f"  PyTorch CLS 目录: {get_size(TORCH_MODEL_PATH):>8.2f} MB")
print(f"  ONNX    CLS 文件: {get_size(ONNX_MODEL_PATH):>8.2f} MB")
print(f"  PyTorch MLM 目录: {get_size(TORCH_MODEL_PATH_MLM):>8.2f} MB")
print(f"  ONNX    MLM 文件: {get_size(ONNX_MODEL_PATH_MLM):>8.2f} MB")

print("\n" + "=" * 60)
print("✅ 对比完成")
print("=" * 60)
