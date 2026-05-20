#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: metrics.py
@author: YQ
@date: 2026-05-20
@desc: TODO: 在这里简单描述本文件的功能。
"""



from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _probs_preds_from_logits(logits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """二分类 logits -> 正类概率 + 预测标签。

    使用 softmax（与 cross_entropy 对齐），而非 sigmoid。
    logits 可能是 (N, 2) 或已经是 (N,) 的概率/分数。

    Returns:
        (pos_probs, preds): 正类概率数组, 0.5 阈值预测标签
    """
    if logits.ndim == 2 and logits.shape[-1] == 2:
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp_logits[:, 1] / exp_logits.sum(axis=1)
    else:
        probs = np.asarray(logits).flatten()

    preds = (probs > 0.5).astype(int)
    return probs, preds


def compute_cls_metrics(eval_pred) -> Dict[str, float]:
    """分类任务评估指标（Trainer compute_metrics 格式）。

    返回键会自动加上 eval_ 前缀（Trainer 行为）。
    """
    predictions, labels = eval_pred
    probs, preds = _probs_preds_from_logits(predictions)
    labels = np.asarray(labels)

    # AUC（仅在有多于一个类别时计算）
    auc = 0.0
    if len(np.unique(labels)) > 1:
        try:
            auc = roc_auc_score(labels, probs)
        except ValueError:
            auc = 0.0

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)

    return {
        "auc": auc,
        "acc": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def compute_ks_statistic(probs: np.ndarray, labels: np.ndarray) -> float:
    """计算 KS 统计量（风控常用）。"""
    df = np.stack([probs, labels], axis=1)
    df = df[np.argsort(df[:, 0])]
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos

    cum_pos = np.cumsum(df[:, 1])
    cum_neg = np.arange(1, len(labels) + 1) - cum_pos

    ks = np.max(np.abs(cum_pos / (n_pos + 1e-8) - cum_neg / (n_neg + 1e-8)))
    return float(ks)


def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: int = 10,
) -> float:

    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    # 避免除零
    expected_pct = np.clip(expected_pct, 1e-6, None)
    actual_pct = np.clip(actual_pct, 1e-6, None)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)
