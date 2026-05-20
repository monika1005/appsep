#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: rope.py
@author: YQ
@date: 2026-05-20
@desc: TODO: 在这里简单描述本文件的功能。
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """将 x 的后半部分旋转到前半部分的负值。

    用于 RoPE：
        x = [x1, x2, x3, x4, x5, x6, x7, x8]
        rotate_half(x) = [-x5, -x6, -x7, -x8, x1, x2, x3, x4]
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """旋转位置编码（RoPE）。

    核心思想：不给 Q/K 添加绝对位置编码，而是在 Q/K 的每个维度上
    乘以一个与位置相关的旋转矩阵。

    预计算 cos/sin 频率，复用计算结果，效率更高。

    Args:
        dim: 每个 head 的维度
        base: 旋转基频，默认 10000
        device: 设备
    """

    def __init__(self, dim: int, base: float = 10000.0, device=None):
        super().__init__()
        self.dim = dim
        self.base = base
        # 注册为 buffer，确保在 .to(device) 时自动迁移
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: int,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """对 q, k 应用 RoPE。

        Args:
            q: query tensor, shape (B, H, L, D)
            k: key tensor, shape (B, H, L, D)
            seq_len: 序列长度
            offset: 位置偏移（用于 prefill 后继续生成）

        Returns:
            (rotated_q, rotated_k)
        """
        # 预计算所有位置的 cos/sin（仅在 seq_len 变化时重新计算）
        t = torch.arange(seq_len, device=q.device, dtype=torch.float32) + offset
        freqs = torch.outer(t, self.inv_freq.to(q.device))
        emb = torch.cat([freqs, freqs], dim=-1)  # (L, D)

        cos = emb.cos()[None, None, :, :]   # (1, 1, L, D)
        sin = emb.sin()[None, None, :, :]   # (1, 1, L, D)

        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)

        return q_embed, k_embed


def apply_rope(q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """简化的 RoPE 应用函数（与旧代码兼容）。

    注意：此函数每次调用都重新计算频率，效率较低。
    推荐使用 RotaryEmbedding 模块。

    Args:
        q: query tensor, shape (B, H, L, D)
        k: key tensor, shape (B, H, L, D)

    Returns:
        (rotated_q, rotated_k)
    """
    _batch, _head, seq_len, dim = q.shape
    inv_freq = (
        1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    ).to(q.device)
    t = torch.arange(seq_len, device=q.device, dtype=torch.float32)
    sinusoid = torch.outer(t, inv_freq)
    sinusoid = torch.cat([sinusoid, sinusoid], dim=-1)
    cos = sinusoid.cos()[None, None].to(q.dtype)
    sin = sinusoid.sin()[None, None].to(q.dtype)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
