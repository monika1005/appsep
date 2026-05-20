# appsep - AppList 序列建模工具包
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: __init__.py
@author: YQ
@date: 2026-05-20
@desc:
AppSeq 是一个用于 AppList 序列建模的 PyTorch 工具包。

主要模块：
- models: 模型定义（AppMLM, AppCLS, AppConfig 等）
- data: 数据处理和 Dataset
- utils: 工具函数（RoPE, Trainer, Metrics 等）

使用示例：

预训练：
    from appsep import AppConfig, AppMLM, ModelConfig
    from appsep import AppMLMDataset
    from appsep.pretrain import main as pretrain_main

    # 命令行运行
    # python -m appsep pretrain
    # torchrun --standalone --nproc_per_node=8 -m appsep pretrain

微调：
    from appsep import AppCLS, CLSModelConfig
    from appsep.finetune import main as finetune_main

    # 命令行运行
    # python -m appsep finetune --pretrained ./ckpt_app_mlm_xxx/final
"""

import sys
import os

# 确保 appsep 目录在 path 中（支持 python -m appsep 调用）
_appsep_dir = os.path.dirname(os.path.abspath(__file__))
if _appsep_dir not in sys.path:
    sys.path.insert(0, _appsep_dir)

from appsep.models.base import AppConfig, ModelConfig, CLSModelConfig
from appsep.models.mlm import AppMLM
from appsep.models.cls import AppCLS
from appsep.data.dataset import (
    AppMLMDataset,
    AppCLSDataset,
    set_valid_num_weight,
    set_time_weight_year,
    prepare_cls_dataframe,
)
from appsep.utils.rope import apply_rope, RotaryEmbedding
from appsep.utils.metrics import (
    compute_cls_metrics,
    _probs_preds_from_logits,
    compute_ks_statistic,
    compute_psi,
)

__version__ = "0.1.0"

__all__ = [
    # configs
    "AppConfig",
    "ModelConfig",
    "CLSModelConfig",
    # models
    "AppMLM",
    "AppCLS",
    # datasets
    "AppMLMDataset",
    "AppCLSDataset",
    # weight functions
    "set_valid_num_weight",
    "set_time_weight_year",
    "prepare_cls_dataframe",
    # utils
    "apply_rope",
    "RotaryEmbedding",
    "compute_cls_metrics",
    "_probs_preds_from_logits",
    "compute_ks_statistic",
    "compute_psi",
]


def main():
    
    import argparse

    parser = argparse.ArgumentParser(description="AppSeq CLI")
    parser.add_argument("command", choices=["pretrain", "finetune"], help="子命令")
    args, unknown_args = parser.parse_known_args()

    if args.command == "pretrain":
        from appsep.pretrain import main as _main
        _main()
    elif args.command == "finetune":
        from appsep.finetune import main as _main
        _main()


if __name__ == "__main__":
    main()
