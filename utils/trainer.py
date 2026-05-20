#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: trainer.py
@author: YQ
@date: 2026-05-20
@desc: TODO: 在这里简单描述本文件的功能。
"""



from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader
from transformers import Trainer
from transformers.trainer_callback import TrainerCallback
from transformers.training_args import TrainingArguments


class MetricsFixTrainer(Trainer):


    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[list] = None,
        metric_key_prefix: str = "eval",
    ) -> Any:
        output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix,
        )

        # 防御性处理
        if output.metrics is None or not isinstance(output.metrics, dict):
            fallback_loss = 0.0
            if hasattr(output, "loss") and output.loss is not None:
                fallback_loss = float(output.loss)
            output.metrics = {"eval_loss": fallback_loss}

        return output


class LoggingTrainer(Trainer):
    """增强日志的 Trainer，在每个 epoch 结束后记录系统信息。"""

    def __init__(self, *args, log_gpu_stats: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_gpu_stats = log_gpu_stats

    def _maybe_log_save_evaluate(self, *args, **kwargs):
        result = super()._maybe_log_save_evaluate(*args, **kwargs)

        # 定期打印 GPU 内存使用情况
        if self.log_gpu_stats and torch.cuda.is_available():
            import gc
            gc.collect()
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            self.log({
                "gpu_mem_allocated_gb": round(allocated, 2),
                "gpu_mem_reserved_gb": round(reserved, 2),
            })

        return result


def create_trainer(
    model,
    args: TrainingArguments,
    train_dataset,
    eval_dataset=None,
    compute_metrics=None,
    callbacks=None,
    **kwargs,
) -> MetricsFixTrainer:
    """创建 Trainer 的工厂函数。"""
    trainer_cls = MetricsFixTrainer

    return trainer_cls(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks or [],
        **kwargs,
    )
