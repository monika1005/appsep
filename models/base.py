"""
@file: base.py
@desc: 基础配置类
"""

from typing import Any, Dict, Optional

from transformers import BertConfig
from transformers.configuration_utils import PretrainedConfig


class AppConfig(BertConfig):
    """App 序列模型的基础配置。

    继承自 BertConfig，复用 HuggingFace 的标准配置结构。
    新增 emb_size 参数控制词嵌入维度（可与 hidden_size 不同）。

    Args:
        vocab_size: 词表大小（含特殊 token）
        emb_size: 词嵌入维度（默认 128）
        hidden_size: Transformer 隐藏层维度（默认 256）
        num_hidden_layers: Transformer 层数（默认 4）
        num_attention_heads: 注意力头数（默认 4）
        intermediate_size: FFN 中间层维度（默认 1024）
        max_position_embeddings: 最大位置数（默认 200）
        hidden_dropout_prob: Dropout 概率（默认 0.1）
        attention_probs_dropout_prob: 注意力 Dropout（默认 0.1）
        pad_token_id: PAD token 的 ID（默认 0）
        layer_norm_eps: LayerNorm epsilon（默认 1e-12）
        initializer_range: 初始化标准差（默认 0.02）
    """

    model_type = "app"

    def __init__(
        self,
        vocab_size: int = 10005,
        emb_size: int = 128,
        hidden_size: int = 256,
        num_hidden_layers: int = 4,
        num_attention_heads: int = 4,
        intermediate_size: int = 1024,
        max_position_embeddings: int = 200,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        pad_token_id: int = 0,
        layer_norm_eps: float = 1e-12,
        initializer_range: float = 0.02,
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
        # 强制使用 eager 模式（避免 sdpa 等自动选择导致不一致）
        self._attn_implementation = "eager"

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典，用于保存配置或传给其他模型。"""
        return {
            "vocab_size": self.vocab_size,
            "emb_size": self.emb_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "max_position_embeddings": self.max_position_embeddings,
            "hidden_dropout_prob": self.hidden_dropout_prob,
            "attention_probs_dropout_prob": self.attention_probs_dropout_prob,
            "pad_token_id": self.pad_token_id,
            "layer_norm_eps": self.layer_norm_eps,
            "initializer_range": self.initializer_range,
        }


class ModelConfig(PretrainedConfig):
    """MLM 预训练模型的顶层配置。

    包装 AppConfig，用于保存/加载完整模型。
    """

    model_type = "flow"

    def __init__(
        self,
        app_cfg: Optional[Dict] = None,
        pad_token_id: int = 0,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.app_config = app_cfg or {}
        self.pad_token_id = pad_token_id


class CLSModelConfig(PretrainedConfig):
    """下游分类模型的配置。

    在 AppConfig 基础上添加分类任务相关参数。
    """

    model_type = "app_cls"

    def __init__(
        self,
        app_cfg: Optional[Dict] = None,
        num_labels: int = 2,
        dropout: float = 0.2,
        pad_token_id: int = 0,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.app_config = app_cfg or {}
        self.num_labels = num_labels
        self.dropout = dropout
        self.pad_token_id = pad_token_id
