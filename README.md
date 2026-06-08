# appsep

基于 Transformer 的 **AppList 序列建模**工具包，用于从用户手机应用安装列表中学习序列表示，并支持下游风控分类任务（如逾期预测）。

整体流程：**数据预处理 → MLM 预训练 → 分类微调 → ONNX 导出部署**。

## 特性

- **MLM 预训练**：对 App 安装序列做 Masked Language Model 自监督学习
- **分类微调**：加载预训练 backbone，在标签数据上做二分类微调（支持样本加权、时间衰减加权）
- **RoPE 位置编码**：Encoder 使用 Rotary Position Embedding，无需额外位置 embedding
- **分布式训练**：基于 Hugging Face Trainer，支持 `torchrun` 多卡 DDP
- **OOT 评估**：按申请日期切分 Out-of-Time 验证集，用于 early stopping 与最终评估（AUC / KS）
- **ONNX 导出**：将分类模型与 Embedding 模型导出为 ONNX，便于线上推理

## 项目结构

```
appsep/
├── config.py           # 预训练 / 微调 / 数据处理配置
├── data_process.py     # 原始 JSON → App ID 序列，构建词表
├── pretrain.py         # MLM 预训练入口
├── finetune.py         # 分类微调入口
├── torch2onnx.py       # PyTorch → ONNX 转换与精度对比
├── app_cls.py          # 早期独立版分类脚本（已被 finetune.py 替代）
├── models/
│   ├── base.py         # AppConfig / ModelConfig / CLSModelConfig
│   ├── embeddings.py   # App Embedding 层
│   ├── encoder.py      # 带 RoPE 的 Transformer Encoder
│   ├── mlm.py          # AppMLM 预训练模型
│   └── cls.py          # AppCLS 分类模型
├── data/
│   └── dataset.py      # AppMLMDataset / AppCLSDataset
└── utils/
    ├── trainer.py      # 自定义 Trainer
    ├── rope.py         # RoPE 实现
    └── metrics.py      # AUC / KS / PSI 等指标
```

## 环境依赖

主要依赖：

- Python 3.9+
- PyTorch
- transformers
- datasets
- polars
- safetensors
- scikit-learn
- pyyaml
- onnxruntime（仅 ONNX 导出/验证时需要）

建议在项目根目录创建虚拟环境后安装：

```bash
pip install torch transformers datasets polars safetensors scikit-learn pyyaml onnxruntime
```

## 快速开始

### 1. 数据预处理

原始数据中，每行 `app_name` 字段为 JSON 字符串，包含用户安装的应用列表。`AppSeqProcess` 会：

1. 解析 JSON，按 `lastUpdateTime` 降序排序
2. 统计 App 频次，过滤低频 App，构建词表（默认最多 10,000 个 App + 5 个特殊 token）
3. 将包名序列编码为整数 ID，输出 parquet

```bash
cd appsep
python data_process.py
```

或在代码中自定义参数：

```python
from appsep.data_process import AppSeqProcess

processor = AppSeqProcess(
    data_dir="./yq1005",
    min_app_count=200,
    max_vocab_size=10000,
    vocab_save_path="app2id.yaml",
    output_parquet_path="./yq1005/df_app.parquet",
)
processor.main()
```

输出文件：

| 文件 | 说明 |
|------|------|
| `app2id.yaml` | App 包名 → ID 映射词表 |
| `df_app.parquet` | 含 `app_name_encoded` 列的编码后数据 |

特殊 token 定义（`config.py`）：

| Token | ID |
|-------|----|
| `<PAD>` | 0 |
| `<UNK>` | 1 |
| `[CLS]` | 2 |
| `[SEP]` | 3 |
| `[MASK]` | 4 |

### 2. MLM 预训练

```bash
# 单卡
python pretrain.py

# 或通过模块入口
python -m appsep pretrain

# 多卡 DDP（8 卡示例）
torchrun --standalone --nproc_per_node=8 pretrain.py
```

常用参数（也可在 `config.py` 的 `PretrainConfig` 中修改）：

```bash
python pretrain.py --lr 5e-4 --epochs 10 --bs 64 --data ./df_app.parquet --output ./ckpt_app_mlm
```

训练完成后，最优模型保存在 `{output_dir}/final/`。

默认模型规模：

| 参数 | 默认值 |
|------|--------|
| vocab_size | 10,005 |
| max_len | 200 |
| hidden_size | 256 |
| emb_size | 128 |
| layers | 4 |
| heads | 4 |
| ffn | 1,024 |

### 3. 分类微调

需要两份数据：

- `df_app.parquet`：含 `app_name_encoded` 的应用序列
- `df_y.parquet`：含 `order_id`、标签列（默认 `dpd5_ever`）、`apply_date` 等

通过 `order_id` ↔ `apply_no` 关联后，按 `apply_date` 切分训练集与 OOT 集。

```bash
# 单卡
python finetune.py

# 指定预训练权重
python finetune.py --pretrained ./ckpt_app_mlm/final --output ./ckpt_app_cls

# 多卡
torchrun --standalone --nproc_per_node=4 finetune.py
```

微调特性：

- 从预训练 checkpoint 加载 backbone 权重
- 支持表现期加权（`set_valid_num_weight`）与时间衰减加权（`set_time_weight_year`）
- 以 OOT 集 AUC 作为 `metric_for_best_model`，配合 `EarlyStoppingCallback`
- 训练结束后输出 OOT 的 AUC、KS，并保存 `oot_scores.parquet`

### 4. ONNX 导出

用于不支持 PyTorch 直推的推理环境：

```bash
python torch2onnx.py \
  --torch_model ./ckpt_app_cls/final/ \
  --onnx_model ./app_cls_new.onnx \
  --torch_model_mlm ./ckpt_app_mlm/final/ \
  --onnx_model_mlm ./app_mlm_new.onnx
```

导出两个模型：

| 模型 | 输入 | 输出 |
|------|------|------|
| **CLS 分类模型** | `input_ids`, `attention_mask` | `logits` |
| **MLM Embedding 模型** | `input_ids`, `attention_mask` | `cls_emb`, `mean_emb`, `hidden_states` |

线上推理时，业务侧需先将 App 包名按 `app2id.yaml` 映射为 ID 序列，再构造模型输入。序列格式需与训练保持一致：`[CLS] + app_ids + [SEP] + <PAD>`。

```python
CLS_ID, SEP_ID, PAD_ID = 2, 3, 0

def app_seq_to_model_input(app_seq, max_len=200):
    """将已编码的 App ID 列表转为模型输入。"""
    body = app_seq[: max_len - 2]          # 预留 [CLS] 和 [SEP]
    tokens = [CLS_ID] + body + [SEP_ID]
    pad_len = max_len - len(tokens)
    input_ids = tokens + [PAD_ID] * pad_len
    attention_mask = [1] * len(tokens) + [0] * pad_len
    return input_ids, attention_mask
```

## 配置说明

所有默认超参集中在 `config.py`：

| 配置类 | 用途 |
|--------|------|
| `PretrainConfig` | MLM 预训练 |
| `CLSConfig` | 分类微调 |
| `DataProcessConfig` | 数据预处理 |

关键路径默认值：

```python
# 预训练
data_path   = "./df_app.parquet/df_app.parquet"
output_dir  = "./ckpt_app_mlm"

# 微调
data_path         = "./df_app.parquet"
label_path        = "./df_y.parquet"
pretrained_model  = "./ckpt_app_mlm/final"
output_dir        = "./ckpt_app_cls"
oot_cutoff        = "2026-03-01"   # OOT 切分日期
label_col         = "dpd5_ever"    # 标签列
```

## 模型架构

```
Input (App ID 序列)
    ↓
AppEmbeddings (Embedding + LayerNorm + Linear 投影)
    ↓
BertEncoderWithRoPE (N 层 Transformer，注意力层使用 RoPE)
    ↓
┌─────────────────┬──────────────────────┐
│   MLM Head      │   [CLS] → Classifier  │
│ (预训练)         │   (微调)              │
└─────────────────┴──────────────────────┘
```

- **预训练任务**：随机 mask 15% 的 token，预测被 mask 的 App ID
- **微调任务**：取 `[CLS]` 位置的 hidden state，接线性分类头做二分类

## Python API

```python
from appsep import (
    AppConfig, AppMLM, AppCLS,
    ModelConfig, CLSModelConfig,
    AppMLMDataset, AppCLSDataset,
    compute_cls_metrics,
)
```

## License

见 [LICENSE](./LICENSE)。
