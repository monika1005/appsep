# appsep 使用教程

> **项目简介**：基于用户手机 App 安装列表的序列建模项目，通过 MLM（Masked Language Model）预训练方式，将 App 序列编码为稠密向量表示，可用于下游用户画像、推荐系统等任务。

---

## 目录

1. [项目结构](#1-项目结构)
2. [核心逻辑详解](#2-核心逻辑详解)
   - [数据预处理 data_process.py](#21-数据预处理-data_processpy)
   - [MLM 预训练 app_pretrain.py](#22-mlm-预训练-app_pretrainpy)
3. [环境依赖](#3-环境依赖)
4. [数据准备](#4-数据准备)
5. [快速开始](#5-快速开始)
6. [参数配置说明](#6-参数配置说明)
7. [多卡分布式训练](#7-多卡分布式训练)
8. [输出文件说明](#8-输出文件说明)
9. [常见问题](#9-常见问题)

---

## 1. 项目结构

```
appsep/
├── data_process.py      # 数据预处理：App序列解析、词表构建、编码
├── app_pretrain.py      # MLM预训练：模型定义与训练入口
├── README.md
└── LICENSE
```

运行后生成：

```
appsep/
├── app2id.yaml          # App词表（app包名 → 整数ID）
├── yq1005/
│   └── df_app.parquet   # 编码后的序列数据
├── cache/
│   ├── train_hf/        # HuggingFace Dataset格式训练集缓存
│   └── val_hf/          # HuggingFace Dataset格式验证集缓存
└── ckpt_app_mlm_<时间戳>/
    ├── checkpoint-*/    # 各epoch检查点
    ├── final/           # 最优模型
    └── logs/            # TensorBoard日志
```

---

## 2. 核心逻辑详解

### 2.1 数据预处理 `data_process.py`

#### 输入数据格式

原始数据存储在 Parquet 文件中，`app_name` 列为 JSON 字符串，每条记录对应一个用户的 App 列表：

```json
[
  {"appName": "com.whatsapp.w4b", "appTitle": "WhatsApp Business", "firstInstallTime": 1762980542265, "lastUpdateTime": 1773526029835},
  {"appName": "com.truecaller",   "appTitle": "Truecaller",         "firstInstallTime": 1774270611614, "lastUpdateTime": 1774270611614}
]
```

#### 处理流程（流水线设计）

```
load_data()
    ↓ 读取 app_df_ok.parquet + app_df_em.parquet（可追加额外数据源）
    ↓ 垂直合并为单一 DataFrame

to_list()
    ↓ 解析 JSON 字符串 → List[str]
    ↓ 按 lastUpdateTime 降序排列（最近更新的 App 排在最前面）

build_vocab()
    ↓ 统计全量 App 出现频次
    ↓ 过滤低频（< min_app_count = 200）
    ↓ 取 Top-N（max_vocab_size = 10000）
    ↓ 加入5个特殊Token：[PAD]=0, [UNK]=1, [CLS]=2, [SEP]=3, [MASK]=4

save_vocab()
    ↓ 保存为 app2id.yaml

encode_seq()
    ↓ 词表外 App → [UNK]=1
    ↓ 每个用户序列 → List[int]

save_result()
    ↓ 保存为 df_app.parquet（含 app_name_encoded 列）
```

#### 核心类 `AppSeqProcess`

| 方法 | 功能 |
|------|------|
| `load_data()` | 读取并合并 Parquet 原始数据 |
| `to_list()` | JSON 解析 + 时间排序，输出 `List[str]` |
| `build_vocab()` | 频次过滤 + Top-K 截断构建词表 |
| `save_vocab()` | 词表写入 YAML 文件 |
| `encode_seq()` | 序列整数编码，OOV 用 `[UNK]` |
| `save_result()` | 写出 Parquet 文件 |
| `main()` | 按顺序调用以上方法（链式调用） |

---

### 2.2 MLM 预训练 `app_pretrain.py`

#### 模型架构

```
输入 App ID 序列
        ↓
[AppEmbeddings]
  word_embedding(vocab=10005, emb_dim=128)
  → LayerNorm → Dropout
  → Linear(128 → 256)   ← 低维Embedding投影到高维Hidden
        ↓
[BertEncoderWithRoPE]  ×4层
  BertAttentionWithRoPE
    q/k/v 线性映射(256→256)
    → 应用 RoPE 旋转位置编码
    → Scaled Dot-Product Attention
    → out_proj
  + 残差 + LayerNorm
  FFN: Linear(256→1024) → GELU → Linear(1024→256)
  + 残差 + LayerNorm
        ↓
[MLM Head]
  Linear(256→128) → GELU → LayerNorm
  → 与 word_embedding 权重矩阵转置相乘（权重共享）
  → + bias(vocab_size)
        ↓
输出 vocab_size 维 logits
```

**关键设计点：**

- **RoPE（旋转位置编码）**：替换传统绝对位置嵌入，相对位置信息通过旋转变换注入 Q/K，泛化性更强。
- **Embedding 分离**：Embedding 维度 128 < Hidden 维度 256，通过线性层升维，减少词表参数量。
- **MLM 权重共享**：MLM Head 最终投影复用 Embedding 矩阵转置，减少参数且有正则效果。
- **高效损失计算**：只对被 mask 的位置（`labels != -100`）计算 Cross-Entropy，避免 padding 干扰。

#### MLM Masking 策略

对每条序列（去掉 `[CLS]`/`[SEP]` 后），每个 Token 以 15% 概率参与 masking：

| 概率 | 处理方式 |
|------|----------|
| 80% | 替换为 `[MASK]` |
| 10% | 替换为随机词表 Token |
| 10% | 保持原 Token 不变 |

验证集使用固定随机种子，确保每次评估结果一致。

#### 训练超参数（默认值）

| 参数 | 值 | 说明 |
|------|-----|------|
| `VOCAB_SIZE` | 10005 | 词表大小（10000 + 5特殊Token） |
| `MAX_LEN` | 200 | 最大序列长度 |
| `HIDDEN` | 256 | Transformer隐层维度 |
| `EMB_SIZE` | 128 | Embedding维度 |
| `LAYERS` | 4 | Transformer层数 |
| `HEADS` | 4 | 注意力头数 |
| `FFN` | 1024 | FFN中间层维度 |
| `MLM_PROB` | 0.15 | Masking概率 |
| `PER_DEV_BS` | 64 | 单卡批大小 |
| `GRAD_ACCUM` | 2 | 梯度累积步数 |
| `EPOCHS` | 10 | 训练轮数 |
| `LR` | 5e-4 | 学习率 |
| `TRAIN_RATIO` | 0.7 | 训练集比例 |

---

## 3. 环境依赖

```bash
pip install polars pyyaml torch transformers datasets
```

| 依赖 | 用途 |
|------|------|
| `polars` | 高性能 DataFrame 处理 |
| `pyyaml` | 词表 YAML 序列化 |
| `torch` | 深度学习框架 |
| `transformers` | Trainer / BertConfig / PreTrainedModel |
| `datasets` | HuggingFace Dataset 缓存管理 |

**建议版本：**

```bash
pip install polars>=0.20 pyyaml torch>=2.0 transformers>=4.40 datasets>=2.18
```

---

## 4. 数据准备

在工作目录下创建 `yq1005/` 文件夹，放入两个 Parquet 文件：

```
yq1005/
├── app_df_ok.parquet    # 主数据集
└── app_df_em.parquet    # 补充数据集
```

**Parquet 文件必须包含的列：**

| 列名 | 类型 | 说明 |
|------|------|------|
| `app_name` | `String` | JSON 格式的 App 列表字符串 |

**`app_name` 列中每条 JSON 记录必须包含：**

| 字段 | 说明 |
|------|------|
| `appName` | App 包名（如 `com.whatsapp`） |
| `lastUpdateTime` | 最后更新时间戳（毫秒），用于排序 |

---

## 5. 快速开始

### Step 1：数据预处理

```bash
cd /path/to/appsep
python data_process.py
```

预期输出：

```
数据加载完成，总样本数：100000
序列解析完成（排序字段=lastUpdateTime, 降序=True）
词表构建完成，总词表大小：10005
词表已保存至：app2id.yaml
序列编码完成
结果已保存至：./yq1005/df_app.parquet
```

### Step 2：单卡训练

```bash
python app_pretrain.py
```

### Step 3：多卡训练（推荐）

```bash
# 4卡示例
torchrun --nproc_per_node=4 app_pretrain.py

# 指定可见GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 app_pretrain.py
```

### Step 4：查看训练进度

```bash
# 实时查看日志
tail -f ckpt_app_mlm_<时间戳>/logs/...

# 或直接观察终端输出（每50步打印一次loss）
```

---

## 6. 参数配置说明

### 修改数据预处理参数

在 `data_process.py` 底部的 `__main__` 块中修改：

```python
run = AppSeqProcess(
    data_dir="./yq1005",          # 原始数据目录
    min_app_count=200,            # 词表最低频次阈值（越高词表越小）
    max_vocab_size=10000,         # 词表最大大小
    vocab_save_path="app2id.yaml",
    output_parquet_path="./yq1005/df_app.parquet",
    sort_key="lastUpdateTime",    # 序列排序字段
    sort_desc=True,               # True=降序（最新在前）
    extract_key="appName",        # 从JSON中提取的字段名
    # extra_sources=["./extra/more_data.parquet"],  # 额外数据源
)
```

### 修改训练超参数

在 `app_pretrain.py` 顶部的全局常量中修改：

```python
VOCAB_SIZE = 10005   # 必须与词表实际大小一致（app2id.yaml 条目数）
MAX_LEN    = 200     # 序列截断长度
HIDDEN     = 256     # 增大可提升模型容量，但需要更多显存
LAYERS     = 4       # Transformer层数
PER_DEV_BS = 64      # 根据显存调整
EPOCHS     = 10
LR         = 5e-4
DATA_PATH  = "./df_app.parquet/df_app.parquet"  # 编码数据路径
```

### 修改训练策略

在 `make_args()` 函数中调整：

```python
fp16 = True       # V100/A100使用fp16；A100也可开bf16
bf16 = False      # 开bf16时关闭fp16

eval_strategy = "epoch"   # 也可改为 "steps"
logging_steps = 50        # 日志打印频率
save_total_limit = 3      # 最多保留3个checkpoint
```

---

## 7. 多卡分布式训练

项目使用 **PyTorch DDP + NCCL** 实现数据并行：

```
rank=0 进程：
  1. 准备目录
  2. 读取 Parquet → 打乱 → 切分 train/val
  3. 保存为 HuggingFace Dataset 缓存

所有进程（含 rank=0）：
  → dist.barrier() 等待 rank=0 完成数据准备
  → 各自从缓存加载 train/val 序列
  → 构建 AppMLMDataset
  → 初始化模型（DDP 自动同步权重）
  → HuggingFace Trainer 管理训练循环
```

**等效批大小计算：**

```
等效 batch = per_device_batch × GPU数量 × grad_accum
           = 64 × 4 × 2 = 512
```

**运行命令：**

```bash
# 单机多卡
torchrun --nproc_per_node=<GPU数量> app_pretrain.py

# 多机多卡（节点0）
torchrun \
  --nnodes=2 \
  --node_rank=0 \
  --master_addr=<主节点IP> \
  --master_port=29500 \
  --nproc_per_node=4 \
  app_pretrain.py
```

---

## 8. 输出文件说明

### 词表文件 `app2id.yaml`

```yaml
'[PAD]': 0
'[UNK]': 1
'[CLS]': 2
'[SEP]': 3
'[MASK]': 4
com.whatsapp: 5
com.google.android.gm: 6
...
```

### 编码数据 `df_app.parquet`

| 列名 | 类型 | 说明 |
|------|------|------|
| `app_name` | `List[String]` | 解析后的包名列表（已排序） |
| `app_name_encoded` | `List[Int32]` | 对应的整数 ID 列表 |
| 其他原始列 | - | 透传保留 |

### 模型检查点 `ckpt_app_mlm_<时间戳>/final/`

```
final/
├── config.json          # ModelConfig（含 AppConfig）
├── model.safetensors    # 模型权重
└── trainer_state.json   # 训练状态
```

**加载预训练模型：**

```python
from app_pretrain import AppMLM, ModelConfig

model = AppMLM.from_pretrained("./ckpt_app_mlm_<时间戳>/final")

# 提取用户序列的 [CLS] 向量作为用户表示
import torch
input_ids = torch.tensor([[2, 5, 6, 7, 3]])      # [CLS] + app_ids + [SEP]
attention_mask = torch.ones_like(input_ids)
with torch.no_grad():
    out = model.bert(model.app_embeddings(input_ids))
cls_embedding = out[0][:, 0, :]  # shape: (batch, hidden=256)
```

---

## 9. 常见问题

**Q：`app2id.yaml` 中词表大小与 `VOCAB_SIZE` 不一致导致报错？**

A：在 `app_pretrain.py` 中修改 `VOCAB_SIZE` 为实际词表大小：

```python
import yaml
with open("app2id.yaml") as f:
    VOCAB_SIZE = len(yaml.safe_load(f))
```

---

**Q：显存不足（OOM）？**

A：依次尝试：
1. 减小 `PER_DEV_BS`（如 32 或 16）
2. 增大 `GRAD_ACCUM` 保持等效 batch 不变
3. 减小 `MAX_LEN`（如 128）
4. 减小 `HIDDEN` 或 `LAYERS`

---

**Q：单卡运行时 `dist.init_process_group` 报错？**

A：单卡直接运行 `python app_pretrain.py` 即可，不需要 `torchrun`。代码中 `WORLD_SIZE` 默认为 1，`dist.init_process_group` 会在单进程下正常初始化。若仍报错，可注释掉 `dist.init_process_group` 相关行。

---

**Q：如何使用额外数据源？**

A：在 `AppSeqProcess` 初始化时传入 `extra_sources`：

```python
run = AppSeqProcess(
    data_dir="./yq1005",
    extra_sources=["./other_data/extra.parquet"],
)
```

---

**Q：缓存已存在但数据有更新，如何重新生成？**

A：删除缓存目录后重新运行：

```bash
rm -rf ./cache
python app_pretrain.py
```
