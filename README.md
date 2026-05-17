# appsep

基于用户手机 App 安装列表的序列建模项目。通过 **MLM（Masked Language Model）预训练**，将 App 序列编码为稠密向量表示，可用于用户画像、推荐系统等下游任务。

## 项目结构

```
appsep/
├── data_process.py      # 数据预处理：App序列解析、词表构建、编码
├── app_pretrain.py      # MLM预训练：模型定义与训练入口
├── example_usage.py     # 详细用法示例脚本
└── TUTORIAL.md          # 完整使用教程
```

## 模型架构

- **Embedding**：词嵌入（128维）→ LayerNorm → Linear 升维（256维）
- **Encoder**：4 层 Transformer，使用 **RoPE 旋转位置编码**替代绝对位置编码
- **MLM Head**：Linear → GELU → LayerNorm → 与 Embedding 权重矩阵绑定（weight tying）

## 环境依赖

```bash
pip install polars pyyaml torch transformers datasets
```

## 快速开始

### Step 1：数据预处理

在 `yq1005/` 目录下放入 `app_df_ok.parquet` 和 `app_df_em.parquet`，然后运行：

```bash
python data_process.py
```

输出：`app2id.yaml`（词表）、`yq1005/df_app.parquet`（编码后序列）

### Step 2：训练

```bash
# 单卡
python app_pretrain.py

# 多卡
torchrun --nproc_per_node=4 app_pretrain.py

# 调参
python app_pretrain.py --lr 1e-4 --epochs 20 --batch-size 32

# 快速链路验证（只跑10步）
python app_pretrain.py --dry-run
```

### CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data-path` | `./df_app.parquet/df_app.parquet` | 数据路径 |
| `--output-dir` | 自动生成（含时间戳） | 输出目录 |
| `--epochs` | 10 | 训练轮数 |
| `--lr` | 5e-4 | 学习率 |
| `--batch-size` | 64 | 单卡批大小 |
| `--grad-accum` | 2 | 梯度累积步数 |
| `--max-len` | 200 | 序列最大长度 |
| `--hidden` | 256 | Transformer 隐层维度 |
| `--layers` | 4 | Transformer 层数 |
| `--warmup-ratio` | 0.05 | Warmup 比例 |
| `--patience` | 3 | EarlyStopping patience（epoch 数） |
| `--bf16` | 关闭 | 使用 bf16（Ampere+ GPU），默认 fp16 |
| `--no-gc` | 关闭 | 禁用 gradient checkpointing |
| `--dry-run` | 关闭 | 只跑 10 步，验证链路 |

### 接入监控

```bash
REPORT_TO=tensorboard python app_pretrain.py
REPORT_TO=wandb        python app_pretrain.py
```

## 详细文档

参见 [TUTORIAL.md](./TUTORIAL.md)，包含：
- 数据格式说明与预处理流程
- 模型架构详解
- 多卡分布式训练指南
- 用户向量提取用法
- 常见问题解答

## License

[Apache 2.0](./LICENSE)
