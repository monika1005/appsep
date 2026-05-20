#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: process.py
@author: YQ
@date: 2026-05-20
@desc: 
将原始 JSON App 列表数据转换为模型可用的序列 ID。

功能：
1. 读取多个 parquet 文件
2. 解析 JSON App 列表，按时间排序
3. 构建词表（按频率筛选）
4. 编码序列

"""

import json
import os
from typing import List, Optional

import polars as pl
import yaml


class AppSeqProcess:
    """App 序列数据预处理器。

    输入：JSON 格式的 App 列表
        [{"appName": "com.whatsapp", "lastUpdateTime": 1234567890, ...}, ...]

    输出：Parquet 文件，包含编码后的 app_name_encoded 列
        [0, 1, 2, 100, 35, 8, ...]  # 词表 ID 序列
    """

    def __init__(
        self,
        data_dir: str,
        min_app_count: int = 200,
        max_vocab_size: int = 10000,
        vocab_save_path: str = "app2id.yaml",
        output_parquet_path: str = "./df_app.parquet",
        input_col: str = "app_name",
        output_col: str = "app_name_encoded",
        sort_key: str = "lastUpdateTime",
        sort_desc: bool = True,
        extract_key: str = "appName",
        extra_sources: Optional[List[str]] = None,
    ):
        """
        Args:
            data_dir: 数据目录（包含 app_df_ok.parquet, app_df_em.parquet）
            min_app_count: App 最小出现次数（低于此频率被丢弃）
            max_vocab_size: 最大词表大小（含特殊 token）
            vocab_save_path: 词表保存路径
            output_parquet_path: 输出 parquet 路径
            input_col: 输入列名（JSON 字符串）
            output_col: 输出列名（编码后的 ID 序列）
            sort_key: 排序字段
            sort_desc: 是否降序排序
            extract_key: 从 JSON 中提取的字段名
            extra_sources: 额外数据源路径列表
        """
        self.data_dir = data_dir
        self.min_app_count = min_app_count
        self.max_vocab_size = max_vocab_size
        self.vocab_save_path = vocab_save_path
        self.output_parquet_path = output_parquet_path

        self.input_col = input_col
        self.output_col = output_col
        self.sort_key = sort_key
        self.sort_desc = sort_desc
        self.extract_key = extract_key
        self.extra_sources = extra_sources or []

        self.special_tokens = {
            "[PAD]": 0,
            "[UNK]": 1,
            "[CLS]": 2,
            "[SEP]": 3,
            "[MASK]": 4,
        }

        self.df_app: Optional[pl.DataFrame] = None
        self.app2id: Optional[dict] = None

    def _read_parquet(
        self, path: str, drop_cols: Optional[List[str]] = None
    ) -> pl.DataFrame:
        """读取单个 parquet 文件。"""
        df = pl.read_parquet(path)
        if drop_cols:
            existing = [c for c in drop_cols if c in df.columns]
            if existing:
                df = df.drop(existing)
        return df

    def _make_parser(self):
        """创建 JSON 解析函数。"""
        sort_key = self.sort_key
        sort_desc = self.sort_desc
        extract_key = self.extract_key

        def parse_and_sort(json_str: str) -> List[str]:
            if not json_str:
                return []
            try:
                apps: List[dict] = json.loads(json_str)
                apps.sort(
                    key=lambda x: x.get(sort_key, 0),
                    reverse=sort_desc,
                )
                return [
                    app[extract_key]
                    for app in apps
                    if extract_key in app
                ]
            except (json.JSONDecodeError, TypeError):
                return []

        return parse_and_sort

    def load_data(self) -> "AppSeqProcess":
        """读取数据文件。"""
        frames = [
            self._read_parquet(
                os.path.join(self.data_dir, "app_df_ok.parquet"),
                drop_cols=["rn"],
            ),
            self._read_parquet(
                os.path.join(self.data_dir, "app_df_em.parquet"),
                drop_cols=["rn"],
            ),
        ]

        for path in self.extra_sources:
            frames.append(self._read_parquet(path))

        self.df_app = pl.concat(frames, how="vertical")
        print(f"[AppSeqProcess] 数据加载完成，样本数: {self.df_app.height:,}")
        return self

    def to_list(self) -> "AppSeqProcess":
        """解析 JSON 列并按时间排序。"""
        self.df_app = self.df_app.with_columns(
            pl.col(self.input_col)
            .map_elements(
                self._make_parser(),
                return_dtype=pl.List(pl.String),
            )
            .alias(self.input_col)
        )
        print(
            f"[AppSeqProcess] 序列解析完成 "
            f"(sort_key={self.sort_key}, desc={self.sort_desc})"
        )
        return self

    def build_vocab(self) -> "AppSeqProcess":
        """构建词表（按频率排序）。"""
        app_counts = (
            self.df_app.select(self.input_col)
            .explode(self.input_col)
            .drop_nulls(self.input_col)
            .group_by(self.input_col)
            .agg(pl.len().alias("count"))
            .filter(pl.col("count") >= self.min_app_count)
            .sort("count", descending=True)
            .head(self.max_vocab_size - len(self.special_tokens))
        )

        sorted_apps: List[str] = app_counts[self.input_col].to_list()

        self.app2id = {
            **self.special_tokens,
            **{app: i + len(self.special_tokens) for i, app in enumerate(sorted_apps)},
        }

        print(
            f"[AppSeqProcess] 词表构建完成，"
            f"词表大小: {len(self.app2id)} "
            f"(special={len(self.special_tokens)}, apps={len(sorted_apps)})"
        )
        return self

    def save_vocab(self) -> "AppSeqProcess":
        """保存词表为 YAML 文件。"""
        with open(
            self.vocab_save_path, "w", encoding="utf-8"
        ) as f:
            yaml.safe_dump(
                self.app2id,
                f,
                allow_unicode=True,
                sort_keys=False,
                default_flow_style=False,
            )
        print(f"[AppSeqProcess] 词表已保存: {self.vocab_save_path}")
        return self

    def encode_seq(self) -> "AppSeqProcess":
        """将 App 名称序列转换为 ID 序列。"""
        unk_id = self.special_tokens["[UNK]"]

        keys = list(self.app2id.keys())
        vals = list(self.app2id.values())
        mapping = pl.DataFrame(
            {self.input_col: keys, "app_id": vals},
            schema={self.input_col: pl.String, "app_id": pl.Int32},
        )

        self.df_app = self.df_app.with_columns(
            pl.col(self.input_col)
            .list.eval(
                pl.element().replace(
                    old=mapping[self.input_col],
                    new=mapping["app_id"],
                    default=pl.lit(unk_id, dtype=pl.Int32),
                )
            )
            .alias(self.output_col)
        )
        print("[AppSeqProcess] 序列编码完成")
        return self

    def save_result(self) -> "AppSeqProcess":
        """保存结果到 parquet 文件。"""
        os.makedirs(os.path.dirname(self.output_parquet_path), exist_ok=True)
        self.df_app.write_parquet(self.output_parquet_path)
        print(f"[AppSeqProcess] 结果已保存: {self.output_parquet_path}")
        return self

    def main(self) -> pl.DataFrame:
        """一键执行全部预处理流程。"""
        return (
            self.load_data()
            .to_list()
            .build_vocab()
            .save_vocab()
            .encode_seq()
            .save_result()
            .df_app
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./yq1005")
    parser.add_argument("--output", default="./yq1005/df_app.parquet")
    parser.add_argument("--vocab", default="app2id.yaml")
    parser.add_argument("--min-count", type=int, default=200)
    parser.add_argument("--max-vocab", type=int, default=10000)
    args = parser.parse_args()

    run = AppSeqProcess(
        data_dir=args.data_dir,
        min_app_count=args.min_count,
        max_vocab_size=args.max_vocab,
        vocab_save_path=args.vocab,
        output_parquet_path=args.output,
    )
    df = run.main()
    print(f"输出列: {df.columns}")
