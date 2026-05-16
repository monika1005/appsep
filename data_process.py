#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: app_data.py
@author: YQ
@date: 2026-05-15
@desc: app 数据预处理
"""

import os
import json
import yaml
import polars
from typing import Optional, Callable


'''
input: pd.DataFrame '[{"appName":"com.whatsapp.w4b","appTitle":"WhatsApp\xa0Business","firstInstallTime":1762980542265,"lastUpdateTime":1773526029835},
                      {"appName":"com.truecaller","appTitle":"Truecaller","firstInstallTime":1774270611614,"lastUpdateTime":1774270611614}]'

output: pd.DataFrame [8,9]
'''

class AppSeqProcess:
    def __init__(
        self,
        data_dir: str,
        min_app_count: int = 200,
        max_vocab_size: int = 10000,
        vocab_save_path: str = "app2id.yaml",
        output_parquet_path: str = "./yq1005/df_app.parquet",
        input_col: str = "app_name",           
        output_col: str = "app_name_encoded",  
        sort_key: str = "lastUpdateTime",     
        sort_desc: bool = True,             
        extract_key: str = "appName",          
        extra_sources: Optional[list[str]] = None, 
    ):
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

        self.special_tokens: dict[str, int] = {
            "[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4
        }

        self.df_app: Optional[polars.DataFrame] = None
        self.app2id: Optional[dict[str, int]] = None


    def _read_parquet(self, path: str, drop_cols: Optional[list[str]] = None) :

        df = polars.read_parquet(path)
        if drop_cols:
            df = df.drop([c for c in drop_cols if c in df.columns])
        return df

    def _make_parser(self) :

        sort_key, sort_desc, extract_key = self.sort_key, self.sort_desc, self.extract_key

        def parse_and_sort(json_str: str) -> list[str]:
            if not json_str:
                return []
            try:
                apps: list[dict] = json.loads(json_str)
                apps.sort(key=lambda x: x.get(sort_key, 0), reverse=sort_desc)
                return [app[extract_key] for app in apps if extract_key in app]
            except (json.JSONDecodeError, TypeError):
                return []

        return parse_and_sort


    def load_data(self):

        frames = [
            self._read_parquet(os.path.join(self.data_dir, "app_df_ok.parquet"), drop_cols = ["rn"] ),
            self._read_parquet(os.path.join(self.data_dir, "app_df_em.parquet"), drop_cols = ["rn"] ),
        ]
        for path in self.extra_sources:
            frames.append(self._read_parquet(path))

        self.df_app = polars.concat(frames, how="vertical")
        print(f"数据加载完成，总样本数：{self.df_app.shape[0]}")
        return self

    def to_list(self):

        self.df_app = self.df_app.with_columns(
            polars.col(self.input_col)
                  .map_elements(self._make_parser(), return_dtype=polars.List(polars.String))
                  .alias(self.input_col)
        )
        print(f"序列解析完成（排序字段={self.sort_key}, 降序={self.sort_desc}）")
        return self

    def build_vocab(self) :

        app_counts = (
            self.df_app.select(self.input_col)
                       .explode(self.input_col)
                       .drop_nulls(self.input_col)
                       .group_by(self.input_col)
                       .agg(polars.len().alias("count"))
                       .filter(polars.col("count") >= self.min_app_count)
                       .sort("count", descending=True)
                       .head(self.max_vocab_size)
        )
        sorted_apps: list[str] = app_counts[self.input_col].to_list()
        self.app2id = {
            **self.special_tokens,
            **{app: i + len(self.special_tokens) for i, app in enumerate(sorted_apps)}
        }
        print(f"词表构建完成，总词表大小：{len(self.app2id)}")
        return self

    def save_vocab(self) :
        with open(self.vocab_save_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.app2id, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
        print(f"词表已保存至：{self.vocab_save_path}")
        return self

    def encode_seq(self) :
        keys = list(self.app2id.keys())
        vals = list(self.app2id.values())
        mapping = polars.DataFrame(
            {self.input_col: keys, "app_id": vals},
            schema={self.input_col: polars.String, "app_id": polars.Int32}
        )
        unk_id = self.special_tokens["[UNK]"]
        self.df_app = self.df_app.with_columns(
            polars.col(self.input_col)
                  .list.eval(
                      polars.element().replace(
                          old=mapping[self.input_col],
                          new=mapping["app_id"],
                          default=polars.lit(unk_id, dtype=polars.Int32)
                      )
                  )
                  .alias(self.output_col)
        )
        print("序列编码完成")
        return self

    def save_result(self) :
        os.makedirs(os.path.dirname(self.output_parquet_path), exist_ok=True)
        self.df_app.write_parquet(self.output_parquet_path)
        print(f"结果已保存至：{self.output_parquet_path}")
        return self

    def main(self) :
        (
            self.load_data()
                .to_list()
                .build_vocab()
                .save_vocab()
                .encode_seq()
                .save_result()
        )
        return self.df_app


if __name__ == "__main__":
    run = AppSeqProcess(
        data_dir="./yq1005",
        min_app_count=200,
        max_vocab_size=10000,
        vocab_save_path="app2id.yaml",
        output_parquet_path="./yq1005/df_app.parquet",
        sort_key="lastUpdateTime",
        sort_desc=True,
        extract_key="appName",
    )
    df_result = run.main()
    print(df_result.columns)
