#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: data_process.py
@author: YQ
@date: 2026-05-12
@desc: 数据处理部分
"""

import polars,pandas,gc,yaml
from collections import Counter
pandas.set_option('display.max_rows', None)
pandas.set_option('display.max_columns', None)
pandas.set_option('display.width', None)

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
# import matplotlib.pyplot as plt 




from sqlalchemy.engine import create_engine
from pyhive import hive

hive_engine = hive.Connection(host='10.166.17.181', port='10000', username='qing.yu01',password='monika!2122',auth='LDAP')

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from sqlalchemy import create_engine
from urllib.parse import quote  # 密码中有url不允许的特殊字符


user = "qing.yu01"      # Replace with your LDAP username
password = "monika!2122"  # Replace with your LDAP password

# Create the connection string
connection_string = (
    f"trino://{user}:{quote(password)}@10.166.16.209:8443/hive/default"
    f"?protocol=https"  # Use HTTPS for SSL
    f"&verify=false"  # Verify SSL certificates
)

# Create SQLAlchemy engine
trino_engine = create_engine(connection_string)

def get_data(sql,engine):
    df=pandas.read_sql(sql,engine)
    columns = df.columns
    columns_dict = {column:column.split('.')[-1] for column in columns}
    df.rename(columns=columns_dict,inplace=True)
    print(df.shape)
    return df


from imp import reload
import numpy as np
import pandas as pd

# import  opay_tools.model_train_new  as _model_train
# reload(_model_train)

# from opay_tools.model_train_new import TrainAndSave, WeightPipeline, make_auc_lift_metric


import os,polars
import random
import datetime
import pickle
from typing import Tuple, List, Union

import numpy as np
import pandas as pd
import xgboost,lightgbm
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def optimize_app_df(app_df):

    apply_ts_series = (
        pd.to_datetime(app_df['apply_time'].astype(str), format='%Y%m%d%H%M%S')
        .astype('int64') // 10**6  # ns -> ms
    ).to_numpy()

    app_lst_arr = app_df['app_lst'].to_numpy()

    INF = float('inf')
    result = [None] * len(app_lst_arr)

    for i, apps in enumerate(app_lst_arr):
        if not apps:
            result[i] = []
            continue
        ts = apply_ts_series[i]

        picked = [
            (app.get('firstInstallTime', INF), app['appName'])
            for app in apps
            if app.get('firstInstallTime', INF) < ts
            and app.get('lastUpdateTime', INF) < ts
        ]
        picked.sort(key=lambda x: x[0])
        result[i] = [name for _, name in picked]

    app_df['app_name'] = result
    return app_df



if __name__ == "__main__":
    app_df_ok = polars.read_parquet('./yq1005/app_df_ok.parquet')
    app_df_em = polars.read_parquet('./yq1005/app_df_em.parquet')
    app_df_em = app_df_em.drop('rn')

    df_app = polars.concat([app_df_ok,app_df_em],how = 'vertical')
    df_app = df_app.with_columns(
    polars.col("app_name").list.reverse().alias("app_name1")
    )

    all_apps = (
    df_app.select("app_name1")          
          .explode("app_name1")
          .drop_nulls("app_name1")
          )

    MIN_COUNT = 200
    app_counts = (
        all_apps.group_by("app_name1")
                .agg(polars.len().alias("count"))
                .filter(polars.col("count") >= MIN_COUNT)
                .sort("count", descending=True)
                .head(10000)
    )

    del all_apps   

    SPECIAL_TOKENS = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}
    sorted_apps = app_counts["app_name1"].to_list()
    del app_counts       

    app2id = {**SPECIAL_TOKENS, **{app: i + 5 for i, app in enumerate(sorted_apps)}}
    del sorted_apps


    with open("app2id_new.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(app2id, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    print(f"词表大小: {len(app2id)}")   

    mapping = polars.DataFrame({
    "app_name1": list(app2id.keys()),
    "app_id":    list(app2id.values()),
    }, schema={"app_name1": polars.String, "app_id": pl.Int32})



    app_name_series = polars.Series("app_name1", list(app2id.keys()), dtype=polars.String)
    app_id_series   = polars.Series("app_id",    list(app2id.values()), dtype=polars.Int32)

    df_app = df_app.with_columns(
        polars.col("app_name1")
        .list.eval(
            polars.element().replace(
                old=app_name_series,
                new=app_id_series,
                default=polars.lit(1, dtype=polars.Int32),
            )
        )
        .cast(polars.List(polars.Int32))
        .alias("app_name_encoded")
    )

    df_app.write_parquet('./yq1005/df_app.parquet')


