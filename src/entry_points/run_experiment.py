"""基本的な実験用スクリプト."""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config.config import Config
from data.make_dataset import split_data
from features.features import process_features
from metrics.metrics import cosine_similarity
from models.model import Model
from post_process.post_process import post_process
from results.mlflow import save_with_mlflow
from results.save_result import save_results

DATA_PATH = "../../data"
RESULT_PATH = "../results"


def load_data(data_path):
    train_data = pd.read_parquet(Path(data_path, "train.parquet"))
    test_data = pd.read_parquet(Path(data_path, "test.parquet"))
    return train_data, test_data


def train(cfg, X_train, y_train, X_valid, y_valid):
    model = Model(cfg)
    model.train(X_train, y_train, X_valid, y_valid)
    return model


def predict(model, X_pred):
    y_pred = model.predict(X_pred)
    return y_pred


def run_experiment(cfg):
    # データ読み込み
    train_data, test_data = load_data(DATA_PATH)
    # データ分割
    ds = split_data(**cfg.get_split_settings(), train_data, test_data)
    # 特徴量選択・作成
    ds = process_features(cfg, ds)
    # モデル作成
    model = train(cfg, ds.X_train, ds.y_train, ds.X_valid, ds.y_valid)
    # 予測
    y_eval_pred = predict(model, ds.X_eval)
    y_test_pred = predict(model, ds.X_test)
    # 後処理
    y_eval_pred = post_process(cfg, y_eval_pred)
    y_test_pred = post_process(cfg, y_test_pred)
    # 精度算出
    metric = cosine_similarity(ds.y_eval, y_eval_pred)
    # 保存
    save_results(cfg, ds.y_eval, y_eval_pred, ds.id_test, y_test_pred)
    if cfg.track_with_mlflow():
        save_with_mlflow(cfg, model.model, metric)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("cfg_file", help="yamlファイルのパス")

    args = parser.parse_args()

    start_time = time.time()
    cfg = Config(args.cfg_file)
    run_experiment(cfg)
    print(f"学習にかかった時間：{time.time() - start_time}")
