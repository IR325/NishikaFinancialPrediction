"""LightGBMのn_estimatorsの値による精度の違いを確認する."""

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(
    "/Users/ryusuke/Downloads/study/data_competition/nishika_金融時系列予測/NishikaFinancialPrediction/src"
)  # TODO: もっといい方法がありそう
from config.config import Config
from data.make_dataset import split_data
from features.make_features import make_features
from metrics.metrics import cosine_similarity
from models.models import train_model
from results.mlflow import save_with_mlflow
from results.save_result import save_results

DATA_PATH = "../../data"
RESULT_PATH = "../results"


def main(cfg):
    # データ読み込み
    train_data = pd.read_parquet(Path(DATA_PATH, "train.parquet"))
    test_data = pd.read_parquet(Path(DATA_PATH, "test.parquet"))
    # データ分割
    ds = split_data(cfg, train_data, test_data)
    # 特徴量選択・作成
    ds = make_features(cfg, ds)
    # モデル作成
    model = train_model(cfg, ds.X_train, ds.y_train, ds.X_valid, ds.y_valid)
    # 精度確認
    y_eval_pred = model.predict(ds.X_eval)
    metric = cosine_similarity(ds.y_eval, y_eval_pred)
    # 本番データの予測
    y_test_pred = model.predict(ds.X_test)
    save_results(cfg, ds.id_test, y_test_pred)
    # 保存
    if cfg.track_with_mlflow():
        save_with_mlflow(cfg, model, metric)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("cfg_file", help="yamlファイルのパス")

    args = parser.parse_args()

    start_time = time.time()
    cfg = Config(args.cfg_file)
    main(cfg)
    print(f"学習にかかった時間：{time.time() - start_time}")
