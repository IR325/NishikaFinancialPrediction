"""アンサンブル実験用スクリプト"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config.config import Config
from config.ensemble_config import EnsembleConfig
from data.make_dataset import split_data
from entry_points.run_experiment import load_data, predict, train
from features.features import process_features
from metrics.metrics import cosine_similarity
from models.model import Model
from post_process.post_process import post_process
from results.mlflow import save_with_mlflow
from results.save_result import save_results

DATA_PATH = "../../data"
RESULT_PATH = "../results"


def ensemble_preds(preds:np.ndarray):
    """予測値をアンサンブルする
    Args:
        preds: np.ndarray([予測レコード数, アンサンブル数])
    return: np.ndarray([予測レコード数])
    """
    return np.mean(preds, axis=0)

# TODO: 各学習器共通の設定と個別の設定をどう分けるか(SlidingWindowとかどうしてたっけ)
def run_ensemble(ecfg: EnsembleConfig):
    eval_preds, test_preds = ecfg.empty_arrays()
    # 各cfgに則って予測を行う
    train_data, test_data = load_data(DATA_PATH)
    # データ分割
    ds = split_data(**ecfg.get_split_settings(), train_data, test_data)
    for cfg in ecfg.get_cfgs():
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
        # 格納
        eval_preds = np.vstack([eval_preds, y_eval_pred])
        test_preds = np.vstack([test_preds, y_test_pred])
    # アンサンブル
    ensembled_eval_pred = ensemble_preds(eval_preds)
    ensembled_test_pred = ensemble_preds(test_preds)
    # 保存
    save_results()