"""アンサンブル実験用スクリプト"""

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config.config import Config
from config.ensemble_config import EnsembleConfig
from data.make_dataset import split_data
from entry_points.run_experiment import load_data, post_process, predict, train
from features.features import process_features
from metrics.metrics import cosine_similarity
from results.mlflow import save_ensemble_with_mlflow
from results.save_result import save_ensembled_results

DATA_PATH = "../../data"
RESULT_PATH = "../results"


def store_individual_results(eval_preds, y_eval_pred, test_preds, y_test_pred):
    eval_preds = np.append(eval_preds, y_eval_pred)
    test_preds = np.append(test_preds, y_test_pred)
    return eval_preds, test_preds


def ensemble_preds(ecfg, eval_preds: np.ndarray, test_preds: np.ndarray):
    """予測値をアンサンブルする
    Args:
        preds: np.ndarray([予測レコード数, アンサンブル数])
    return: np.ndarray([予測レコード数])
    """
    eval_preds = eval_preds.reshape(ecfg.ensemble_num, -1)
    test_preds = test_preds.reshape(ecfg.ensemble_num, -1)
    return np.mean(eval_preds, axis=0), np.mean(test_preds, axis=0)


def save_ensemble_experiment(ecfg, y_eval, y_eval_pred, id_test, y_test_pred, metric):
    save_ensembled_results(ecfg, y_eval, y_eval_pred, id_test, y_test_pred)
    if ecfg.track_with_mlflow():
        save_ensemble_with_mlflow(ecfg, metric)


# TODO: 各学習器共通の設定と個別の設定をどう分けるか(SlidingWindowとかどうしてたっけ)
def run_ensemble(ecfg: EnsembleConfig):
    eval_preds, test_preds = np.array([]), np.array([])
    # データ読み込み
    train_data, test_data = load_data(ecfg["data_path"])
    # データ分割
    # TODO: validのrandom_state変更できなくてはいけないのでは．．．
    base_ds = split_data(train_data, test_data, **ecfg.get_split_settings())
    for cfg in ecfg.get_cfgs():
        # 特徴量選択・作成
        ds = copy.deepcopy(base_ds)
        ds = process_features(cfg, ds)
        # 学習
        model = train(cfg, ds.X_train, ds.y_train, ds.X_valid, ds.y_valid)
        # 予測
        y_eval_pred, y_test_pred = predict(model, ds.X_eval, ds.X_test)
        # 後処理
        y_eval_pred, y_test_pred = post_process(cfg, y_eval_pred, y_test_pred)
        # 精度計算
        # metric = cosine_similarity(ds.y_eval, y_eval_pred)
        # 結果保存
        # save_experiment(cfg, ds.y_eval, y_eval_pred, ds.id_test, y_test_pred, model.model, metric)
        # アンサンブル用に予測結果を格納 # TODO: RESULTクラスを作っても良い
        eval_preds, test_preds = store_individual_results(
            eval_preds, y_eval_pred, test_preds, y_test_pred
        )

    # アンサンブル
    ensembled_eval_pred, ensembled_test_pred = ensemble_preds(
        ecfg, eval_preds, test_preds
    )
    # 精度算出
    metric = cosine_similarity(ds.y_eval, ensembled_eval_pred)
    # 保存
    save_ensemble_experiment(
        ecfg, ds.y_eval, ensembled_eval_pred, ds.id_test, ensembled_test_pred, metric
    )
