"""mlflowへ保存用のスクリプト."""

import os
from pathlib import Path

import pandas as pd

import mlflow

# 各種パスを指定
# DB_PATH = "../../mlflow/db/test.db"
DB_PATH = "/Users/ryusuke/Downloads/study/data_competition/nishika_金融時系列予測/NishikaFinancialPrediction/mlflow/db/test.db"
ARTIFACT_LOCATION = "../../mlflow"
MODEL_LOCATION = os.path.join(ARTIFACT_LOCATION, "models")


def _log_params(cfg):
    """パラメータの保存"""
    # モデル関連
    model_name, params = cfg.get_model_settings()
    mlflow.log_param("model_name", model_name)
    mlflow.log_params(params)
    # 特徴量関連
    (
        add_features,
        del_features,
        select_method,
        select_params,
    ) = cfg.get_feature_settings()
    for i, feature in enumerate(add_features):
        mlflow.log_param(f"add_feature_{i+1}", feature)
    for i, feature in enumerate(del_features):
        mlflow.log_param(f"del_feature_{i+1}", feature)
    mlflow.log_param("select_method", select_method)
    mlflow.log_params(select_params)


def _log_metrics(metrics):
    mlflow.log_metric("cosine_similarity", metrics)


def _log_model(cfg, model):
    model_name, _, _ = cfg.get_experiment_settings()
    save_path = cfg.get_model_save_path()
    if model_name == "lightgbm_classifier" or model_name == "lightgbm_regressor":
        mlflow.lightgbm.log_model(lgb_model=model, artifact_path=save_path)
    else:
        mlflow.sklearn.log_model("model", model)


def _log_results(cfg):
    mlflow.log_artifact(cfg.get_eval_save_path())
    mlflow.log_artifact(cfg.get_test_save_path())


def _log_results_for_ensemble(ecfg):
    mlflow.log_artifact(ecfg.get_eval_save_path())
    mlflow.log_artifact(ecfg.get_test_save_path())


def save_with_mlflow(cfg, model, metrics):
    # トラッキングサーバ（バックエンド）の場所を指定
    tracking_uri = f"sqlite:///{DB_PATH}"
    mlflow.set_tracking_uri(tracking_uri)

    # Experimentの生成
    experiment_name, run_name = cfg.get_mlflow_settings()
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            name=experiment_name,
            artifact_location=ARTIFACT_LOCATION,
        )
    else:  # 当該Experiment存在するとき、IDを取得
        experiment_id = experiment.experiment_id

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        # 実験条件の保存
        _log_params(cfg)
        # モデルの保存
        _log_model(cfg, model)
        # 精度の保存
        _log_metrics(metrics)
        # 実験結果の保存
        _log_results(cfg)


def save_ensemble_with_mlflow(ecfg, metrics):
    # トラッキングサーバ（バックエンド）の場所を指定
    tracking_uri = f"sqlite:///{DB_PATH}"
    mlflow.set_tracking_uri(tracking_uri)

    # Experimentの生成
    experiment_name, run_name = ecfg.get_mlflow_settings()
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            name=experiment_name,
            artifact_location=ARTIFACT_LOCATION,
        )
    else:  # 当該Experiment存在するとき、IDを取得
        experiment_id = experiment.experiment_id

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        # 精度の保存
        _log_metrics(metrics)
        # 実験結果の保存
        _log_results_for_ensemble(ecfg)
