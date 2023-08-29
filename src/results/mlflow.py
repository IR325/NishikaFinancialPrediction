"""mlflowへ保存用のスクリプト."""

import os
from pathlib import Path

import pandas as pd

import mlflow

# 各種パスを指定
DB_PATH = "../../mlflow/db/test.db"
ARTIFACT_LOCATION = "../../mlflow"
MODEL_LOCATION = os.path.join(ARTIFACT_LOCATION, "models")
EXPERIMENT_NAME = "experiment_1"


def save_with_mlflow(cfg, model, metrics):
    # トラッキングサーバ（バックエンド）の場所を指定
    tracking_uri = f"sqlite:///{DB_PATH}"
    mlflow.set_tracking_uri(tracking_uri)

    # Experimentの生成
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            name=EXPERIMENT_NAME,
            artifact_location=ARTIFACT_LOCATION,
        )
    else:  # 当該Experiment存在するとき、IDを取得
        experiment_id = experiment.experiment_id

    with mlflow.start_run(experiment_id=experiment_id, run_name=cfg["run_name"]) as run:
        # 実験条件の保存
        _log_params(cfg)
        # TODO: モデルの保存
        _log_model(cfg, model)
        # 精度の保存
        _log_metrics(metrics)
        # 実験結果の保存
        _log_results(cfg)


def _log_params(cfg):
    """パラメータの保存"""
    model_name, params, features = cfg.experiment_settings()
    mlflow.log_param("model_name", model_name)
    mlflow.log_params(params)
    for i, feature in enumerate(features):
        mlflow.log_param(f"feature_{i}", feature)


def _log_metrics(metrics):
    mlflow.log_metric("cosine_similarity", metrics)


def _log_model(cfg, model):
    model_name, _, _ = cfg.experiment_settings()
    save_path = cfg.get_model_save_path()
    if model_name == "lightgbm":
        mlflow.lightgbm.log_model(lgb_model=model, artifact_path=save_path)
    else:
        mlflow.sklearn.log_model("model", model)


def _log_results(cfg):
    mlflow.log_artifact(cfg.get_csv_save_path())