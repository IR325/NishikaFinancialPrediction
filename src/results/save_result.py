"""結果の保存."""

from pathlib import Path

import pandas as pd

from config.config import Config
from config.ensemble_config import EnsembleConfig


def _format_for_submission(id_test, y_test_pred):
    """予測結果を提出できる形式に変換する."""
    df = (
        pd.DataFrame(data=y_test_pred, index=id_test, columns=["target"])
        .reset_index()
        .rename(columns={"index": "id"})
    )
    return df


def _format_for_save(y_eval, y_eval_pred):
    return pd.DataFrame({"y": y_eval, "y_pred": y_eval_pred})


def save_results(cfg: Config, y_eval, y_eval_pred, id_test, y_test_pred):
    eval_result = _format_for_save(y_eval, y_eval_pred)
    test_result = _format_for_submission(id_test, y_test_pred)
    Path(cfg.get_eval_save_path()).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.get_test_save_path()).parent.mkdir(parents=True, exist_ok=True)
    eval_result.to_csv(cfg.get_eval_save_path(), index=False)
    test_result.to_csv(cfg.get_test_save_path(), index=False)


def save_ensembled_results(
    ecfg: EnsembleConfig, y_eval, y_eval_pred, id_test, y_test_pred
):
    eval_result = _format_for_save(y_eval, y_eval_pred)
    test_result = _format_for_submission(id_test, y_test_pred)
    Path(ecfg.get_eval_save_path()).parent.mkdir(parents=True, exist_ok=True)
    Path(ecfg.get_test_save_path()).parent.mkdir(parents=True, exist_ok=True)
    eval_result.to_csv(ecfg.get_eval_save_path(), index=False)
    test_result.to_csv(ecfg.get_test_save_path(), index=False)
