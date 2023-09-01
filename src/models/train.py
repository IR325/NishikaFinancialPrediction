"""モデル関連のスクリプト."""

import sys

import lightgbm as lgb

sys.path.append(
    "/Users/ryusuke/Downloads/study/data_competition/nishika_金融時系列予測/NishikaFinancialPrediction/src"
)  # TODO: もっといい方法がありそう
from config.config import Config


def train(cfg, X_train, y_train, X_valid=None, y_valid=None):
    model_name, params = cfg.get_model_info()
    if model_name == "lightgbm_regressor":
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=10, verbose=True),
                lgb.log_evaluation(1),
            ],
        )
    elif model_name == "lightgbm_classifier":
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=10, verbose=True),
                lgb.log_evaluation(1),
            ],
        )
    return model
