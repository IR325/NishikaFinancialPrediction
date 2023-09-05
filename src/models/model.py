"""モデル関連のスクリプト."""
import sys

import lightgbm as lgb
import numpy as np

from config.config import Config


class Model:
    def __init__(self, cfg):
        self.model_name, self.params = cfg.get_model_info()
        self.convert_dict = {}
        self.reverse_convert_dict = {}

    def _make_convert_dict(self, y):
        for i, target in enumerate(sorted(list(set(y)))):
            self.convert_dict[target] = i
        self.reverse_convert_dict = {v: k for k, v in self.convert_dict.items()}

    def _convert_to_categorical(self, y_train, y_valid=None):
        self._make_convert_dict(y_train)
        y_train = np.vectorize(lambda x: self.convert_dict[x])(y_train)
        y_valid = np.vectorize(lambda x: self.convert_dict[x])(y_valid)
        return y_train, y_valid

    def train(self, X_train, y_train, X_valid=None, y_valid=None):
        if self.model_name == "lightgbm_regressor":
            self.model = lgb.LGBMRegressor(**self.params)
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=10, verbose=True),
                    lgb.log_evaluation(1),
                ],
            )
        elif self.model_name == "lightgbm_classifier":
            y_train, y_valid = self._convert_to_categorical(y_train, y_valid)
            self.model = lgb.LGBMClassifier(**self.params)
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=10, verbose=True),
                    lgb.log_evaluation(1),
                ],
            )

    def _convert_to_numerical(self, y_pred):
        y_pred = np.vectorize(lambda x: self.reverse_convert_dict[x])(y_pred)
        return y_pred

    def predict(self, X_pred):
        y_pred = self.model.predict(X_pred)
        if self.model_name == "lightgbm_regressor":
            pass
        elif self.model_name == "lightgbm_classifier":
            y_pred = self._convert_to_numerical(y_pred)
        return y_pred


def train(cfg, X_train, y_train, X_valid, y_valid):
    model = Model(cfg)
    model.train(X_train, y_train, X_valid, y_valid)
    return model


def predict(model, X_pred):
    y_pred = model.predict(X_pred)
    return y_pred
