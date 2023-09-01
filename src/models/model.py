"""モデル関連のスクリプト."""
import sys

import lightgbm as lgb

sys.path.append(
    "/Users/ryusuke/Downloads/study/data_competition/nishika_金融時系列予測/NishikaFinancialPrediction/src"
)  # TODO: もっといい方法がありそう
from config.config import Config


class Model:
    def __init__(self, cfg):
        self.model_name, self.params = cfg.get_model_info()

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
            y_train, y_valid = self._convert_to_categorical(y_train, y_valid=None)
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

    def predict(self, X_pred):
        y_pred = self.model.predict(X_pred)
        if self.model_name == "lightgbm_regressor":
            pass
        elif self.model_name == "lightgbm_classifier":
            y_pred = self._convert_to_numerical(y_pred)

    def _convert_to_categorical(y_train, y_valid=None):
        pass

    def _convert_to_numerical(y_pred):
        pass