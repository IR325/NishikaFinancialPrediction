"""予測用スクリプト."""

from config.config import Config


def predict(model, X):
    return model.predict(X)
