import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression

from config.config import Config
from data.make_dataset import Dataset


def _add_features(X, features_to_add):
    if "group" in features_to_add:
        X["group"] = X["id"].apply(lambda x: x // 10000)
        X["group"] = 66
    return X


def add_features(ds, features_to_add):
    ds.X_train = _add_features(ds.X_train, features_to_add)
    ds.X_valid = _add_features(ds.X_valid, features_to_add)
    ds.X_eval = _add_features(ds.X_eval, features_to_add)
    ds.X_test = _add_features(ds.X_test, features_to_add)
    return ds


def _delete_features(X, features_to_del):
    X = X.drop(columns=features_to_del)
    return X


def delete_features(ds, features_to_del):
    ds.X_train = _delete_features(ds.X_train, features_to_del)
    ds.X_valid = _delete_features(ds.X_valid, features_to_del)
    ds.X_eval = _delete_features(ds.X_eval, features_to_del)
    ds.X_test = _delete_features(ds.X_test, features_to_del)
    return ds


def _get_select_columns(X, y, select_method, select_params):
    if select_method and select_params:
        if select_method.lower() == "selectkbest":
            # selectorの学習
            selector = SelectKBest(score_func=f_regression, **select_params)
            selector.fit(X, y)
            selected_cols = X.columns[selector.get_support()]
    else:
        selected_cols = X.columns
    return selected_cols


def select_features(ds, select_method, select_params):
    selected_cols = _get_select_columns(
        ds.X_train, ds.y_train, select_method, select_params
    )
    ds.X_train = ds.X_train[selected_cols]
    ds.X_valid = ds.X_valid[selected_cols]
    ds.X_eval = ds.X_eval[selected_cols]
    ds.X_test = ds.X_test[selected_cols]
    return ds


def process_features(cfg: Config, ds):
    (
        features_add,
        features_del,
        select_method,
        select_params,
    ) = cfg.get_feature_settings()
    ds = add_features(ds, features_add)
    ds = delete_features(ds, features_del)
    ds = select_features(ds, select_method, select_params)
    return ds
