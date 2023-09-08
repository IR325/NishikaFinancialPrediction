import pandas as pd

from config.config import Config
from data.make_dataset import Dataset


def _add_group_features(ds):
    def _group_features(df):
        df["group"] = df["id"].apply(lambda x: x // 10000)
        return df

    ds.X_train = _group_features(ds.X_train)
    ds.X_valid = _group_features(ds.X_valid)
    ds.X_eval = _group_features(ds.X_eval)
    ds.X_test = _group_features(ds.X_test)
    return ds


def _drop_id(ds):
    ds.X_train = ds.X_train.drop(columns=["id"])
    ds.X_valid = ds.X_valid.drop(columns=["id"])
    ds.X_eval = ds.X_eval.drop(columns=["id"])
    ds.X_test = ds.X_test.drop(columns=["id"])
    return ds


def make_features(cfg: Config, ds: Dataset):
    if features := cfg["features"]:
        if "group" in features:
            ds = _add_group_features(ds)
        if "id" in features:
            pass
    else:
        ds = _drop_id(ds)
    return ds
