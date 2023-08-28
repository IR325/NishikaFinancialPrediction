import pandas as pd

from config.config import Config
from data.make_dataset import Dataset


def make_features(cfg: Config, ds: Dataset):
    if "group" in cfg["features"]:
        ds = _add_group_features(ds)
    return ds


def _add_group_features(ds):
    def _group_features(df):
        df["group"] = df["id"].apply(lambda x: x // 10000)
        return df

    ds.X_train = _group_features(ds.X_train)
    ds.X_valid = _group_features(ds.X_valid)
    ds.X_eval = _group_features(ds.X_eval)
    ds.X_test = _group_features(ds.X_test)
    return ds
