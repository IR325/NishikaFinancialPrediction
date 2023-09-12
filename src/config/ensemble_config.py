import os

import numpy as np
import yaml

from config.config import Config


class EnsembleConfig:
    def __init__(self, file_path, cfgs):
        # 共通の実験設定を読み込み
        with open(file_path, "r") as yml:
            self.ecfg = yaml.safe_load(yml)
        # 個別の実験設定
        self.cfgs = cfgs
        self.ensemble_num = len(cfgs)
        # ecfgをもとにcfgを書き換える
        # for i, cfg in enumerate(self.cfgs):
        #     cfg["eval_csv_save_path"] = self._overwrite_save_path(
        #         self.cfg["eval_csv_save_path"], f"eval_individual_{i}.csv"
        #     )
        #     cfg["test_csv_save_path"] = self._overwrite_save_path(
        #         self.cfg["test_csv_save_path"], f"test_individual_{i}.csv"
        #     )

    def __getitem__(self, key):
        return self.ecfg[key]

    def __setitem__(self, key, value):
        self.ecfg[key] = value

    def _overwrite_save_path(save_path: str, filename: str):
        return os.path.join(os.path.dirname(save_path), filename)

    def get_split_settings(self):
        return self.split_settings(self)

    def empty_arrays(self):
        arr = np.empty((0, self.ensemble_num))
        return arr, arr

    def get_cfgs(self):
        return self.cfgs

    def get_split_settings(self):
        return self.ecfg["split"]

    def get_test_save_path(self):
        return self.ecfg["test_csv_save_path"]

    def get_eval_save_path(self):
        return self.ecfg["eval_csv_save_path"]

    def track_with_mlflow(self):
        return self.ecfg["mlflow"]

    def get_mlflow_settings(self):
        return (self.ecfg["experiment_name"], self.ecfg["run_name"])
