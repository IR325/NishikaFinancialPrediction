from pathlib import Path

import yaml


class Config:
    def __init__(self, file_path):
        with open(file_path, "r") as yml:
            self.cfg = yaml.safe_load(yml)

    def __getitem__(self, key):
        return self.cfg[key]

    def __setitem__(self, key, value):
        self.cfg[key] = value

    def get_split_settings(self):
        return (
            self.cfg["split"]["split_type"],
            self.cfg["split"]["valid_size"],
            self.cfg["split"]["valid_random_state"],
            self.cfg["split"]["eval_size"],
            self.cfg["split"]["eval_random_state"],
        )

    def get_model_settings(self):
        return self.cfg["model"]["model_name"], self.cfg["model"]["params"]

    def get_experiment_settings(self):
        return (
            self.cfg["model"]["model_name"],
            self.cfg["model"]["params"],
            self.cfg["features"] if self.cfg["features"] else [],
        )

    def get_mlflow_settings(self):
        return (self.cfg["experiment_name"], self.cfg["run_name"])

    def track_with_mlflow(self):
        return self.cfg["mlflow"]

    def get_model_save_path(self):
        return self.cfg["model_save_path"]

    def get_test_save_path(self):
        return self.cfg["test_csv_save_path"]

    def get_eval_save_path(self):
        return self.cfg["eval_csv_save_path"]

    def get_post_process_settings(self):
        if self.cfg["post_process"]:
            return (
                self.cfg["post_process"]["process_type"],
                self.cfg["post_process"]["params"],
            )
        else:
            return None, None

    def get_feature_settings(self):
        feat_settings = self.cfg["features"]
        add_features = feat_settings.get("add", [])
        del_features = feat_settings.get("delete", [])
        if feat_settings.get("select", None):
            select_method = feat_settings["select"]["method"]
            select_params = feat_settings["select"]["params"]
        else:
            select_method, select_params = None, None
        return add_features, del_features, select_method, select_params
