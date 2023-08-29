from pathlib import Path

import yaml


class Config:
    def __init__(self, file_path):
        with open(file_path, "r") as yml:
            self.cfg = yaml.safe_load(yml)

    def __getitem__(self, key):
        return self.cfg[key]

    def get_split_info(self):
        return (
            self.cfg["split"]["split_type"],
            self.cfg["split"]["valid_size"],
            self.cfg["split"]["test_size"],
        )

    def get_model_info(self):
        return self.cfg["model"]["model_name"], self.cfg["model"]["params"]

    def experiment_settings(self):
        return (
            self.cfg["model"]["model_name"],
            self.cfg["model"]["params"],
            self.cfg["features"],
        )

    def track_with_mlflow(self):
        return self.cfg["mlflow"]

    def get_model_save_path(self):
        return self.cfg["model_save_path"]

    def get_csv_save_path(self):
        return self.cfg["csv_save_path"]
