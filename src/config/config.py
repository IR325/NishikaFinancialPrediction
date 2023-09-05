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

    def get_split_info(self):
        return (
            self.cfg["split"]["split_type"],
            self.cfg["split"]["valid_size"],
            self.cfg["split"]["test_size"],
        )

    def get_model_info(self):
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
