import numpy as np
import yaml

from config.config import Config


class EnsembleConfig:
    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.ensemble_num = len(cfgs)
        self.split_settings = cfgs[0].get_split_settings()
        for i, cfg in enumerate(self.cfgs):
            if cfg.get_split_settings() != self.split_settings:
                raise ValueError(f"{i}番目のconfigのsplit_settingsが他と異なります.")

    def get_split_settings(self):
        return self.split_settings(self)

    def empty_arrays(self):
        arr = np.empty((0, self.ensemble_num))
        return arr, arr

    def get_cfgs(self):
        return self.cfgs
