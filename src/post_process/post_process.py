import numpy as np
import pandas as pd

from config.config import Config
from data.make_dataset import Dataset


def post_process(cfg, y):
    process_type, params = cfg.get_post_process_settings()
    if process_type == "number":
        upper_limit = np.sort(y)[-params["upper"]["limit"]]
        lower_limit = np.sort(y)[params["lower"]["limit"]]
        y[y >= upper_limit] = params["upper"]["value"]
        y[y <= lower_limit] = params["lower"]["value"]
    else:
        pass
    return y
