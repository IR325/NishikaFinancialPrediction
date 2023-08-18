import pandas as pd
import numpy as np

def cosine_similarity(y_true, y_pred):
    if isinstance(y_true, pd.Series):
        y_true = y_true.values.reshape(-1)
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values.reshape(-1)
    score = np.dot(y_true, y_pred) / (np.linalg.norm(y_true) * np.linalg.norm(y_pred))
    return score