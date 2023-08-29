"""結果の保存."""

import pandas as pd


def save_results(cfg, id_test, y_test_pred):
    """予測結果を提出できる形式に変換する."""
    df = (
        pd.DataFrame(data=y_test_pred, index=id_test, columns=["target"])
        .reset_index()
        .rename(columns={"index": "id"})
    )
    df.to_csv(cfg.get_csv_save_path(), index=False)
