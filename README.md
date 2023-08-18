# 各notebook概要
- 20230816_eda.ipynb
    - EDA・学習・予測・結果保存までを行うnotebook
    - LightGBMの評価指標を間違えてコサイン類似度にしていることが問題点（カスタムメトリックの備忘録として残しておく）
- 20230817_fix_metric.ipynb
    - metricをコサイン類似度からRMSEに変更
    - なぜvalidに対する精度とtestに対する制度がここまで乖離するのだろうか
- 20230817_tuning_hp.ipynb
    - optunaのHPチューニングできるLightGBMを利用
    - trainとvalidに対する精度がこれまでより低い　なぜだろう
 