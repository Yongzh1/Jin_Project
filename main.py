import argparse
import numpy as np
import pandas as pd

from dataset import SessionKFold, read_csv_with_ws
from evaluator import Evaluator
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from trainer import Trainer


def main(model_name, if_under_sampling, if_fill_missing, window_size):
    # データセット読み込む
    feature = pd.read_csv("Feature/feature.csv")

    # 欠損値を0で埋めるかどうか
    if if_fill_missing == True:
        feature.fillna(0, inplace=True)

    # window_sizeによって必要な特徴量を読み込む
    extracted_feature = read_csv_with_ws(feature, window_size)

    # トレーニングデータ
    X = extracted_feature.values
    y = feature["ground_truth"].values
    groups = feature["session_number"].values

    # セッション単位でFold分け
    X_train_valid, X_test, y_train_valid, y_test, groups_train_valid, groups_test = (
        SessionKFold(X, y, groups)
    )

    # SVMとロジスティックの場合は正規化
    if model_name == "svm" or model_name == "lr":
        scaler = StandardScaler()
        X_train_valid = scaler.fit_transform(X_train_valid)
        X_test = scaler.transform(X_test)

    # モデル選択
    trainer = Trainer(
        model_name, if_under_sampling, if_fill_missing, window_size, n_trials=200
    )

    # ハイパーパラメータチューニング
    trainer.optimize(X_train_valid, y_train_valid, groups_train_valid)

    # Best Parameterに基づいて再学習
    trainer.retrain_model(X_train_valid, y_train_valid)

    # 予測
    y_pred = trainer.predict(X_test)

    # 評価
    eval = Evaluator(y_test, y_pred)
    eval.f1()
    eval.pr_auc()
    eval.roc_auc()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # 引数の追加
    parser.add_argument("model_name", type=str, help="Model name to be trained")
    parser.add_argument(
        "--if_under_sampling",
        type=bool,
        default=False,
        help="Whether to perform under-sampling (default: False)",
    )

    parser.add_argument(
        "--if_fill_missing",
        type=bool,
        default=True,
        help="Whether to perform filling missing value (default: False)",
    )

    parser.add_argument(
        "--window_size",
        type=int,
        default=1,
        help="select the window size",
    )

    args = parser.parse_args()

    main(
        args.model_name, args.if_under_sampling, args.if_fill_missing, args.window_size
    )
