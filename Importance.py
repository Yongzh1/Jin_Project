import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from dataset import read_csv_with_ws

MODEL_PATH = "Models/trained_model/"


def show_importance(model_name, window_size):
    # ファイル名
    model_filename = f"{model_name}_ws_{window_size}.joblib"
    model = joblib.load("Models/trained_model/" + model_filename)

    # 特徴量
    feature = pd.read_csv("Feature/feature.csv")

    # window_sizeによって必要な特徴量を読み込む
    X_train = read_csv_with_ws(feature, window_size)

    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_).flatten()  # SVM, ロジスティック回帰
    else:
        print("このモデルは特徴量重要度を取得できません")
        return None

    feature_names = (
        X_train.columns
        if isinstance(X_train, pd.DataFrame)
        else [f"feature_{i}" for i in range(len(importance))]
    )

    # データフレーム化してソート
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importance})
    importance_df = importance_df.sort_values(by="Importance", ascending=False).head(10)

    # 可視化
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["Feature"], importance_df["Importance"], color="skyblue")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Top 10 Feature Importances")
    plt.gca().invert_yaxis()  # 重要度が高い順に並べる
    # 特徴量名のラベルを回転させる
    plt.xticks(rotation=45, ha="right")
    # 自動調整
    plt.tight_layout()
    plt.show()
    plt.savefig(f"Importance/{model_name}_ws_{window_size}_importance.png")
