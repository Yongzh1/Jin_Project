import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from itertools import combinations


def read_data():
    # アイテムデータ読み込み
    jp_products = pd.read_csv("Datasets/jp_products.csv")

    # セッション参照
    Session_After_RS = pd.read_csv("Datasets/Session_After_RS.csv")

    # 全てのRandom Sampling後のセッションを一つのリストに
    session_list_After_RS = []

    # session_list_After_RS[i][j]
    # i: セッション番号
    # j: i番目のセッションのj番目のアイテム

    for i in range(0, len(Session_After_RS)):
        session = Session_After_RS.iloc[i, 1].split()
        session_list_After_RS.append(session)

    # annotationデータの読み込み
    anno_data = pd.read_csv("Annotation/anno_data/anno_data.csv")
    anno_data = anno_data.sample(frac=1, random_state=42)  # シャッフルする

    return jp_products, session_list_After_RS, anno_data


def SessionKFold(X, y, groups, n_splits=5):
    group_kfold = GroupKFold(n_splits=5)
    for train_valid_idx, test_idx in group_kfold.split(X, y, groups=groups):
        X_train_valid, X_test = X[train_valid_idx], X[test_idx]
        y_train_valid, y_test = y[train_valid_idx], y[test_idx]
        groups_train_valid, groups_test = groups[train_valid_idx], groups[test_idx]
        break  # 最初の分割のみを取得

    return X_train_valid, X_test, y_train_valid, y_test, groups_train_valid, groups_test


def read_csv_with_ws(DataFrame, window_size):
    if window_size > 4:
        print("このwindowsizeに対応していません")
        return

    item_dict = [f"l{window_size - i}" for i in range(window_size)] + [
        f"r{i+1}" for i in range(window_size)
    ]

    feature_list = []
    for item_1, item_2 in combinations(item_dict, 2):
        name = f"Item2Vec_similarity_{item_1}_{item_2}"
        feature_list.append(name)
    for item_1, item_2 in combinations(item_dict, 2):
        name = f"OpenAI_title_similarity_{item_1}_{item_2}"
        feature_list.append(name)
    for item_1, item_2 in combinations(item_dict, 2):
        name = f"OpenAI_brand_similarity_{item_1}_{item_2}"
        feature_list.append(name)
    for item_1, item_2 in combinations(item_dict, 2):
        name = f"price_distance_{item_1}_{item_2}"
        feature_list.append(name)

    X = DataFrame.loc[:, feature_list]

    return X
