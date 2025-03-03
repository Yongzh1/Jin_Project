import numpy as np
import pandas as pd
import json
import argparse

from tqdm import tqdm
from dataset import read_data
from embedding_model import Embedding
from itertools import combinations

SAVE_DIRECTORY = "Feature/"
WINDOW_SIZE = 4


def save_to_csv(feature_list, output_file):
    # CSVファイルに保存
    save_path = SAVE_DIRECTORY + output_file
    df = pd.DataFrame(feature_list)
    df.to_csv(save_path, index=False)
    print(f"出力完了: {save_path}")


def price_distance(DataFrame, feature, feature_name, item_1, item_2):
    """
    DataFrame:Item_priceを調べる辞書
    """
    if item_1 in ["<BOS>", "<EOS>"] or item_2 in ["<BOS>", "<EOS>"]:
        feature[feature_name] = np.nan
    else:
        price_1 = DataFrame.loc[DataFrame["id"] == item_1, "price"].squeeze()
        price_2 = DataFrame.loc[DataFrame["id"] == item_2, "price"].squeeze()

        feature[feature_name] = abs(price_1 - price_2) / min(price_1 + 1, price_2 + 1)


def cosine_similarity(vector1, vector2):  # コサイン類似度
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    if norm1 != 0 and norm2 != 0:
        similarity = np.clip(dot_product / (norm1 * norm2), -1.0, 1.0)
    else:
        similarity = 0  # ゼロベクトルとの類似度は0と定義します

    return similarity


def item_behavior_sim(embedding, feature, feature_name, item_1, item_2):
    if (item_1 in embedding.i2v.wv) and (item_2 in embedding.i2v.wv):
        feature[feature_name] = cosine_similarity(
            embedding.Item2Vec(item_1), embedding.Item2Vec(item_2)
        )
    else:
        feature[feature_name] = np.nan


def item_title_sim(embedding, feature, feature_name, item_1, item_2):
    if item_1 in embedding.openai_title and item_2 in embedding.openai_title:
        feature[feature_name] = cosine_similarity(
            embedding.OpenAI_title(item_1), embedding.OpenAI_title(item_2)
        )
    else:
        feature[feature_name] = np.nan


def item_brand_sim(embedding, feature, feature_name, item_1, item_2):
    if item_1 in embedding.openai_brand and item_2 in embedding.openai_brand:
        feature[feature_name] = cosine_similarity(
            embedding.OpenAI_brand(item_1), embedding.OpenAI_brand(item_2)
        )
    else:
        feature[feature_name] = np.nan


def main():

    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(
        description="Feature listをCSVに保存するプログラム"
    )
    parser.add_argument("output_file", help="出力するCSVファイル名を指定してください")
    args = parser.parse_args()

    # データセット読み込み
    print("データセット読み込み中")
    jp_products, session_list_After_RS, anno_data = read_data()

    # Embeddingモデル読み込み
    print("埋め込みモデル読み込み中")
    embedding = Embedding()
    print("埋め込みモデル読み込み完了")

    # 特徴量抽出
    feature_list = []

    # window内取り出すアイテムを決定
    item_map = {i: f"l{WINDOW_SIZE - i}" for i in range(WINDOW_SIZE)}
    item_map.update({i + WINDOW_SIZE: f"r{i + 1}" for i in range(WINDOW_SIZE)})

    #######ここの10を後でlen(anno_data)に変える#######

    for k in tqdm(range(len(anno_data)), desc="Featuring data"):  # 全てのセッションに対して
        y_label = [
            int(x) for x in anno_data.iloc[k, 2].split()
        ]  # 分割点情報をリストとして変更

        for i in range(0, len(y_label)):  # ラベルごとに対して
            item_dict = {}

            start = i - WINDOW_SIZE + 1
            end = i + WINDOW_SIZE

            # window_size内のアイテムidを取得
            for j, mapped_index in enumerate(range(start, end + 1)):
                if mapped_index < 0:
                    value = session_list_After_RS[anno_data.iloc[k, 0]][0]
                elif mapped_index > len(y_label):
                    value = session_list_After_RS[anno_data.iloc[k, 0]][len(y_label)]
                else:
                    value = session_list_After_RS[anno_data.iloc[k, 0]][mapped_index]
                item_dict[item_map[j]] = value

            feature = {}

            # ラベル位置情報 - セッション番号(Session_list_After_RSの中の)
            feature["session_number"] = anno_data.iloc[k, 0]

            # ラベル位置情報 - セッション内でのラベルindex
            feature["label_index"] = i

            # 切れ目所属するセッションの長さ
            feature["session_length"] = len(y_label) + 1
            """
            # アイテムのリスト
            items = {'l1': item_l_1, 'l2': item_l_2, 'r1': item_r_1, 'r2': item_r_2}
            """
            # Item2Vec類似度
            for (key1, item1), (key2, item2) in combinations(item_dict.items(), 2):
                name = f"Item2Vec_similarity_{key1}_{key2}"
                item_behavior_sim(embedding, feature, name, item1, item2)

            # OpenAI_title類似度
            for (key1, item1), (key2, item2) in combinations(item_dict.items(), 2):
                name = f"OpenAI_title_similarity_{key1}_{key2}"
                item_title_sim(embedding, feature, name, item1, item2)

            # OpenAI_brand類似度
            for (key1, item1), (key2, item2) in combinations(item_dict.items(), 2):
                name = f"OpenAI_brand_similarity_{key1}_{key2}"
                item_brand_sim(embedding, feature, name, item1, item2)

            # 値段の近さ
            for (key1, item1), (key2, item2) in combinations(item_dict.items(), 2):
                name = f"price_distance_{key1}_{key2}"
                price_distance(jp_products, feature, name, item1, item2)

            # 正解ラベル
            feature["ground_truth"] = y_label[i]

            # 保存
            feature_list.append(feature)

    #  CSVファイルに保存
    save_to_csv(feature_list, args.output_file)


if __name__ == "__main__":
    main()
