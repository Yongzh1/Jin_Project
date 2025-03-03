import pandas as pd

from gensim.models import Word2Vec
from gensim.models import KeyedVectors

def train_Item2Vec(data, mc,epo,vs):
    print('Training item2vec model')
    model = Word2Vec(data, 
                     vector_size=vs, 
                     sample = 1e-3,
                     negative = 1,
                     window = 6, 
                     min_count = mc, 
                     epochs = epo, #100回を目標
                     sg=1, )
    print('Complete!')
    model.save(f"../Embedding/Item2Vec.bin")
    print(f'Saved in "../Embedding/Item2Vec.bin" ')


def main(): 
    # 全セッションデータを読み込む
    session_csv = pd.read_csv('../Datasets/session_code.csv')

    # インデックス情報を読み込む
    session_df = pd.read_csv('../Datasets/Session_After_RS.csv')
    df_indices_to_remove = session_df[1000:3399]

    # インデックス情報をリスト化
    indices_to_remove = df_indices_to_remove['session_number'].tolist() 

    # インデックス情報を使ってデータを取り除く
    df_remaining = session_csv.drop(indices_to_remove)

    # BOSとEOSを追加
    """
    for idx in range(len(df_remaining)):
        df_remaining.iloc[idx]['s e s s i o n'] =  f"<BOS> {df_remaining.iloc[idx]['s e s s i o n']} <EOS>"
    """
    # 出力
    df_remaining.to_csv('../Datasets/Item2Vec_Train.csv', index=False)
    
    # Item2Vec トレーニングデータ読み込み（上のと同じ）
    i2v_train_df = pd.read_csv('../Datasets/Item2Vec_Train.csv')

    # 全てのセッションを一つのリストに
    session_list = []
    
    # session_list[i][j]
    # i: セッション番号
    # j: i番目のセッションのj番目のアイテム
    
    for i in range(0, len(i2v_train_df)):
        session = i2v_train_df.iloc[i, 0].split()
        session_list.append(session)

    train_Item2Vec(session_list, 1, 100,200)

if __name__ == "__main__":
    main()
