import pandas as pd
import numpy as np
import csv
import configparser

from openai import OpenAI
from tqdm import tqdm

config = configparser.ConfigParser()
config.read("../Config/config.ini")

output_file_brand = 'brand_embeddings_3small.txt'
output_file_title = 'title_embeddings_3small.txt'

def generate_openai_embedding(text, client):
    if isinstance(text, str):#普通の商品名の場合
        embedding = client.embeddings.create(input = [text], model="text-embedding-3-small").data[0].embedding
        return np.array(embedding,dtype=np.float32)
    else: #nanの場合
        return np.zeros(1536)


def output_embedding(output_file, item_list_df, target, client):
    num = len(item_list_df)
    with open(output_file, 'w', encoding='utf-8') as file:
        # ASINと埋め込みベクトルをファイルに書き込む
        file.write(f"{num} 1536\n")
        for row in tqdm(item_list_df.itertuples(), desc=f"Processing {target}"):  
            id = row.item_id
            embed_target = getattr(row, target)
            # ベクトルをスペース区切りの文字列に変換
            embedding = ' '.join(map(str, generate_openai_embedding(embed_target, client)))
            # 商品名とベクトルをファイルに書き込む
            file.write(f"{id} {embedding}\n")  
            
def main():    
    # 埋め込みモデル
    client = OpenAI(
        api_key=config['OPENAI_API']['API_KEY'],
    )

    # 埋め込み対象 CSV
    item_list_df = pd.read_csv('item_title_brand.csv')

    # 埋め込みして出力
    output_embedding(output_file_brand, item_list_df, 'brand', client)
    print('brand埋め込み完成')
    output_embedding(output_file_title, item_list_df, 'item_title', client)
    print('title埋め込み完成')
    
if __name__ == "__main__":
    main()


