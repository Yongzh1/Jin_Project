from gensim.models import Word2Vec
from gensim.models import KeyedVectors

class Embedding:
    def __init__(self):
        self.i2v = Word2Vec.load('Embedding/Item2Vec.bin')
        self.openai_title = KeyedVectors.load_word2vec_format('Embedding/title_embeddings_3small.txt', binary=False)
        self.openai_brand = KeyedVectors.load_word2vec_format('Embedding/brand_embeddings_3small.txt', binary=False)
        
    def Item2Vec(self, item_id):
        
        return self.i2v.wv[item_id]
    
    def Word2Vec(self, item_id):
        
        return self.w2v[item_id]

    def OpenAI_title(self,item_id):

        return self.openai_title[item_id]

    def OpenAI_brand(self,item_id):

        return self.openai_brand[item_id]

