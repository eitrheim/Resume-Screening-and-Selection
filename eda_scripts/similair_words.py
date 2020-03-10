import pandas as pd
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import Word2Vec
import multiprocessing

# read tokenized resume data and clean
df = pd.read_csv("TokenizedWords.csv", header=None)
df.head()
list(df.columns)
df = df.drop(columns=[0])
df[1] = df[1].astype(str)

# train word2vec model
EMB_DIM = 300
w2v = Word2Vec(df[1], size=EMB_DIM, window=5, min_count=5, negative=15, iter=10, workers=multiprocessing.cpu_count())
vectors = w2v.wv

# Look for similar keywords
# education
vectors.similar_by_word("education")