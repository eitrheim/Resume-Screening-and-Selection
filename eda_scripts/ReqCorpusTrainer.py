import nltk
from gensim.models import Word2Vec
import multiprocessing
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import sys
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from spacy.lang.en import English
nlp = English()
from spacy.lang.en.stop_words import STOP_WORDS

req = pd.read_csv('data/full_requisition_data.csv')

stop_words = STOP_WORDS
stop_words.update("a","to","of","by","in","the")

def remove_stop_words(x):
  try:
    word_tokens = word_tokenize(x)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    return ' '.join(filtered_sentence)
  except:
    print(x.values)
    sys.exit(1)

def remove_punctuation(text):
  for punctuation in string.punctuation:
    text = text.replace(punctuation, '')
  return text

def lowercase_all(text):
  text_lower = text.lower()
  return text_lower


req["Job Description"] = req["Job Description"].astype(str)
req["Job Description"] = req["Job Description"].apply(remove_punctuation)
req["Job Description"] = req["Job Description"].apply(lowercase_all)
req["Job Description"] = req["Job Description"].apply(remove_stop_words)
req["Job Description"] = req["Job Description"].apply(word_tokenize)
sentences = req["Job Description"]

EMB_DIM=300
w2v=Word2Vec(sentences,size=EMB_DIM, window = 5, min_count = 1, negative = 15, iter = 10, workers = multiprocessing.cpu_count())
vectors = w2v.wv
vectors.similar_by_word('requirements', topn = 10)


req['Job Description'][0]
