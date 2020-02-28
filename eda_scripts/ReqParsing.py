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

RawReq = pd.read_csv('data/full_requisition_data.csv', encoding = 'latin-1')
RawReq.columns

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
    

RawReq["Job Description"] = RawReq["Job Description"].astype(str)
RawReq["Job Description"] = RawReq["Job Description"].apply(remove_stop_words)
RawReq["Job Description"] = RawReq["Job Description"].apply(remove_punctuation)
RawReq["Job Description"] = RawReq["Job Description"].apply(word_tokenize)

RawReq["Job Description"][0]

sentences.to_csv("TokenizedWords.csv")