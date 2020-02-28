import pandas as pd
import numpy as np
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from gensim.models import doc2vec
from collections import namedtuple
from nltk import tokenize
from spacy.lang.en.stop_words import STOP_WORDS
import string

##### Read data for 5 sample positions ###########

# read raw text data for embeddings
job_text = pd.read_csv("data/job_description_one_hot.csv", index_col=0)
resume_text = pd.read_csv("data/resume_summary_one_hot.csv", index_col=0)
resume_text = resume_text[resume_text["Resume Text"] != "['nan']"]



##### embeddings Doc2Vec #####

from gensim.models import doc2vec
from collections import namedtuple
import pandas as pd
import numpy as np
from nltk import tokenize
from spacy.lang.en.stop_words import STOP_WORDS
import string

stopwords = STOP_WORDS
EMAIL_REGEX = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,4}"
PHONE_REGEX = r"\(?(\d{3})?\)?[\s\.-]{0,2}?(\d{3})[\s\.-]{0,2}(\d{4})"
NAME_REGEX = r'[A-Z]([a-z]+|\.)(?:\s+[A-Z]([a-z]+|\.))*(?:\s+[a-z][a-z\-]+){0,2}\s+[A-Z](a-z+|\.)'

def remove_punctuation(text):
  for punctuation in string.punctuation:
    text = text.replace(punctuation, '')
  return text 

def generate_embeddings(ReqID, vectorSize=300):
  #import data
  obs = pd.read_csv('~/data/resume_summary_one_hot.csv', usecols = [0,3,12], encoding = 'latin-1')
  obs.columns = ["ReqID", "ID", "text"]
  obs = obs[["ReqID", "ID", "text"]]

  jobs = pd.read_csv('~/data/job_description_one_hot.csv', usecols=[0,1])
  jobs["ID"] = jobs.ReqID
  jobs = jobs[["ReqID", "ID", "text"]]
  
  frames = [jobs, obs]
  df_ = pd.concat(frames)

  final_frame = df_[df_.ReqID == ReqID]
  ID_df = final_frame.ID
  ID_df.index = range(len(final_frame))
  final_frame = final_frame.text

  #Data Cleaning
  final_frame = final_frame.astype(str)
  final_frame = final_frame.str.replace(r'\n','')
  final_frame.replace(regex=True,inplace=True,to_replace=EMAIL_REGEX, value = r'')
  final_frame.replace(regex=True,inplace=True,to_replace=PHONE_REGEX, value = r'')
  final_frame.replace(regex=True,inplace=True,to_replace=NAME_REGEX, value = r'')

  final_frame.text = final_frame.apply(remove_punctuation)
  final_frame.dropna(axis=0, inplace=True)
  jobdocs = []
  for job in final_frame:
    jobdocs.append(job)
  
  tupleJobDocs = []
  analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
  for i,text in enumerate(jobdocs):
    lowerwords = text.lower().split()
    filteredwords = [word for word in lowerwords if word not in stopwords]
    tags = [i]
    tupleJobDocs.append(analyzedDocument(filteredwords,tags))
  modelJOB = doc2vec.Doc2Vec(tupleJobDocs, vector_size = vectorSize, min_count = 1, workers = 4)

  columns = list(range(vectorSize))
  embeddings_df_ = pd.DataFrame(columns = columns)

  for i in range(len(final_frame)):
    if i > 0:
      temp_df = pd.DataFrame(modelJOB.docvecs[i])
      temp_df = temp_df.T
      _df_ = pd.concat([_df_, temp_df])
      #print(_df_.shape)
    else:
      _df_ = pd.DataFrame(modelJOB.docvecs[i])
      _df_ = _df_.T

  _df_.index = range(len(final_frame))
  final_frame = pd.concat([ID_df, _df_], axis = 1)
  return final_frame








