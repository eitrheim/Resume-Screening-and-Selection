import pandas as pd
import numpy as np
import nltk
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, auc_score
from lightfm.cross_validation import random_train_test_split
import scipy as sp
import math
from scipy import sparse
from nltk.tokenize import RegexpTokenizer
from gensim.models import doc2vec
from collections import namedtuple
from nltk import tokenize
from spacy.lang.en.stop_words import STOP_WORDS
import string
import gc

stopwords = STOP_WORDS
EMAIL_REGEX = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,4}"
PHONE_REGEX = r"\(?(\d{3})?\)?[\s\.-]{0,2}?(\d{3})[\s\.-]{0,2}(\d{4})"
NAME_REGEX = r'[A-Z]([a-z]+|\.)(?:\s+[A-Z]([a-z]+|\.))*(?:\s+[a-z][a-z\-]+){0,2}\s+[A-Z](a-z+|\.)'

##### read text files #####
job_text = pd.read_csv("~/data/full_requisition_data_tokenized.csv").fillna('')

resume_text = pd.read_csv("~/data/Candidate Report_tokenized.csv").fillna('')

##### embeddings Doc2Vec #####

def remove_punctuation(text):
  for punctuation in string.punctuation:
    text = text.replace(punctuation, '')
  return text
  
def generate_embeddings(type_, text_df, vectorSize=300):
  #import data
  _type = ["res"]
  if type_ in _type:
    frame = text_df
    frame.rename(columns = {'Candidate ID':'ID','Resume Text':'text'}, inplace=True)
    frame = frame.drop(frame.columns[0], axis = 1)
    frame = frame[['ID', 'text']]
  else:  
    frame = text_df
    frame.rename(columns = {'Req ID':'ID','Job Description':'text'}, inplace=True)
    frame = frame[["ID","text"]]
  
  final_frame = frame


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
    else:
      _df_ = pd.DataFrame(modelJOB.docvecs[i])
      _df_ = _df_.T

  _df_.index = range(len(final_frame))
  final_frame = pd.concat([ID_df, _df_], axis = 1)
  return final_frame

#resume_embeddings = generate_embeddings("res", resume_text, 300)
#job_embeddings = generate_embeddings("job", job_text, 300)

resume_embeddings = pd.read_csv("~/data/Resume_Embeddings.csv").fillna('')
job_embeddings = pd.read_csv("~/data/Job_Embeddings.csv").fillna('')

resume_text = pd.read_csv("~/data/Candidate Report_tokenized.csv").fillna('')
####### prepare item features and user features
# item features
resume_embeddings.set_index("ID", inplace=True)
resume_features_sparse = sparse.csr_matrix(resume_embeddings.values)

job_embeddings.set_index("ID", inplace=True)
job_features_sparse = sparse.csr_matrix(job_embeddings.values)

# read the interaction matrix
# interaction_sparse = sparse.load_npz('data/interaction_v4.npz')
interaction_sparse = sparse.load_npz('data/interaction_v5.npz')
interaction_sparse.data = np.nan_to_num(interaction_sparse.data, copy=False)

# train test split for cv
train, test = random_train_test_split(interaction_sparse, test_percentage=0.3, random_state = None)

# free memory
del job_embeddings
del resume_embeddings
del interaction_sparse
gc.collect()

##### create and train LightFM model ######
NUM_THREADS = 4
NUM_COMPONENTS = 30
NUM_EPOCHS = 50
ITEM_ALPHA = 1e-6
K_num = 5

model = LightFM(loss='warp'
               , item_alpha=ITEM_ALPHA
               , no_components=NUM_COMPONENTS)

%time model = model.fit(interactions=train, user_features=job_features_sparse, item_features=resume_features_sparse, epochs=NUM_EPOCHS, num_threads=NUM_THREADS)

%time test_precision = precision_at_k(model, test, user_features=job_features_sparse, item_features=resume_features_sparse, k=K_num).mean()
print('test precision at k: %s' %test_precision)

%time train_precision = precision_at_k(model, train, user_features=job_features_sparse, item_features=resume_features_sparse, k=K_num).mean()
print('train precision at k: %s' %train_precision)

%time test_auc = auc_score(model, test,user_features=job_features_sparse, item_features=resume_features_sparse, num_threads=NUM_THREADS).mean()
print('test AUC: %s' %test_auc)

%time train_auc = auc_score(model, train,user_features=job_features_sparse, item_features=resume_features_sparse, num_threads=NUM_THREADS).mean()
print('train AUC: %s' %train_auc)
