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


##### Read data ######
# read structure + one hot encoded dfs
job_dummies_ideal = pd.read_csv("~/data/job_description_one_hot_ideal_FULL.csv")
resume_dummies = pd.read_csv("~/data/resume_summary_one_hot.csv")

# just the one hot
job_dummies_ideal.drop(job_dummies_ideal.columns[1:45], axis=1, inplace=True)
resume_dummies.drop(resume_dummies.columns[2:47], axis=1, inplace=True)

##### embeddings Doc2Vec #####

#def remove_punctuation(text):
#  for punctuation in string.punctuation:
#    text = text.replace(punctuation, '')
#  return text 
#
#def generate_embeddings(forResumes=True, vectorSize=300):
#  # check parameter and decide wether do embedding on jd or resume
#  if forResumes:
#    obs = pd.read_csv('~/data/resume_summary_one_hot.csv', usecols = [0,3,12], encoding = 'latin-1')
#    obs.columns = ["ReqID", "ID", "text"]
#    final_frame = obs[["ReqID", "ID", "text"]]
#  else:
#    jobs = pd.read_csv('~/data/job_description_one_hot.csv', usecols=[0,1])
#    jobs["ID"] = jobs.ReqID
#    final_frame = jobs[["ReqID", "ID", "text"]]
#  
#  ID_df = final_frame.ID
#  ID_df.index = range(len(final_frame))
#  final_frame = final_frame.text
#
#  #Data Cleaning
#  stopwords = STOP_WORDS
#  EMAIL_REGEX = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,4}"
#  PHONE_REGEX = r"\(?(\d{3})?\)?[\s\.-]{0,2}?(\d{3})[\s\.-]{0,2}(\d{4})"
#  NAME_REGEX = r'[A-Z]([a-z]+|\.)(?:\s+[A-Z]([a-z]+|\.))*(?:\s+[a-z][a-z\-]+){0,2}\s+[A-Z](a-z+|\.)'
#
#  final_frame = final_frame.astype(str)
#  final_frame = final_frame.str.replace(r'\n','')
#  final_frame.replace(regex=True,inplace=True,to_replace=EMAIL_REGEX, value = r'')
#  final_frame.replace(regex=True,inplace=True,to_replace=PHONE_REGEX, value = r'')
#  final_frame.replace(regex=True,inplace=True,to_replace=NAME_REGEX, value = r'')
#
#  final_frame.text = final_frame.apply(remove_punctuation)
#  final_frame.dropna(axis=0, inplace=True)
#  jobdocs = []
#  for job in final_frame:
#    jobdocs.append(job)
#  
#  tupleJobDocs = []
#  analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
#  for i,text in enumerate(jobdocs):
#    lowerwords = text.lower().split()
#    filteredwords = [word for word in lowerwords if word not in stopwords]
#    tags = [i]
#    tupleJobDocs.append(analyzedDocument(filteredwords,tags))
#  modelJOB = doc2vec.Doc2Vec(tupleJobDocs, vector_size = vectorSize, min_count = 1, workers = 4)
#
#  columns = list(range(vectorSize))
#  embeddings_df_ = pd.DataFrame(columns = columns)
#
#  for i in range(len(final_frame)):
#    if i > 0:
#      temp_df = pd.DataFrame(modelJOB.docvecs[i])
#      temp_df = temp_df.T
#      _df_ = pd.concat([_df_, temp_df])
#      #print(_df_.shape)
#    else:
#      _df_ = pd.DataFrame(modelJOB.docvecs[i])
#      _df_ = _df_.T
#
#  _df_.index = range(len(final_frame))
#  final_frame = pd.concat([ID_df, _df_], axis = 1)
#  return final_frame
#
#resume_embeddings = generate_embeddings(forResumes=True)
#job_embeddings = generate_embeddings(forResumes=False)

####### prepare item features and user features
# item features
resume_dummies.drop("ReqID", axis=1, inplace=True)
resume_dummies.set_index("CanID", inplace=True)

# random sample columns
resume_dummies = resume_dummies.sample(300, axis=1, random_state=1)

resume_features_sparse = sparse.csr_matrix(resume_dummies.values)

job_dummies_ideal.set_index("ReqID", inplace=True)

job_dummies_ideal = job_dummies_ideal.sample(300, axis=1, random_state=1)

job_features_sparse = sparse.csr_matrix(job_dummies_ideal.values)

# read the interaction matrix
interaction_sparse = sparse.load_npz('data/interaction_v5.npz')
interaction_sparse.data = np.nan_to_num(interaction_sparse.data, copy=False)

# train test split for cv
train, test = random_train_test_split(interaction_sparse, test_percentage=0.3, random_state = None)

# free memory
del resume_dummies
del job_dummies_ideal
del interaction_sparse
gc.collect()

##### create and train LightFM model ######
NUM_THREADS = 4
NUM_COMPONENTS = 30
NUM_EPOCHS = 50
ITEM_ALPHA = 1e-6

model = LightFM(loss='warp'
               , item_alpha=ITEM_ALPHA
               , no_components=NUM_COMPONENTS)

%time model = model.fit(interactions=train, user_features=job_features_sparse, item_features=resume_features_sparse, epochs=NUM_EPOCHS, num_threads=NUM_THREADS)

%time train_precision = precision_at_k(model, train, user_features=job_features_sparse, item_features=resume_features_sparse, k=5).mean()
print('train precision at k: %s' %train_precision)

%time test_precision = precision_at_k(model, test, user_features=job_features_sparse, item_features=resume_features_sparse, k=5).mean()
print('test precision at k: %s' %test_precision)

%time train_auc = auc_score(model, train, num_threads=NUM_THREADS).mean()
print('train AUC: %s' %train_auc)
%time test_auc = auc_score(model, test, num_threads=NUM_THREADS).mean()
print('test AUC: %s' %test_auc)



