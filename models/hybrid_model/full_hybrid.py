import pandas as pd
import numpy as np
import nltk
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, auc_score
from lightfm.cross_validation import random_train_test_split
import scipy as sp
import math
import re
from scipy import sparse
from nltk.tokenize import RegexpTokenizer
from gensim.models import doc2vec
from collections import namedtuple
from nltk import tokenize
import string
import gc

##### Read data ######
# read structure + one hot encoded dfs
job_dummies_ideal = pd.read_csv("~/data/job_description_one_hot_ideal_FULL.csv")
resume_dummies = pd.read_csv("~/data/resume_summary_one_hot.csv")

# just the one hot
job_dummies_ideal.drop(job_dummies_ideal.columns[1:45], axis=1, inplace=True)
resume_dummies.drop(resume_dummies.columns[2:47], axis=1, inplace=True)
resume_dummies.drop("ReqID", axis=1, inplace=True)

resume_embeddings = pd.read_csv("~/data/Resume_Embeddings.csv").fillna('')
job_embeddings = pd.read_csv("~/data/Job_Embeddings.csv").fillna('')

resume_features = resume_dummies.merge(resume_embeddings, how="left", left_on='CanID', right_on='ID')
job_features = job_dummies_ideal.merge(job_embeddings, how="left", left_on='ReqID', right_on='ID')

resume_features.set_index('CanID', inplace=True)
resume_features.drop('ID', axis=1, inplace=True)
resume_features.drop('Unnamed: 0', axis=1, inplace=True)

job_features.set_index('ReqID', inplace=True)
job_features.drop('ID', axis=1, inplace=True)
job_features.drop('Unnamed: 0', axis=1, inplace=True)

resume_features = resume_features.sample(300, axis=1, random_state=1)
job_features = job_features.sample(300, axis=1, random_state=1)

resume_features_sparse = sparse.csr_matrix(resume_features.values)
job_features_sparse = sparse.csr_matrix(job_features.values)

####### generate interaction matrix
interaction_sparse = sparse.load_npz('data/interaction_v5.npz')
interaction_sparse.data = np.nan_to_num(interaction_sparse.data, copy=False)

# train test split for cv
train, test = random_train_test_split(interaction_sparse,test_percentage=0.3, random_state = None)

# free memory
del resume_dummies
del job_dummies_ideal
del interaction_sparse
del resume_embeddings
del job_embeddings
del job_features
del resume_features
gc.collect()

##### create and train LightFM model ######
K_num = 5
model = LightFM(loss='warp', item_alpha=1e-6, no_components=30)

model = model.fit(interactions=train,
                  user_features=job_features_sparse,
                  item_features=resume_features_sparse,
                  epochs=50,
                  num_threads=4)

train_precision = precision_at_k(model, train, user_features=job_features_sparse, item_features=resume_features_sparse, k=K_num).mean()
print('train precision at k: %s' %train_precision)
test_precision = precision_at_k(model, test, user_features=job_features_sparse, item_features=resume_features_sparse, k=K_num).mean()
print('test precision at k: %s' %test_precision)

train_auc = auc_score(model, train,user_features=job_features_sparse, item_features=resume_features_sparse, num_threads=4).mean()
print('train AUC: %s' %train_auc)
test_auc = auc_score(model, test,user_features=job_features_sparse, item_features=resume_features_sparse, num_threads=4).mean()
print('test AUC: %s' %test_auc)