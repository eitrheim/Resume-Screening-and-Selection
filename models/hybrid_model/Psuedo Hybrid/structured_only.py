import pandas as pd
import numpy as np
import nltk
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, auc_score, recall_at_k
from lightfm.cross_validation import random_train_test_split
import scipy as sp
import math

#################### Read data ####################
# read structure + one hot encoded dfs
job_dummies = pd.read_csv("~/data/job_description_one_hot_FULL.csv")
job_dummies_ideal = pd.read_csv("~/data/job_description_one_hot_ideal_FULL.csv")
resume_dummies = pd.read_csv("~/data/resume_summary_one_hot.csv")

# just the structured
job_features = job_dummies[job_dummies.columns[:45]]
job_features_ideal = job_dummies_ideal[job_dummies_ideal.columns[:45]]
resume_features = resume_dummies[resume_dummies.columns[:47]]

# just the one hot
job_dummies.drop(job_dummies.columns[1:45], axis=1, inplace=True)
job_dummies_ideal.drop(job_dummies_ideal.columns[1:45], axis=1, inplace=True)
resume_dummies.drop(resume_dummies.columns[2:47], axis=1, inplace=True)

def full_output_structured(job_dummies, resume_dummies):
  
  # get dummies for job description
  jd_df = job_dummies
  #[job_dummies.ReqID == jobID]
  # get dummies for resumes 
  resume_df = resume_dummies
  #[resume_dummies.ReqID == jobID]
  # drop req ids from resumes
  resume_df.drop(["ReqID"], inplace=True, axis=1)
  # rename the col names for ID
  jd_df.rename(columns = {'ReqID':'ID'}, inplace=True)
  resume_df.rename(columns = {'CanID':'ID'}, inplace=True)
  # concat together
  df = pd.concat([jd_df, resume_df])
  
  return df


def ID_selector(jobID):
  _structured = full_output_structured(job_dummies,
                                      resume_dummies)
  sparse_matrix = sp.sparse.csr_matrix(_structured.set_index("ID").values)
  return _structured, sparse_matrix

### Convert data to sparse matrix and split for cv###
#Known jobID's (["e3625ad", "39ee3f", "45de815", "40a2c38","63146c6"])
_str,_spr = ID_selector("dfasdfadfdsd")

_train, _test = random_train_test_split(_spr, test_percentage=0.25, random_state = None)

### create and train LightFM model ###
NUM_THREADS = 4
NUM_COMPONENTS = 30
NUM_EPOCHS = 300
ITEM_ALPHA = 1e-6

_model = LightFM(loss='warp'
                    , item_alpha=ITEM_ALPHA
                    , no_components=NUM_COMPONENTS)
_model.item_biases = 0.0

%time _model_fit = _model.fit(_train, epochs=NUM_EPOCHS, num_threads=NUM_THREADS)
#%time pos1_modelTest = pos1_model.fit(pos1_test, epochs=NUM_EPOCHS, num_threads=NUM_THREADS)

train_precision = precision_at_k(_model, _train, k=10).mean()
print('train precision at k: %s' %train_precision)
test_precision = precision_at_k(_model, _test, k=10).mean()
print('test precision at k: %s' %test_precision)

train_auc = auc_score(_model, _train, num_threads=NUM_THREADS).mean()
print('train AUC: %s' %train_auc)
test_auc = auc_score(_model, _test, num_threads=NUM_THREADS).mean()
print('test AUC: %s' %test_auc)





