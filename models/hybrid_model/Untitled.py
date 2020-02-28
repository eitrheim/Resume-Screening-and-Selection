import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import doc2vec
from collections import namedtuple
from lightfm import LightFM
from lightfm.cross_validation import random_train_test_split
import scipy as sp
import gc

#################### text embeddings TFIDF ####################
def GenerateTfidfEmbedding(req_id, job_text_df, resume_text_df):
  pos_jd_text = job_text_df
  pos_jd_text.rename(columns = {'Req ID':'ID','Job Description':'text'}, inplace=True)
  pos_jd_text.ID = req_id
  pos_jd_text = pos_jd_text[['ID', 'text']]
  pos_resume_text = resume_text_df
  pos_resume_text.rename(columns = {'Candidate ID':'ID','Resume Text':'text'}, inplace=True)
  pos_resume_text = pos_resume_text[['ID', 'text']]
  
  #append to same df
  df = pos_jd_text.append(pos_resume_text)
  df.set_index('ID', inplace=True)
  
  # join words and vectorize
  tokenizer = RegexpTokenizer(r'\w+')
  df['text'] = df['text'].apply(lambda x: tokenizer.tokenize(x))
  df['text'] = df['text'].apply(lambda x: ' '.join(x))
  tfidf = TfidfVectorizer()
  tfidf_embedding = tfidf.fit_transform(df['text'])
  tfidf_embedding = pd.DataFrame(tfidf_embedding.toarray())
  tfidf_embedding.insert(loc=0, column="ID", value=df.index)
  
  return tfidf_embedding
#################### text embeddings COUNT ####################
def GenerateCountEmbedding(req_id, job_text_df, resume_text_df):
  pos_jd_text = job_text_df
  pos_jd_text.rename(columns = {'Req ID':'ID','Job Description':'text'}, inplace=True)
  pos_jd_text.ID = req_id
  pos_jd_text = pos_jd_text[['ID', 'text']]
  pos_resume_text = resume_text_df
  pos_resume_text.rename(columns = {'Candidate ID':'ID','Resume Text':'text'}, inplace=True)
  pos_resume_text = pos_resume_text[['ID', 'text']]
  
  #append to same df
  df = pos_jd_text.append(pos_resume_text)
  df.set_index('ID', inplace=True)
  
  # join words and vectorize
  tokenizer = RegexpTokenizer(r'\w+')
  df['text'] = df['text'].apply(lambda x: tokenizer.tokenize(x))
  df['text'] = df['text'].apply(lambda x: ' '.join(x))
  count = CountVectorizer()
  pos_embedding = count.fit_transform(df['text'])
  pos_embedding = pd.DataFrame(pos_embedding.toarray())
  pos_embedding.insert(loc=0, column="ID", value=df.index)
  
  return pos_embedding

#################### text embeddings D2V ####################
def Generate_Doc2Vec_Embeddings(Req, job_text_df, resume_text_df, vectorSize=300):
  jobs = job_text_df
  obs = resume_text
  jobs.rename(columns = {'Req ID':'ID','Job Description':'text'}, inplace=True)
  obs.rename(columns = {'Candidate ID':'ID','Resume Text':'text'}, inplace=True)
  final_frame = pd.concat([jobs[["ID","text"]], obs[['ID', 'text']]]).reset_index(drop=True)

  tupleJobDocs = []
  analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
  for i,text in enumerate(final_frame.text):
    lowerwords = re.sub(r'[^a-z ]','',text).split()
    tags = [i]
    tupleJobDocs.append(analyzedDocument(lowerwords,tags))
    
  modelJOB = doc2vec.Doc2Vec(tupleJobDocs, vector_size = vectorSize, min_count = 1, workers = 4)

  temp_df = pd.DataFrame(columns = list(range(vectorSize)))
  for i in range(len(final_frame)):
    temp_df = temp_df.append(pd.DataFrame(modelJOB.docvecs[i]).T, ignore_index=True)

  final_frame =  pd.concat([final_frame.ID, temp_df], axis = 1)

  return final_frame


#################### Read data ####################
# read structure + one hot encoded dfs
job_dummies = pd.read_csv("~/data/job_description_one_hot_FULL.csv")
job_dummies_ideal = pd.read_csv("~/data/job_description_one_hot_ideal_FULL.csv")
resume_dummies = pd.read_csv("~/data/resume_summary_one_hot.csv")

# just the structured
job_features = job_dummies[job_dummies.columns[:44]]
job_features_ideal = job_dummies_ideal[job_dummies_ideal.columns[:44]]
resume_features = resume_dummies[resume_dummies.columns[:47]]

# just the one hot
job_dummies.drop(job_dummies.columns[1:44], axis=1, inplace=True)
job_dummies_ideal.drop(job_dummies_ideal.columns[1:44], axis=1, inplace=True)
resume_dummies.drop(resume_dummies.columns[2:47], axis=1, inplace=True)

job_text = pd.read_csv("~/data/full_requisition_data_tokenized.csv").fillna('')
resume_text = pd.read_csv("~/data/Candidate Report_tokenized.csv").fillna('')

resume_dummies.rename(columns = {'CanID':'ID'}, inplace=True)
job_dummies['ID'] =job_dummies['ReqID'] 
job_dummies = job_dummies[resume_dummies.columns] #so they are in the same order
job_dummies_ideal['ID'] =job_dummies_ideal['ReqID']
job_dummies_ideal = job_dummies_ideal[resume_dummies.columns]

all_dummies = pd.concat([resume_dummies, job_dummies])
all_dummies_ideal = pd.concat([resume_dummies, job_dummies_ideal])

resume_text = resume_text[resume_text['Resume Text'] != '[\'nan\']']
resume_dummies = resume_dummies[resume_dummies.ID.isin(resume_text['Candidate ID'])]
all_dummies = pd.concat([resume_dummies, job_dummies])
all_dummies_ideal = pd.concat([resume_dummies, job_dummies_ideal])

#keep job IDs that(1)had at least one candidate with a resume looked at,
#                 (2)at least 5 applicants with resumes

jobs_reviewed_atleast_once = ['Review', 'Completion',  'Phone Screen',
                              'Schedule Interview', 'Offer Rejected',
                              'Schedule interview', 
                              'No Show (Interview / First Day)', 'Offer',
                              'Second Round Interview', 
                              'Background Check', 'Revise Offer',
                              'Final Round Interview']
temp_df = resume_text[resume_text['Latest Recruiting Step'].isin(jobs_reviewed_atleast_once)]
temp_df = temp_df[temp_df['Resume Text'] !=  '[\'nan\']']
x = temp_df[['Req ID', 'Candidate ID','Resume Text']]
x = x.merge(job_text, how='left',on='Req ID')
x = x['Req ID'].value_counts()
x = x[x >= 5]
jobIDs = x.index
K=5


vectorSize = 300
### Combine with TFIDF embedding ###`

PlaceboTest = "All"
D2Vembedding = Generate_Doc2Vec_Embeddings(PlaceboTest, job_text, resume_text, vectorSize)
all_dummies_ideal = all_dummies_ideal.drop(columns = ["ReqID"],axis=1)
D2V_onehot = D2Vembedding.merge(all_dummies_ideal#                                   , how="left"
                                   , on="ID")
#tfidfembedding = GenerateTfidfEmbedding(PlaceboTest, job_text, resume_text)
#tfidf_onehot = tfidfembedding.merge(all_dummies_ideal
#                                   , how="left"
#                                   , on="ID")
#countembedding = GenerateCountEmbedding(PlaceboTest, job_text, resume_text)
#count_onehot = countembedding.merge(all_dummies_ideal
#                                   , how="left"
#                                   , on="ID")

# free memory
del job_text
del resume_text
del all_dummies
del job_dummies
del resume_dummies
del job_features
del job_features_ideal
del resume_features
gc.collect()

### create and train LightFM model ###
NUM_THREADS = 4
NUM_COMPONENTS = 30
NUM_EPOCHS = 5
ITEM_ALPHA = 1e-6

vectorSize = 300
k_num = 5


model = LightFM(loss='warp'
                    , item_alpha=ITEM_ALPHA
                    , no_components=NUM_COMPONENTS)


############################ MODELS ############################
############################ One Hot Only ############################
pos_ = all_dummies_ideal.dropna()
pos_spr = sp.sparse.csr_matrix(pos_.set_index("ID").values)
pos_train, pos_test = random_train_test_split(pos_spr
                                                , test_percentage=0.25
                                                , random_state = None)

%time model = model.fit(pos_spr, epochs=NUM_EPOCHS, num_threads=NUM_THREADS)

#train_auc = auc_score(model, pos_train, num_threads=NUM_THREADS).mean()

train_precision = precision_at_k(model, pos_train, user_features=job_features_sparse, item_features=resume_features_sparse, k=K_num).mean()
print('train precision at k: %s' %train_precision)
test_precision = precision_at_k(model, pos_test, user_features=job_features_sparse, item_features=resume_features_sparse, k=K_num).mean()
print('test precision at k: %s' %test_precision)

############################ TEXT MODELS ############################
############################ Count Vector ONLY ############################
#pos_ = countembedding.dropna()
#pos_spr = sp.sparse.csr_matrix(pos_.set_index("ID").values)
#pos_train, pos_test = random_train_test_split(pos_spr
#                                                , test_percentage=0.25
#                                                , random_state = None)

#%time pos_model = pos_model.fit(pos_spr, epochs=NUM_EPOCHS, num_threads=NUM_THREADS)

#train_auc = auc_score(pos_model, pos_, num_threads=NUM_THREADS).mean()

#train_precision = precision_at_k(pos_model, pos_train, user_features=job_features_sparse, item_features=resume_features_sparse, k=K_num).mean()
#print('train precision at k: %s' %train_precision)
#test_precision = precision_at_k(pos_model, pos_test, user_features=job_features_sparse, item_features=resume_features_sparse, k=K_num).mean()
#print('test precision at k: %s' %test_precision)

############################ TFIDF Vector ONLY ############################
pos_ = tfidfembedding.dropna()
pos_spr = sp.sparse.csr_matrix(pos_.set_index("ID").values)
pos_train, pos_test = random_train_test_split(pos_spr
                                                , test_percentage=0.25
                                                , random_state = None)

%time model = model.fit(pos_spr, epochs=NUM_EPOCHS, num_threads=NUM_THREADS)

#train_auc = auc_score(model, pos_train, num_threads=NUM_THREADS).mean()

train_precision = precision_at_k(model, pos_train, user_features=job_features_sparse, item_features=resume_features_sparse, k=K_num).mean()
print('train precision at k: %s' %train_precision)
test_precision = precision_at_k(model, pos_test, user_features=job_features_sparse, item_features=resume_features_sparse, k=K_num).mean()
print('test precision at k: %s' %test_precision)

############################ TFIDF Vector ONLY ############################
#pos_ = D2Vembedding.dropna()
#pos_spr = sp.sparse.csr_matrix(pos_.set_index("ID").values)
#pos_train, pos_test = random_train_test_split(pos_spr
#                                                , test_percentage=0.25
#                                                , random_state = None)

#%time pos_model = pos_model.fit(pos_spr, epochs=NUM_EPOCHS, num_threads=NUM_THREADS)

#train_auc = auc_score(pos_model, pos_, num_threads=NUM_THREADS).mean()

#train_precision = precision_at_k(pos_model, pos_train, user_features=job_features_sparse, item_features=resume_features_sparse, k=K_num).mean()
#print('train precision at k: %s' %train_precision)
#test_precision = precision_at_k(pos_model, pos_test, user_features=job_features_sparse, item_features=resume_features_sparse, k=K_num).mean()
#print('test precision at k: %s' %test_precision)

############################ COMBINED MODELS ############################
############################ Count Vector ONLY ############################
#pos_ = count_onehot.dropna()
#pos_spr = sp.sparse.csr_matrix(pos_.set_index("ID").values)
#pos_train, pos_test = random_train_test_split(pos_spr
#                                                , test_percentage=0.25
#                                                , random_state = None)

#%time model = model.fit(pos_spr, epochs=NUM_EPOCHS, num_threads=NUM_THREADS)

#train_auc = auc_score(model, pos_, num_threads=NUM_THREADS).mean()

#train_precision = precision_at_k(model, pos_train, user_features=job_features_sparse, item_features=resume_features_sparse, k=K_num).mean()
#print('train precision at k: %s' %train_precision)
#test_precision = precision_at_k(model, pos_test, user_features=job_features_sparse, item_features=resume_features_sparse, k=K_num).mean()
#print('test precision at k: %s' %test_precision)

############################ TFIDF Vector  ############################
#pos_ = tfidf_onehot.dropna()
#pos_spr = sp.sparse.csr_matrix(pos_.set_index("ID").values)
#pos_train, pos_test = random_train_test_split(pos_spr
#                                                , test_percentage=0.25
#                                                , random_state = None)

#%time model = model.fit(pos_spr, epochs=NUM_EPOCHS, num_threads=NUM_THREADS)

#train_auc = auc_score(pos_model, pos_, num_threads=NUM_THREADS).mean()

#train_precision = precision_at_k(model, pos_train, user_features=job_features_sparse, item_features=resume_features_sparse, k=K_num).mean()
#print('train precision at k: %s' %train_precision)
#test_precision = precision_at_k(model, pos_test, user_features=job_features_sparse, item_features=resume_features_sparse, k=K_num).mean()
#print('test precision at k: %s' %test_precision)

############################ D2V Vector  ############################
pos_ = D2V_onehot.dropna()
pos_spr = sp.sparse.csr_matrix(pos_.set_index("ID").values)
pos_train, pos_test = random_train_test_split(pos_spr
                                                , test_percentage=0.25
                                                , random_state = None)

%time model = model.fit(pos_spr, epochs=NUM_EPOCHS, num_threads=NUM_THREADS)

train_auc = auc_score(model, pos_train, num_threads=NUM_THREADS).mean()

train_precision = precision_at_k(model, pos_train, user_features=job_features_sparse, item_features=resume_features_sparse, k=K_num).mean()
print('train precision at k: %s' %train_precision)
test_precision = precision_at_k(model, pos_test, user_features=job_features_sparse, item_features=resume_features_sparse, k=K_num).mean()
print('test precision at k: %s' %test_precision)