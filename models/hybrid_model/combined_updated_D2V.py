import pandas as pd
import numpy as np
import string
#import nltk
from nltk.tokenize import word_tokenize
from spacy.lang.en.stop_words import STOP_WORDS
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from gensim.models import doc2vec
from collections import namedtuple
from lightfm import LightFM
from lightfm.cross_validation import random_train_test_split
import scipy as sp

# hide settingwithcopywarning
pd.options.mode.chained_assignment = None

stopwords = STOP_WORDS
EMAIL_REGEX = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,4}"
PHONE_REGEX = r"\(?(\d{3})?\)?[\s\.-]{0,2}?(\d{3})[\s\.-]{0,2}(\d{4})"
NAME_REGEX = r'[A-Z]([a-z]+|\.)(?:\s+[A-Z]([a-z]+|\.))*(?:\s+[a-z][a-z\-]+){0,2}\s+[A-Z](a-z+|\.)'


def remove_punctuation(text):
  for punctuation in string.punctuation:
    text = text.replace(punctuation, '')
  return text
  
def generate_embeddings(ReqID, job_text_df, resume_text_df, vectorSize=300):
  #import data
  if ReqID == "All" or "all":
    obs = resume_text_df
    jobs = job_text_df
  else:
    jobs = job_text_df[job_text_df["Req ID"]==ReqID]
    obs = resume_text[resume_text_df["Req ID"]==ReqID]
  print(jobs.columns)
  print(obs.columns)
  jobs.rename(columns = {'Req ID':'ID','Job Description':'text'}, inplace=True)
  jobs = jobs[["ID","text"]]
  obs.rename(columns = {'Candidate ID':'ID','Resume Text':'text'}, inplace=True)
  obs = obs.drop(obs.columns[0], axis = 1)
  obs = obs[['ID', 'text']]
  
  print(jobs.columns)
  print(obs.columns)
  
  frames = [jobs, obs]
  final_frame = pd.concat(frames)

  final_frame
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





job_dummies = pd.read_csv("~/data/job_description_one_hot.csv")
job_dummies_ideal = pd.read_csv("~/data/job_description_one_hot_ideal.csv")
job_dummies_FULL = pd.read_csv("~/data/job_description_one_hot_FULL.csv")
job_dummies_ideal_FULL = pd.read_csv("~/data/job_description_one_hot_ideal_FULL.csv")
resume_dummies = pd.read_csv("~/data/resume_summary_one_hot.csv")

# just the one hot
job_dummies.drop(job_dummies.columns[1:45], axis=1, inplace=True)
job_dummies_ideal.drop(job_dummies_ideal.columns[1:45], axis=1, inplace=True)
job_dummies_FULL.drop(job_dummies_FULL.columns[1:45], axis=1, inplace=True)
job_dummies_ideal_FULL.drop(job_dummies_ideal_FULL.columns[1:45], axis=1, inplace=True)
resume_dummies.drop(resume_dummies.columns[2:47], axis=1, inplace=True)

job_text = pd.read_csv("~/data/full_requisition_data_tokenized.csv").fillna('')
resume_text = pd.read_csv("~/data/Candidate Report_tokenized.csv").fillna('')
job_text_clean = job_text.drop('Job Description',axis=1).rename(columns={'Job Description Clean': 'Job Description'})


#drop all rows that do not have a resume
resume_text = resume_text[resume_text['Resume Text'] != '[\'nan\']']
resume_dummies = resume_dummies[resume_dummies.CanID.isin(resume_text['Candidate ID'])]
all_dummies = pd.concat([resume_dummies, job_dummies])
all_dummies_ideal = pd.concat([resume_dummies, job_dummies_ideal])
all_dummies_FULL = pd.concat([resume_dummies, job_dummies_FULL])
all_dummies_ideal_FULL = pd.concat([resume_dummies, job_dummies_ideal_FULL])
full_dummies = pd.concat([resume_dummies, job_dummies_ideal])

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

# ...  and have 100+ applicants
temp_df = x[x >= 100].index
jobIDs = temp_df
K=25

# ...  and have 60-100 applicants
temp_df = x[x >= 60][x < 100].index
jobIDs = temp_df
K=20

# ... and have 40-60 applicants
temp_df = x[x >= 40][x < 60].index
jobIDs = temp_df
K=15

# ... and have 20-40 applicants
temp_df = x[x >= 20][x < 40].index
jobIDs = temp_df
K=10

# ... and have 10-20 applicants
temp_df = x[x >= 10][x < 20].index
jobIDs = temp_df
K=5

### create and train LightFM model ###
NUM_THREADS = 4
NUM_COMPONENTS = 30
NUM_EPOCHS = 5
ITEM_ALPHA = 1e-6

vectorSize = 300

pos_model = LightFM(loss='warp'
                    , item_alpha=ITEM_ALPHA
                    , no_components=NUM_COMPONENTS)

### Combine with TFIDF embedding ###
job_text = job_text.head(500)
resume_text = resume_text.head(1000)
PlaceboTest = "All"
test = generate_embeddings(PlaceboTest, job_text, resume_text, vectorSize)
all_dummies_ideal_FULL = all_dummies_ideal_FULL.drop(columns = ["ReqID"],axis=1)
pos_ = test.merge(all_dummies_ideal_FULL
                                   , how="left"
                                   , on="ID")
pos_ = pos_.dropna()
pos_spr = sp.sparse.csr_matrix(pos_.set_index("ID").values)
pos_train, pos_test = random_train_test_split(pos_spr
                                                , test_percentage=0.25
                                                , random_state = None)

%time pos_model = pos_model.fit(pos_spr, epochs=NUM_EPOCHS, num_threads=NUM_THREADS)

train_auc = auc_score(pos_model, pos_, num_threads=NUM_THREADS).mean()
print('train AUC: %s' %train_auc)
test_auc = auc_score(pos_model, pos_, num_threads=NUM_THREADS).mean()
print('test AUC: %s' %test_auc)

#################### 4 original resume and FULL job description with td-ifd embeddings ####################
stages = []
for one_job_id in jobIDs:
  pos_tfidf = GenerateTfidfEmbedding(one_job_id, job_text, resume_text)
  pos_tfidf = RecommendTopX(jobID=one_job_id, full_df=pos_tfidf, num_x=K)
  pos_tfidf = pos_tfidf.merge(resume_text[resume_text['Req ID'] == one_job_id], on='Candidate ID',how='left')
  
  stages.append(pos_tfidf['Latest Recruiting Step'].values)

stages = [item for sublist in stages for item in sublist]
stages = pd.DataFrame(stages, columns=['Stages'])['Stages'].value_counts()
good_picks = stages[stages.index.isin(jobs_reviewed_atleast_once)]
p_at_k4 = sum(good_picks)/sum(stages.values)
p_at_k4
outcomes.update({4: p_at_k4})

#################### 6 original resume and Matt's parsed job description with td-ifd embeddings ####################
stages = []
for one_job_id in jobIDs:
  pos_tfidf_clean_resume = GenerateTfidfEmbedding(one_job_id, job_text_clean, resume_text)
  pos_tfidf_clean_resume = RecommendTopX(jobID=one_job_id, full_df=pos_tfidf_clean_resume, num_x=K)
  pos_tfidf_clean_resume = pos_tfidf_clean_resume.merge(resume_text[resume_text['Req ID'] == one_job_id], on='Candidate ID',how='left')
  
  stages.append(pos_tfidf_clean_resume['Latest Recruiting Step'].values)

stages = [item for sublist in stages for item in sublist]
stages = pd.DataFrame(stages, columns=['Stages'])['Stages'].value_counts()
good_picks = stages[stages.index.isin(jobs_reviewed_atleast_once)]
p_at_k6 = sum(good_picks)/sum(stages.values)
p_at_k6 
outcomes.update({6: p_at_k6})

#################### 8 original resume and FULL job description with td-ifd embeddings and one hot ####################
stages = []
for one_job_id in jobIDs:
  pos_tfidf = GenerateTfidfEmbedding(one_job_id, job_text, resume_text)
  pos_tfidf['ReqID'] = np.repeat(one_job_id,len(pos_tfidf))
  pos_tfidf_with_hot = pd.DataFrame(pos_tfidf).merge(all_dummies_FULL, how="left", on=["ID",'ReqID'])
  pos_tfidf_with_hot = RecommendTopX(jobID=one_job_id, full_df=pos_tfidf_with_hot.drop('ReqID',axis=1), num_x=K)
  pos_tfidf_with_hot = pos_tfidf_with_hot.merge(resume_text[resume_text['Req ID'] == one_job_id], on='Candidate ID',how='left')
  
  stages.append(pos_tfidf_with_hot['Latest Recruiting Step'].values)

stages = [item for sublist in stages for item in sublist]
stages = pd.DataFrame(stages, columns=['Stages'])['Stages'].value_counts()
good_picks = stages[stages.index.isin(jobs_reviewed_atleast_once)]
p_at_k8 = sum(good_picks)/sum(stages.values)
p_at_k8 
outcomes.update({8: p_at_k8})

#################### 10 original resume and job description with td-ifd embeddings and one hot encoding but with our alternations ####################
stages = []
for one_job_id in jobIDs:
  pos_tfidf = GenerateTfidfEmbedding(one_job_id, job_text, resume_text)
  pos_tfidf['ReqID'] = np.repeat(one_job_id,len(pos_tfidf))
  pos_tfidf_with_hot_ideal = pd.DataFrame(pos_tfidf).merge(all_dummies_ideal_FULL, how="left", on=["ID",'ReqID'])
  pos_tfidf_with_hot_ideal = RecommendTopX(jobID=one_job_id, full_df=pos_tfidf_with_hot_ideal.drop('ReqID',axis=1), num_x=K)
  pos_tfidf_with_hot_ideal = pos_tfidf_with_hot_ideal.merge(resume_text[resume_text['Req ID'] == one_job_id], on='Candidate ID',how='left')
  
  stages.append(pos_tfidf_with_hot_ideal['Latest Recruiting Step'].values)

stages = [item for sublist in stages for item in sublist]
stages = pd.DataFrame(stages, columns=['Stages'])['Stages'].value_counts()
good_picks = stages[stages.index.isin(jobs_reviewed_atleast_once)]
p_at_k10 = sum(good_picks)/sum(stages.values)
p_at_k10 
outcomes.update({10: p_at_k10})

#################### 12 original resume and Matt's parsed job description with td-ifd embeddings and one hot encoding but with our alternations ####################
stages = []
for one_job_id in jobIDs:
  pos_tfidf = GenerateTfidfEmbedding(one_job_id, job_text_clean, resume_text)
  pos_tfidf['ReqID'] = np.repeat(one_job_id,len(pos_tfidf))
  pos_tfidf_with_hot_ideal = pd.DataFrame(pos_tfidf).merge(all_dummies_ideal, how="left", on=["ID",'ReqID'])
  pos_tfidf_with_hot_ideal = RecommendTopX(jobID=one_job_id, full_df=pos_tfidf_with_hot_ideal.drop('ReqID',axis=1), num_x=K)
  pos_tfidf_with_hot_ideal = pos_tfidf_with_hot_ideal.merge(resume_text[resume_text['Req ID'] == one_job_id], on='Candidate ID',how='left')
  
  stages.append(pos_tfidf_with_hot_ideal['Latest Recruiting Step'].values)

stages = [item for sublist in stages for item in sublist]
stages = pd.DataFrame(stages, columns=['Stages'])['Stages'].value_counts()
good_picks = stages[stages.index.isin(jobs_reviewed_atleast_once)]
p_at_k12 = sum(good_picks)/sum(stages.values)
p_at_k12 
outcomes.update({12: p_at_k12})
 
  # Drop columns that are all 0
  #tdidf_select_hot = tdidf_select_hot.T[tdidf_select_hot.any()].T
  #use stem and top uni and other qual
  
  #mos_experience
  #certifications
  #major_minor
  #degree level
  #technical skills
  #company industry

  
  


#group jobs by kind and get an average ideal candidate

outcomes
sorted(outcomes.items(), key=lambda item: item[1])




