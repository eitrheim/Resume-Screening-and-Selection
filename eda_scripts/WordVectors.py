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

#job_embeddings = generate_embeddings("job", job_text, 300)
#job_embeddings.to_csv("data/Job_Embeddings.csv")

#resume_embeddings = generate_embeddings("res", resume_text, 300)
#resume_embeddings.to_csv("data/Resume_Embeddings.csv")

def generate_embeddingsFULL(Req, job_text_df, resume_text_df, vectorSize=300):
  #import data
  All = ["All", "all"]
  if Req in All:
    obs = resume_text_df
    jobs = job_text_df
    #print("All")
  else:
    jobs = job_text_df[job_text_df["Req ID"]==Req]
    obs = resume_text[resume_text_df["Req ID"]==Req]
    #print("Specific")

  jobs.rename(columns = {'Req ID':'ID','Job Description':'text'}, inplace=True)
  #print("a")
  jobs = jobs[["ID","text"]]
  #print("b")
  obs.rename(columns = {'Candidate ID':'ID','Resume Text':'text'}, inplace=True)
  #print("c")
  obs = obs.drop(obs.columns[0], axis = 1)
  #print("d")
  obs = obs[['ID', 'text']]
  #print("e")

  
  frames = [jobs, obs]
  #print("f")
  final_frame = pd.concat(frames)
  #print("g")

  ID_df = final_frame.ID
  #print("h")
  ID_df.index = range(len(final_frame))
  #print("i")
  final_frame = final_frame.text
  #print("j")
  #print(final_frame.shape)
  #print("k")

  #Data Cleaning
  final_frame = final_frame.astype(str)
  final_frame = final_frame.str.replace(r'\n','')
  final_frame.replace(regex=True,inplace=True,to_replace=EMAIL_REGEX, value = r'')
  final_frame.replace(regex=True,inplace=True,to_replace=PHONE_REGEX, value = r'')
  final_frame.replace(regex=True,inplace=True,to_replace=NAME_REGEX, value = r'')
  #print("l")

  final_frame.text = final_frame.apply(remove_punctuation)
  #print("m")
  final_frame.dropna(axis=0, inplace=True)
  #print("n")
  jobdocs = []
  #print("1")
  for job in final_frame:
    jobdocs.append(job)
  
  tupleJobDocs = []
  #print("o")
  analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
  for i,text in enumerate(jobdocs):
    lowerwords = text.lower().split()
    filteredwords = [word for word in lowerwords if word not in stopwords]
    tags = [i]
    tupleJobDocs.append(analyzedDocument(filteredwords,tags))
  modelJOB = doc2vec.Doc2Vec(tupleJobDocs, vector_size = vectorSize, min_count = 1, workers = 4)
  #print("p")
  columns = list(range(vectorSize))
  #print("q")
  embeddings_df_ = pd.DataFrame(columns = columns)
  #print("r")
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
  print(final_frame.shape)
  return final_frame
  
full_embeddings = generate_embeddingsFULL("all", job_text, resume_text, 300)
full_embeddings.to_csv("data/Full_Embeddings.csv")  


def Generate_Doc2Vec_Embeddings(Req, job_text_df, resume_text_df, vectorSize=300):
  jobs = job_text_df[job_text_df["Req ID"]==Req]
  obs = resume_text[resume_text_df["Req ID"]==Req]
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