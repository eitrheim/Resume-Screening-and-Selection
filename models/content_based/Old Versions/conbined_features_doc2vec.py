import pandas as pd
import numpy as np
#import nltk
from nltk.tokenize import word_tokenize
from spacy.lang.en.stop_words import STOP_WORDS
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
# hide settingwithcopywarning
pd.options.mode.chained_assignment = None
import string
from collections import namedtuple
from gensim.models import doc2vec

#################### Read data ####################
# read structure + one hot encoded dfs
job_dummies = pd.read_csv("~/data/job_description_one_hot.csv")
job_dummies_ideal = pd.read_csv("~/data/job_description_one_hot_ideal.csv")
job_dummies_FULL = pd.read_csv("~/data/job_description_one_hot_FULL.csv")
job_dummies_ideal_FULL = pd.read_csv("~/data/job_description_one_hot_ideal_FULL.csv")
resume_dummies = pd.read_csv("~/data/resume_summary_one_hot.csv")

# just the structured
job_features = job_dummies[job_dummies.columns[:45]]
job_features_ideal = job_dummies_ideal[job_dummies_ideal.columns[:45]]
job_features_FULL = job_dummies_FULL[job_dummies_FULL.columns[:45]]
job_features_ideal_FULL = job_dummies_ideal_FULL[job_dummies_ideal_FULL.columns[:45]]
resume_features = resume_dummies[resume_dummies.columns[:47]]

# just the one hot
job_dummies.drop(job_dummies.columns[1:45], axis=1, inplace=True)
job_dummies_ideal.drop(job_dummies_ideal.columns[1:45], axis=1, inplace=True)
job_dummies_FULL.drop(job_dummies_FULL.columns[1:45], axis=1, inplace=True)
job_dummies_ideal_FULL.drop(job_dummies_ideal_FULL.columns[1:45], axis=1, inplace=True)
resume_dummies.drop(resume_dummies.columns[2:47], axis=1, inplace=True)

##saved this commented out chunk below since it takes some time
## read raw df text data and vectorize for embeddings
#def remove_stop_words(x):
#  try:
#    x = re.sub('\s\s+',' ',x.replace(' - ',' '))
#    word_tokens = word_tokenize(x)
#    filtered_sentence = [w for w in word_tokens if not w in STOP_WORDS]
#    return filtered_sentence
#  except:
#    print(x.values)
#    sys.exit(1)
#    
#job_text = pd.read_csv("~/data/full_requisition_data.csv")
#job_text = job_text[['Req ID', 'Req Title', 'Job Requisition Status', 'Candidate ID','Division', 'Function', 'Job Description']]
#job_text["Job Description"].replace(r'[\d]','',regex=True, inplace=True)
#job_text["Job Description"] = job_text["Job Description"].astype(str).apply(lambda x: re.sub('[^a-z- ]','',x.lower().replace('\n',' ').replace('\t',' ')))
#job_text["Job Description"] = job_text["Job Description"].apply(remove_stop_words)
#
#job_text["Job Description Clean"] = job_text.merge(job_features[['ReqID', 'text']],how='left',left_on='Req ID', right_on='ReqID')['text']
#job_text["Job Description Clean"].replace(r'[\d]','',regex=True, inplace=True)
#job_text["Job Description Clean"] = job_text["Job Description Clean"].astype(str).apply(lambda x: re.sub('[^a-z- ]','',x.lower().replace('\n',' ').replace('\t',' ')))
##removing 1 and 2 letter words
#job_text["Job Description Clean"] = job_text["Job Description Clean"].apply(lambda x: re.sub('\s\w{1,2}\s',' ',x))
#job_text["Job Description Clean"] = job_text["Job Description Clean"].apply(remove_stop_words)
#for i in job_text.index:
#  if job_text["Job Description Clean"][i][0] == 'nan':
#    job_text["Job Description Clean"][i] = job_text["Job Description"][i]
#
#resume_text = pd.read_csv('~/data/Candidate Report.csv', encoding = 'latin-1')
#resume_text = resume_text[['Req ID', 'Candidate ID', 'Latest Recruiting Step', 'Last Recruiting Stage', 'Resume Text']]
#resume_text["Resume Text"].replace(r'[\d]','',regex=True, inplace=True) # remove numbers
#resume_text["Resume Text"] = resume_text["Resume Text"].astype(str).apply(lambda x: re.sub('[^a-z- ]','',x.lower().replace('\n',' ').replace('\t',' ')))
##removing 1 and 2 letter words
#resume_text["Resume Text"] = resume_text["Resume Text"].apply(lambda x: re.sub('\s\w{1,2}\s',' ',x))
#resume_text["Resume Text"] = resume_text["Resume Text"].apply(remove_stop_words)
#
##saving so we don't have to do it everytime
#job_text.to_csv("~/data/full_requisition_data_tokenized.csv", index=False)
#resume_text.to_csv('~/data/Candidate Report_tokenized.csv', index=False)

job_text = pd.read_csv("~/data/full_requisition_data_tokenized.csv").fillna('')
resume_text = pd.read_csv("~/data/Candidate Report_tokenized.csv").fillna('')
job_text_clean = job_text.drop('Job Description',axis=1).rename(columns={'Job Description Clean': 'Job Description'})

stopwords = STOP_WORDS
EMAIL_REGEX = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,4}"
PHONE_REGEX = r"\(?(\d{3})?\)?[\s\.-]{0,2}?(\d{3})[\s\.-]{0,2}(\d{4})"
NAME_REGEX = r'[A-Z]([a-z]+|\.)(?:\s+[A-Z]([a-z]+|\.))*(?:\s+[a-z][a-z\-]+){0,2}\s+[A-Z](a-z+|\.)'


def remove_punctuation(text):
  for punctuation in string.punctuation:
    text = text.replace(punctuation, '')
  return text
  
def generate_embeddings(Req, job_text_df, resume_text_df, vectorSize=300):
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

#################### combining embedding with dummies ####################
#list(set(list(resume_dummies.columns))-set(list(job_dummies.columns)))
resume_dummies.rename(columns = {'CanID':'ID'}, inplace=True)
job_dummies['ID'] =job_dummies['ReqID'] 
job_dummies = job_dummies[resume_dummies.columns] #so they are in the same order
job_dummies_ideal['ID'] =job_dummies_ideal['ReqID']
job_dummies_ideal = job_dummies_ideal[resume_dummies.columns]

job_dummies_FULL['ID'] = job_dummies_FULL['ReqID']
job_dummies_FULL = job_dummies_FULL[resume_dummies.columns]
job_dummies_ideal_FULL['ID'] = job_dummies_ideal_FULL['ReqID']
job_dummies_ideal_FULL = job_dummies_ideal_FULL[resume_dummies.columns]
all_dummies = pd.concat([resume_dummies, job_dummies])
all_dummies_ideal = pd.concat([resume_dummies, job_dummies_ideal])
all_dummies_FULL = pd.concat([resume_dummies, job_dummies_FULL])
all_dummies_ideal_FULL = pd.concat([resume_dummies, job_dummies_ideal_FULL])

#################### Cos Sim and rank candidates ####################
def RecommendTopX(jobID, full_df, num_x=10):
  #returns x recommended resume ID's based on Job Description
  recommended_candidates = []
  candidates_cosine = []
  full_df.fillna(0, inplace=True)
  #full_df.reset_index(inplace=True, drop=True)
  indices = pd.Series(full_df["ID"])
  cos_sim = cosine_similarity(full_df.drop("ID", axis=1)) #pairwise similarities for all samples in the df
  try:
    idx = indices[indices == jobID].index[0]
    score_series = pd.Series(cos_sim[idx]).sort_values(ascending=False)
    top_x = list(score_series.iloc[1:num_x+1].index)
    candidates_cosine = score_series[1:num_x+1].values

    for i in top_x:
      recommended_candidates.append(list(indices)[i])
  except IndexError:
    print(jobID, 'had and index error')
    
  return pd.DataFrame({'Candidate ID':recommended_candidates,
                       'cosine':candidates_cosine})

#################### PRECISION AT K for 12 diff models ####################
#drop all rows that do not have a resume
resume_text = resume_text[resume_text['Resume Text'] != '[\'nan\']']
resume_dummies = resume_dummies[resume_dummies.ID.isin(resume_text['Candidate ID'])]
all_dummies = pd.concat([resume_dummies, job_dummies])
all_dummies_ideal = pd.concat([resume_dummies, job_dummies_ideal])
all_dummies_FULL = pd.concat([resume_dummies, job_dummies_FULL])
all_dummies_ideal_FULL = pd.concat([resume_dummies, job_dummies_ideal_FULL])

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
#temp_df = x[x >= 60][x < 100].index
#jobIDs = temp_df
#K=20

# ... and have 40-60 applicants
#temp_df = x[x >= 40][x < 60].index
#jobIDs = temp_df
#K=15

# ... and have 20-40 applicants
#temp_df = x[x >= 20][x < 40].index
#jobIDs = temp_df
#K=10

# ... and have 10-20 applicants
#temp_df = x[x >= 10][x < 20].index
#jobIDs = temp_df
#K=5

vectorSize = 300

stages = []


#################### 0 baseline, just randomly picking a resume ####################
y = resume_text[resume_text['Req ID'].isin(jobIDs)]
y = y['Latest Recruiting Step'].value_counts()
sum(y[1:])/sum(y) # we consider 'Not Reviewed Not Considered' an irrelevant recs, so if we were to pick a random resume, there is a X chance it would be relevant
outcomes = {0: sum(y[1:])/sum(y)}

#################### 3 original resume and FULL job description with D2V embeddings ####################
stages = []
for one_job_id in jobIDs:

  pos_embedding = generate_embeddings(one_job_id, job_text, resume_text, vectorSize)

  pos_embedding = RecommendTopX(jobID=one_job_id, full_df=pos_embedding, num_x=K)

  pos_embedding = pos_embedding.merge(resume_text[resume_text['Req ID'] == one_job_id], on='Candidate ID',how='left')

  stages.append(pos_embedding['Latest Recruiting Step'].values)

stages = [item for sublist in stages for item in sublist]

stages = pd.DataFrame(stages, columns=['Stages'])['Stages'].value_counts()

good_picks = stages[stages.index.isin(jobs_reviewed_atleast_once)]

p_at_k3 = sum(good_picks)/sum(stages.values)

p_at_k3

outcomes.update({3: p_at_k3})
 
#################### 5 original resume and Matt's parsed job description with D2V embeddings ####################
stages = []
for one_job_id in jobIDs:
  pos_embedding_clean_resume = generate_embeddings(one_job_id, job_text_clean, resume_text, vectorSize)
  pos_embedding_clean_resume = RecommendTopX(jobID=one_job_id, full_df=pos_embedding_clean_resume, num_x=10)
  pos_embedding_clean_resume = pos_embedding_clean_resume.merge(resume_text[resume_text['Req ID'] == one_job_id], on='Candidate ID',how='left')
  
  stages.append(pos_embedding_clean_resume['Latest Recruiting Step'].values)

stages = [item for sublist in stages for item in sublist]
stages = pd.DataFrame(stages, columns=['Stages'])['Stages'].value_counts()
good_picks = stages[stages.index.isin(jobs_reviewed_atleast_once)]
p_at_k5 = sum(good_picks)/sum(stages.values)
p_at_k5 
outcomes.update({5: p_at_k5})



#################### 8 original resume and FULL job description with D2V embeddings and one hot ####################
stages = []
for one_job_id in jobIDs:
  
  pos_tfidf = generate_embeddings(one_job_id, job_text, resume_text, vectorSize)
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

#################### 9 original resume and FULL job description with D2V embeddings and one hot encoding but with our alternations ####################
stages = []
for one_job_id in jobIDs:
  pos_embedding = generate_embeddings(one_job_id, job_text, resume_text, vectorSize)
  pos_embedding['ReqID'] = np.repeat(one_job_id,len(pos_embedding))
  pos_embedding_with_hot_ideal = pd.DataFrame(pos_embedding).merge(all_dummies_ideal_FULL, how="left", on=["ID",'ReqID'])
  pos_embedding_with_hot_ideal = RecommendTopX(jobID=one_job_id, full_df=pos_embedding_with_hot_ideal.drop('ReqID',axis=1), num_x=K)
  pos_embedding_with_hot_ideal = pos_embedding_with_hot_ideal.merge(resume_text[resume_text['Req ID'] == one_job_id], on='Candidate ID',how='left')
  
  stages.append(pos_embedding_with_hot_ideal['Latest Recruiting Step'].values)

stages = [item for sublist in stages for item in sublist]
stages = pd.DataFrame(stages, columns=['Stages'])['Stages'].value_counts()
good_picks = stages[stages.index.isin(jobs_reviewed_atleast_once)]
p_at_k9 = sum(good_picks)/sum(stages.values)
p_at_k9 
outcomes.update({9: p_at_k9})

#################### 11 original resume and Matt's parsed job description with D2V embeddings and one hot encoding but with our alternations ####################
stages = []
for one_job_id in jobIDs:
  pos_embedding = generate_embeddings(one_job_id, job_text_clean, resume_text, vectorSize)
  pos_embedding['ReqID'] = np.repeat(one_job_id,len(pos_embedding))
  pos_embedding_with_hot_ideal = pd.DataFrame(pos_embedding).merge(all_dummies_ideal, how="left", on=["ID",'ReqID'])
  pos_embedding_with_hot_ideal = RecommendTopX(jobID=one_job_id, full_df=pos_embedding_with_hot_ideal.drop('ReqID',axis=1), num_x=K)
  pos_embedding_with_hot_ideal = pos_embedding_with_hot_ideal.merge(resume_text[resume_text['Req ID'] == one_job_id], on='Candidate ID',how='left')
   
  stages.append(pos_embedding_with_hot_ideal['Latest Recruiting Step'].values)

stages = [item for sublist in stages for item in sublist]
stages = pd.DataFrame(stages, columns=['Stages'])['Stages'].value_counts()
good_picks = stages[stages.index.isin(jobs_reviewed_atleast_once)]
p_at_k11 = sum(good_picks)/sum(stages.values)
p_at_k11
outcomes.update({11: p_at_k11})


















#################### 14  ####################
stages = []
for one_job_id in jobIDs:

  candid = resume_text['Candidate ID'][resume_text['Req ID']==one_job_id]
  candid = np.hstack((candid.values,one_job_id))
  resume_dummy = resume_dummies[resume_dummies.ID.isin(candid)].drop_duplicates()
  resume_dummy = resume_dummy[resume_dummy.ReqID == one_job_id]
  
  job_dummy_ideal_FULL = job_dummies_ideal_FULL[job_dummies_ideal_FULL.ReqID == one_job_id]
  #cols = job_dummy_ideal_FULL.columns
  job_dummy_ideal_FULL = pd.DataFrame(np.concatenate([job_dummy_ideal_FULL,resume_dummy])).rename(columns={0: 'ReqID', 1:'ID'})
  #job_dummy_ideal_FULL.columns = cols
  
  
  #if they need a bachelor degree:
  has_bach = [0]
  if job_dummy_ideal_FULL[job_dummy_ideal_FULL.columns[10]][0] == 1:
    for item in range(1,1 + len(job_dummy_ideal_FULL[job_dummy_ideal_FULL.columns[10]][1:])):
      if job_dummy_ideal_FULL[job_dummy_ideal_FULL.columns[10]][item] == 1:
        has_bach.append(item)
        
    has_bach_df = job_dummy_ideal_FULL[job_dummy_ideal_FULL.index.isin(has_bach)]
    has_bach = has_bach[1:]
    no_bach_df = job_dummy_ideal_FULL[~job_dummy_ideal_FULL.index.isin(has_bach)]
    
    has_bach_df = RecommendTopX(jobID=one_job_id, full_df=has_bach_df.drop('ReqID',axis=1), num_x=min(K, len(has_bach_df)))
    no_bach_df = RecommendTopX(jobID=one_job_id, full_df=no_bach_df.drop('ReqID',axis=1), num_x=min(K, len(no_bach_df)))
    
    job_dummy_ideal_FULL = pd.DataFrame(np.concatenate([has_bach_df,no_bach_df]))
    job_dummy_ideal_FULL.columns = ['Candidate ID', 'cosine']
    job_dummy_ideal_FULL = job_dummy_ideal_FULL.head(K)
    
  else:
    job_dummy_ideal_FULL = RecommendTopX(jobID=one_job_id, full_df=job_dummy_ideal_FULL.drop('ReqID',axis=1), num_x=K)
  
  job_dummy_ideal_FULL = job_dummy_ideal_FULL.merge(resume_text[resume_text['Req ID'] == one_job_id], on='Candidate ID',how='left')
  
  stages.append(job_dummy_ideal_FULL['Latest Recruiting Step'].values)  

stages = [item for sublist in stages for item in sublist]
stages = pd.DataFrame(stages, columns=['Stages'])['Stages'].value_counts()
good_picks = stages[stages.index.isin(jobs_reviewed_atleast_once)]
p_at_k14 = sum(good_picks)/sum(stages.values)
p_at_k14 





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
print("K is: ",K)
outcomes
sorted(outcomes.items(), key=lambda item: item[1])