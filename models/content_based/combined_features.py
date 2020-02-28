import pandas as pd
import numpy as np
#import nltk
#from nltk.tokenize import word_tokenize
#from spacy.lang.en.stop_words import STOP_WORDS
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from gensim.models import doc2vec
from collections import namedtuple
# hide settingwithcopywarning
pd.options.mode.chained_assignment = None

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

##for printing an example of the one hot encodings
#cols = [7,10,13,25,86,113,302,310,421,452,521]
#x = resume_dummies[resume_dummies.StemMajor == 1]
#x = x[x.columns[cols]]
#x = x[x.T.sum() != 0]
#x.loc[[9864,51372,136,104493,4753,3310]]

# saved this commented out chunk below since it takes some time
# read raw df text data and vectorize for embeddings
#EMAIL_REGEX = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,4}"
#def remove_stop_words(x):
#  try: 
#    x = re.sub('(\w)(/)(\w)',r'\1 \3',x) # for responsibilities/accountabilities
#    x = re.sub(EMAIL_REGEX,"",x) # for email
#    x = re.sub('[^a-z- ]','',x) # for punctuation
#    x = re.sub(' - ',' ',x) # for dashes
#    x = re.sub('-',' ',x) # for dashes
#    x = re.sub(' \w ',' ',x) # for single letters
#    x = re.sub('\s\s+',' ',x) # for multiple spaces
#    
#    word_tokens = word_tokenize(x)
#    filtered_sentence = [w for w in word_tokens if not w in STOP_WORDS]
#    return filtered_sentence
#  except Exception as e:
#    print("error:",e)
#    print('string:',x)
#    sys.exit(1)
#    
#job_text = pd.read_csv("~/data/full_requisition_data.csv")
#job_text = job_text[['Req ID', 'Req Title', 'Job Requisition Status', 'Candidate ID','Division', 'Function', 'Job Description']]
#job_text["Job Description"].replace(r'[\d]','',regex=True, inplace=True)
#job_text["Job Description"] = job_text["Job Description"].astype(str).apply(lambda x: x.lower().replace('\r',' ').replace('\n',' ').replace('\t',' '))
#job_text["Job Description"] = job_text["Job Description"].apply(remove_stop_words)

#resume_text = pd.read_csv('~/data/Candidate Report.csv', encoding = 'latin-1')
#resume_text = resume_text[['Req ID', 'Candidate ID', 'Latest Recruiting Step', 'Last Recruiting Stage', 'Resume Text']]
#resume_text["Resume Text"].replace(r'[\d]','',regex=True, inplace=True) # remove numbers
#resume_text["Resume Text"] = resume_text["Resume Text"].astype(str).apply(lambda x: x.lower().replace('\r',' ').replace('\n',' ').replace('\t',' '))
##removing 1 and 2 letter words
#resume_text["Resume Text"] = resume_text["Resume Text"].apply(lambda x: re.sub('\s\w{1,2}\s',' ',x))
#resume_text["Resume Text"] = resume_text["Resume Text"].apply(remove_stop_words)
#
##saving so we don't have to do it everytime
#job_text.to_csv("~/data/full_requisition_data_tokenized.csv", index=False)
#resume_text.to_csv('~/data/Candidate Report_tokenized.csv', index=False)

job_text = pd.read_csv("~/data/full_requisition_data_tokenized.csv").fillna('')
resume_text = pd.read_csv("~/data/Candidate Report_tokenized.csv").fillna('')

#################### text embedding (count) ####################
def GenerateCountEmbedding(req_id, job_text_df, resume_text_df):
  pos_jd_text = job_text[job_text["Req ID"]==req_id]
  pos_resume_text = resume_text[resume_text["Req ID"]==req_id]
  pos_jd_text.rename(columns = {'Req ID':'ID','Job Description':'text'}, inplace=True)
  pos_jd_text.ID = req_id
  pos_jd_text = pos_jd_text[['ID', 'text']]
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

#################### text embeddings TFIDF ####################
def GenerateTfidfEmbedding(req_id, job_text_df, resume_text_df):
  pos_jd_text = job_text[job_text["Req ID"]==req_id]
  pos_resume_text = resume_text[resume_text["Req ID"]==req_id]
  pos_jd_text.rename(columns = {'Req ID':'ID','Job Description':'text'}, inplace=True)
  pos_jd_text.ID = req_id
  pos_jd_text = pos_jd_text[['ID', 'text']]
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

#################### doc2vec embeddings TFIDF ####################
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

#################### combining embedding with dummies ####################
#list(set(list(resume_dummies.columns))-set(list(job_dummies.columns)))
resume_dummies.rename(columns = {'CanID':'ID'}, inplace=True)
job_dummies['ID'] =job_dummies['ReqID'] 
job_dummies = job_dummies[resume_dummies.columns] #so they are in the same order
job_dummies_ideal['ID'] =job_dummies_ideal['ReqID']
job_dummies_ideal = job_dummies_ideal[resume_dummies.columns]

all_dummies = pd.concat([resume_dummies, job_dummies])
all_dummies_ideal = pd.concat([resume_dummies, job_dummies_ideal])

##################### Jaccard Sim and rank candidates ####################
#from sklearn.metrics.pairwise import pairwise_distances
#def RecommendTopX(jobID, full_df, num_x=10):
#  #returns x recommended resume ID's based on Job Description
#  recommended_candidates = []
#  candidates_cosine = []
#  full_df.fillna(0, inplace=True)
#  #full_df.reset_index(inplace=True, drop=True)
#  indices = pd.Series(full_df["ID"])
#  cos_sim = pairwise_distances(full_df.drop("ID", axis=1), metric='hamming') #pairwise similarities for all samples in the df
#  try:
#    idx = indices[indices == jobID].index[0]
#    score_series = pd.Series(cos_sim[idx]).sort_values(ascending=False)
#    top_x = list(score_series.iloc[1:num_x+1].index)
#    candidates_cosine = score_series[1:num_x+1].values
#
#    for i in top_x:
#      recommended_candidates.append(list(indices)[i])
#  except IndexError:
#    print(jobID, 'had and index error')
#    
#  return pd.DataFrame({'Candidate ID':recommended_candidates,
#                       'jaccard':candidates_cosine})

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

#to look at diversity
diversity_df = pd.read_csv("~/data/Candidate Report.csv", encoding = 'latin-1').fillna('')
diversity_df = diversity_df[['Req ID', 'Req Title', 'Candidate ID', 'Gender', 'Ethinicity','Source', 'Job Application Source', 'Latest Recruiting Step','Last Recruiting Stage','Resume Text']]
diversity_df.Gender = diversity_df.Gender.map({"Group 1": 1, 'Group 2': 0}).astype(int)
diversity_df.columns = ['ReqID', 'ReqTitle', 'CanID', 'IsMale', 'Ethinicity','Source', 'JobSource','LatestStep', 'LastStage','Text']

#keep job IDs that(1)had at least one candidate with a resume looked at,
#                 (2)at least 5 applicants with resumes

jobs_reviewed_atleast_once = ['Review',
                              'Completion',
                              'Phone Screen',
                              'Schedule Interview',
                              'Offer Rejected',
                              'Schedule interview', 
                              'No Show (Interview / First Day)',
                              'Offer',
                              'Second Round Interview', 
                              'Background Check',
                              'Revise Offer',
                              'Final Round Interview',
                              'Voluntary Withdrew', # NEW ADDT
                              'Salary Expectations too high', # NEW ADDT
                              'Skills or Abilities'] # NEW ADDT

temp_df = resume_text[resume_text['Latest Recruiting Step'].isin(jobs_reviewed_atleast_once)]
temp_df = temp_df[temp_df['Resume Text'] !=  '[\'nan\']']
x = temp_df[['Req ID', 'Candidate ID','Resume Text']]
x = x.merge(job_text, how='left',on='Req ID')
x = x['Req ID'].value_counts()
x = x[x >= 5]
jobIDs = x.index
K=5
doc2_vec_size = 300


## ...  and have 100+ applicants
#temp_df = x[x >= 100].index
#jobIDs = temp_df
#K=25
#
# ...  and have 60-100 applicants
#temp_df = x[x >= 60][x < 100].index
#jobIDs = temp_df
#K=20
#
## ... and have 40-60 applicants
#temp_df = x[x >= 40][x < 60].index
#jobIDs = temp_df
#K=15
#
## ... and have 20-40 applicants
#temp_df = x[x >= 20][x < 40].index
#jobIDs = temp_df
#K=10
#
## ... and have 10-20 applicants
#temp_df = x[x >= 10][x < 20].index
#jobIDs = temp_df
#K=5

#################### 0 baseline, just randomly picking a resume ####################
y = resume_text[resume_text['Req ID'].isin(jobIDs)]
y = y['Latest Recruiting Step'].value_counts()
sum(y[y.index.isin(jobs_reviewed_atleast_once)])/sum(y)
outcomes = {0: sum(y[y.index.isin(jobs_reviewed_atleast_once)])/sum(y)/sum(y)}

#gender
temp_df = diversity_df[diversity_df.ReqID.isin(jobIDs)]
temp_df.IsMale.value_counts()[0]/len(temp_df)
gender_outcomes = {'PctWomenApplied': temp_df.IsMale.value_counts()[0]/len(temp_df)}
#percent of applicants that recruiters were interested in that are women
temp_df = temp_df[temp_df['LatestStep'].isin(jobs_reviewed_atleast_once)]
len(temp_df[temp_df.IsMale ==  0])/len(temp_df)
gender_outcomes.update({'PctWomenInterestedIn': len(temp_df[temp_df.IsMale ==  0])/len(temp_df)})

#ethinicity
race_outcomes = pd.DataFrame(columns=('PctApplied','PctInterestedIn'))
race_label =[]
for race in diversity_df.Ethinicity.unique():
  try:
    a = diversity_df.Ethinicity[diversity_df.ReqID.isin(jobIDs)].value_counts()[race]/sum(diversity_df.ReqID.isin(jobIDs))
  except:
    a = 0
  try:
    temp_df = diversity_df[(diversity_df['LatestStep'].isin(jobs_reviewed_atleast_once) & diversity_df.ReqID.isin(jobIDs))]
    b = len(temp_df[temp_df.Ethinicity ==  race])/len(temp_df)
  except:
    b = 0
  race_outcomes = race_outcomes.append({'PctApplied': a,'PctInterestedIn': b}, ignore_index=True)
  race_label.append(race)
race_outcomes.index = race_label
race_outcomes
  
#################### 1 only one hot encoding ####################
stages = []
diversity = []
genders = []
for one_job_id in jobIDs:
  candid = resume_text['Candidate ID'][resume_text['Req ID']==one_job_id]
  candid = np.hstack((candid.values,one_job_id))
  resume_dummy = resume_dummies[resume_dummies.ReqID == one_job_id]
  resume_dummy = resume_dummy[resume_dummy.ID.isin(candid)].drop_duplicates()
  
  job_dummy = job_dummies[job_dummies.ID == one_job_id]
  job_dummy = pd.DataFrame(np.concatenate([job_dummy,resume_dummy])).rename(columns={0: 'ReqID', 1:'ID'})
  job_dummy = RecommendTopX(jobID=one_job_id, full_df=job_dummy.drop('ReqID',axis=1), num_x=K)
  job_dummy = job_dummy.merge(resume_text[resume_text['Req ID'] == one_job_id], on='Candidate ID',how='left')
  
  stages.append(job_dummy['Latest Recruiting Step'].values)
  div_rows = (diversity_df.ReqID == one_job_id) & (diversity_df.CanID.isin(job_dummy['Candidate ID']))
  diversity.append(diversity_df.Ethinicity[div_rows].values)
  genders.append(diversity_df.IsMale[div_rows].values)

stages = [item for sublist in stages for item in sublist]
stages = pd.DataFrame(stages, columns=['Stages'])['Stages'].value_counts()
good_picks = stages[stages.index.isin(jobs_reviewed_atleast_once)]
p_at_k1 = sum(good_picks)/sum(stages.values)
p_at_k1
outcomes.update({1: p_at_k1})

#gender
genders = [item for sublist in genders for item in sublist]
(len(genders) - sum(genders))/len(genders)
gender_outcomes.update({'one': (len(genders) - sum(genders))/len(genders)})

#ethinicity
diversity = [item for sublist in diversity for item in sublist]
temp_list = []
for race in race_outcomes.index:
  try:
    temp_list.append(diversity.count(race)/len(diversity))
  except:
    temp_list.append(0)
race_outcomes['one'] = temp_list

#################### 2 only one hot encoding but with our alternations for ideal ####################
stages = []
diversity = []
genders = []
for one_job_id in jobIDs:
  candid = resume_text['Candidate ID'][resume_text['Req ID']==one_job_id]
  candid = np.hstack((candid.values,one_job_id))
  resume_dummy = resume_dummies[resume_dummies.ID.isin(candid)].drop_duplicates()
  resume_dummy = resume_dummy[resume_dummy.ReqID == one_job_id]
  
  job_dummy_ideal = job_dummies_ideal[job_dummies_ideal.ReqID == one_job_id]
  job_dummy_ideal = pd.DataFrame(np.concatenate([job_dummy_ideal,resume_dummy])).rename(columns={0: 'ReqID', 1:'ID'})
  job_dummy_ideal = RecommendTopX(jobID=one_job_id, full_df=job_dummy_ideal.drop('ReqID',axis=1), num_x=K)
  job_dummy_ideal = job_dummy_ideal.merge(resume_text[resume_text['Req ID'] == one_job_id], on='Candidate ID',how='left')
  
  stages.append(job_dummy_ideal['Latest Recruiting Step'].values)
  div_rows = (diversity_df.ReqID == one_job_id) & (diversity_df.CanID.isin(job_dummy_ideal['Candidate ID']))
  diversity.append(diversity_df.Ethinicity[div_rows].values)
  genders.append(diversity_df.IsMale[div_rows].values)

stages = [item for sublist in stages for item in sublist]
stages = pd.DataFrame(stages, columns=['Stages'])['Stages'].value_counts()
good_picks = stages[stages.index.isin(jobs_reviewed_atleast_once)]
p_at_k2 = sum(good_picks)/sum(stages.values)
p_at_k2 
outcomes.update({2: p_at_k2})

#gender
genders = [item for sublist in genders for item in sublist]
(len(genders) - sum(genders))/len(genders)
gender_outcomes.update({'two': (len(genders) - sum(genders))/len(genders)})

#ethinicity
diversity = [item for sublist in diversity for item in sublist]
temp_list = []
for race in race_outcomes.index:
  try:
    temp_list.append(diversity.count(race)/len(diversity))
  except:
    temp_list.append(0)
race_outcomes['two'] = temp_list

#################### 3 only count embeddings ####################
stages = []
diversity = []
genders = []
for one_job_id in jobIDs:
  pos_embedding = GenerateCountEmbedding(one_job_id, job_text, resume_text)
  pos_embedding = RecommendTopX(jobID=one_job_id, full_df=pos_embedding, num_x=K)
  pos_embedding = pos_embedding.merge(resume_text[resume_text['Req ID'] == one_job_id], on='Candidate ID',how='left')
  
  stages.append(pos_embedding['Latest Recruiting Step'].values)
  div_rows = (diversity_df.ReqID == one_job_id) & (diversity_df.CanID.isin(pos_embedding['Candidate ID']))
  diversity.append(diversity_df.Ethinicity[div_rows].values)
  genders.append(diversity_df.IsMale[div_rows].values)

stages = [item for sublist in stages for item in sublist]
stages = pd.DataFrame(stages, columns=['Stages'])['Stages'].value_counts()
good_picks = stages[stages.index.isin(jobs_reviewed_atleast_once)]
p_at_k3 = sum(good_picks)/sum(stages.values)
p_at_k3
outcomes.update({3: p_at_k3})

#gender
genders = [item for sublist in genders for item in sublist]
(len(genders) - sum(genders))/len(genders)
gender_outcomes.update({'three': (len(genders) - sum(genders))/len(genders)})

#ethinicity
diversity = [item for sublist in diversity for item in sublist]
temp_list = []
for race in race_outcomes.index:
  try:
    temp_list.append(diversity.count(race)/len(diversity))
  except:
    temp_list.append(0)
race_outcomes['three'] = temp_list

#################### 4 only td-ifd embeddings ####################
stages = []
diversity = []
genders = []
for one_job_id in jobIDs:
  pos_tfidf = GenerateTfidfEmbedding(one_job_id, job_text, resume_text)
  pos_tfidf = RecommendTopX(jobID=one_job_id, full_df=pos_tfidf, num_x=K)
  pos_tfidf = pos_tfidf.merge(resume_text[resume_text['Req ID'] == one_job_id], on='Candidate ID',how='left')
  
  stages.append(pos_tfidf['Latest Recruiting Step'].values)
  div_rows = (diversity_df.ReqID == one_job_id) & (diversity_df.CanID.isin(pos_tfidf['Candidate ID']))
  diversity.append(diversity_df.Ethinicity[div_rows].values)
  genders.append(diversity_df.IsMale[div_rows].values)

stages = [item for sublist in stages for item in sublist]
stages = pd.DataFrame(stages, columns=['Stages'])['Stages'].value_counts()
good_picks = stages[stages.index.isin(jobs_reviewed_atleast_once)]
p_at_k4 = sum(good_picks)/sum(stages.values)
p_at_k4
outcomes.update({4: p_at_k4})

#gender
genders = [item for sublist in genders for item in sublist]
(len(genders) - sum(genders))/len(genders)
gender_outcomes.update({'four': (len(genders) - sum(genders))/len(genders)})

#ethinicity
diversity = [item for sublist in diversity for item in sublist]
temp_list = []
for race in race_outcomes.index:
  try:
    temp_list.append(diversity.count(race)/len(diversity))
  except:
    temp_list.append(0)
race_outcomes['four'] = temp_list

#################### 5 only doc2vec embeddings ####################
stages = []
diversity = []
genders = []
for one_job_id in jobIDs:
  pos_tfidf = Generate_Doc2Vec_Embeddings(one_job_id, job_text, resume_text, vectorSize=doc2_vec_size)
  pos_tfidf = RecommendTopX(jobID=one_job_id, full_df=pos_tfidf, num_x=K)
  pos_tfidf = pos_tfidf.merge(resume_text[resume_text['Req ID'] == one_job_id], on='Candidate ID',how='left')
  
  stages.append(pos_tfidf['Latest Recruiting Step'].values)
  div_rows = (diversity_df.ReqID == one_job_id) & (diversity_df.CanID.isin(pos_tfidf['Candidate ID']))
  diversity.append(diversity_df.Ethinicity[div_rows].values)
  genders.append(diversity_df.IsMale[div_rows].values)

stages = [item for sublist in stages for item in sublist]
stages = pd.DataFrame(stages, columns=['Stages'])['Stages'].value_counts()
good_picks = stages[stages.index.isin(jobs_reviewed_atleast_once)]
p_at_k5 = sum(good_picks)/sum(stages.values)
p_at_k5
outcomes.update({5: p_at_k5})  

#gender
genders = [item for sublist in genders for item in sublist]
(len(genders) - sum(genders))/len(genders)
gender_outcomes.update({'five': (len(genders) - sum(genders))/len(genders)})

#ethinicity
diversity = [item for sublist in diversity for item in sublist]
temp_list = []
for race in race_outcomes.index:
  try:
    temp_list.append(diversity.count(race)/len(diversity))
  except:
    temp_list.append(0)
race_outcomes['five'] = temp_list

#################### 6 count embeddings and one hot ####################
stages = []
diversity = []
genders = []
for one_job_id in jobIDs:
  pos_embedding = GenerateCountEmbedding(one_job_id, job_text, resume_text)
  pos_embedding['ReqID'] = np.repeat(one_job_id,len(pos_embedding))
  pos_embedding_with_hot = pd.DataFrame(pos_embedding).merge(all_dummies, how="left", on=["ID",'ReqID'])
  pos_embedding_with_hot = RecommendTopX(jobID=one_job_id, full_df=pos_embedding_with_hot.drop('ReqID',axis=1), num_x=K)
  pos_embedding_with_hot = pos_embedding_with_hot.merge(resume_text[resume_text['Req ID'] == one_job_id], on='Candidate ID',how='left')
  
  stages.append(pos_embedding_with_hot['Latest Recruiting Step'].values)
  div_rows = (diversity_df.ReqID == one_job_id) & (diversity_df.CanID.isin(pos_embedding_with_hot['Candidate ID']))
  diversity.append(diversity_df.Ethinicity[div_rows].values)
  genders.append(diversity_df.IsMale[div_rows].values)

stages = [item for sublist in stages for item in sublist]
stages = pd.DataFrame(stages, columns=['Stages'])['Stages'].value_counts()
good_picks = stages[stages.index.isin(jobs_reviewed_atleast_once)]
p_at_k6 = sum(good_picks)/sum(stages.values)
p_at_k6 
outcomes.update({6: p_at_k6})

#gender
genders = [item for sublist in genders for item in sublist]
(len(genders) - sum(genders))/len(genders)
gender_outcomes.update({'six': (len(genders) - sum(genders))/len(genders)})

#ethinicity
diversity = [item for sublist in diversity for item in sublist]
temp_list = []
for race in race_outcomes.index:
  try:
    temp_list.append(diversity.count(race)/len(diversity))
  except:
    temp_list.append(0)
race_outcomes['six'] = temp_list

#################### 7 td-ifd embeddings and one hot ####################
stages = []
diversity = []
genders = []
for one_job_id in jobIDs:
  pos_tfidf = GenerateTfidfEmbedding(one_job_id, job_text, resume_text)
  pos_tfidf['ReqID'] = np.repeat(one_job_id,len(pos_tfidf))
  pos_tfidf_with_hot = pd.DataFrame(pos_tfidf).merge(all_dummies, how="left", on=["ID",'ReqID'])
  pos_tfidf_with_hot = RecommendTopX(jobID=one_job_id, full_df=pos_tfidf_with_hot.drop('ReqID',axis=1), num_x=K)
  pos_tfidf_with_hot = pos_tfidf_with_hot.merge(resume_text[resume_text['Req ID'] == one_job_id], on='Candidate ID',how='left')
  
  stages.append(pos_tfidf_with_hot['Latest Recruiting Step'].values)
  div_rows = (diversity_df.ReqID == one_job_id) & (diversity_df.CanID.isin(pos_tfidf_with_hot['Candidate ID']))
  diversity.append(diversity_df.Ethinicity[div_rows].values)
  genders.append(diversity_df.IsMale[div_rows].values)

stages = [item for sublist in stages for item in sublist]
stages = pd.DataFrame(stages, columns=['Stages'])['Stages'].value_counts()
good_picks = stages[stages.index.isin(jobs_reviewed_atleast_once)]
p_at_k7 = sum(good_picks)/sum(stages.values)
p_at_k7 
outcomes.update({7: p_at_k7})

#gender
genders = [item for sublist in genders for item in sublist]
(len(genders) - sum(genders))/len(genders)
gender_outcomes.update({'seven': (len(genders) - sum(genders))/len(genders)})

#ethinicity
diversity = [item for sublist in diversity for item in sublist]
temp_list = []
for race in race_outcomes.index:
  try:
    temp_list.append(diversity.count(race)/len(diversity))
  except:
    temp_list.append(0)
race_outcomes['seven'] = temp_list

#################### 8 doc2vec embeddings and one hot ####################
stages = []
diversity = []
genders = []
for one_job_id in jobIDs:
  pos_tfidf = Generate_Doc2Vec_Embeddings(one_job_id, job_text, resume_text, vectorSize=doc2_vec_size)
  pos_tfidf['ReqID'] = np.repeat(one_job_id,len(pos_tfidf))
  pos_tfidf_with_hot = pd.DataFrame(pos_tfidf).merge(all_dummies, how="left", on=["ID",'ReqID'])
  pos_tfidf_with_hot = RecommendTopX(jobID=one_job_id, full_df=pos_tfidf_with_hot.drop('ReqID',axis=1), num_x=K)
  pos_tfidf_with_hot = pos_tfidf_with_hot.merge(resume_text[resume_text['Req ID'] == one_job_id], on='Candidate ID',how='left')
  
  stages.append(pos_tfidf_with_hot['Latest Recruiting Step'].values)
  div_rows = (diversity_df.ReqID == one_job_id) & (diversity_df.CanID.isin(pos_tfidf_with_hot['Candidate ID']))
  diversity.append(diversity_df.Ethinicity[div_rows].values)
  genders.append(diversity_df.IsMale[div_rows].values)

stages = [item for sublist in stages for item in sublist]
stages = pd.DataFrame(stages, columns=['Stages'])['Stages'].value_counts()
good_picks = stages[stages.index.isin(jobs_reviewed_atleast_once)]
p_at_k8 = sum(good_picks)/sum(stages.values)
p_at_k8 
outcomes.update({8: p_at_k8})

#gender
genders = [item for sublist in genders for item in sublist]
(len(genders) - sum(genders))/len(genders)
gender_outcomes.update({'eight': (len(genders) - sum(genders))/len(genders)})

#ethinicity
diversity = [item for sublist in diversity for item in sublist]
temp_list = []
for race in race_outcomes.index:
  try:
    temp_list.append(diversity.count(race)/len(diversity))
  except:
    temp_list.append(0)
race_outcomes['eight'] = temp_list

#################### 9 count embeddings and one hot with our alternations ####################
stages = []
diversity = []
genders = []
for one_job_id in jobIDs:
  pos_embedding = GenerateCountEmbedding(one_job_id, job_text, resume_text)
  pos_embedding['ReqID'] = np.repeat(one_job_id,len(pos_embedding))
  pos_embedding_with_hot_ideal = pd.DataFrame(pos_embedding).merge(all_dummies_ideal, how="left", on=["ID",'ReqID'])
  pos_embedding_with_hot_ideal = RecommendTopX(jobID=one_job_id, full_df=pos_embedding_with_hot_ideal.drop('ReqID',axis=1), num_x=K)
  pos_embedding_with_hot_ideal = pos_embedding_with_hot_ideal.merge(resume_text[resume_text['Req ID'] == one_job_id], on='Candidate ID',how='left')
  
  stages.append(pos_embedding_with_hot_ideal['Latest Recruiting Step'].values)
  div_rows = (diversity_df.ReqID == one_job_id) & (diversity_df.CanID.isin(pos_embedding_with_hot_ideal['Candidate ID']))
  diversity.append(diversity_df.Ethinicity[div_rows].values)
  genders.append(diversity_df.IsMale[div_rows].values)

stages = [item for sublist in stages for item in sublist]
stages = pd.DataFrame(stages, columns=['Stages'])['Stages'].value_counts()
good_picks = stages[stages.index.isin(jobs_reviewed_atleast_once)]
p_at_k9 = sum(good_picks)/sum(stages.values)
p_at_k9 
outcomes.update({9: p_at_k9})

#gender
genders = [item for sublist in genders for item in sublist]
(len(genders) - sum(genders))/len(genders)
gender_outcomes.update({'nine': (len(genders) - sum(genders))/len(genders)})

#ethinicity
diversity = [item for sublist in diversity for item in sublist]
temp_list = []
for race in race_outcomes.index:
  try:
    temp_list.append(diversity.count(race)/len(diversity))
  except:
    temp_list.append(0)
race_outcomes['nine'] = temp_list

#################### 10 td-ifd embeddings and one hot with our alternations ####################
stages = []
diversity = []
genders = []
for one_job_id in jobIDs:
  pos_tfidf = GenerateTfidfEmbedding(one_job_id, job_text, resume_text)
  pos_tfidf['ReqID'] = np.repeat(one_job_id,len(pos_tfidf))
  pos_tfidf_with_hot_ideal = pd.DataFrame(pos_tfidf).merge(all_dummies_ideal, how="left", on=["ID",'ReqID'])
  pos_tfidf_with_hot_ideal = RecommendTopX(jobID=one_job_id, full_df=pos_tfidf_with_hot_ideal.drop('ReqID',axis=1), num_x=K)
  pos_tfidf_with_hot_ideal = pos_tfidf_with_hot_ideal.merge(resume_text[resume_text['Req ID'] == one_job_id], on='Candidate ID',how='left')
  
  stages.append(pos_tfidf_with_hot_ideal['Latest Recruiting Step'].values)
  div_rows = (diversity_df.ReqID == one_job_id) & (diversity_df.CanID.isin(pos_tfidf_with_hot_ideal['Candidate ID']))
  diversity.append(diversity_df.Ethinicity[div_rows].values)
  genders.append(diversity_df.IsMale[div_rows].values)

stages = [item for sublist in stages for item in sublist]
stages = pd.DataFrame(stages, columns=['Stages'])['Stages'].value_counts()
good_picks = stages[stages.index.isin(jobs_reviewed_atleast_once)]
p_at_k10 = sum(good_picks)/sum(stages.values)
p_at_k10 
outcomes.update({10: p_at_k10})

#gender
genders = [item for sublist in genders for item in sublist]
(len(genders) - sum(genders))/len(genders)
gender_outcomes.update({'ten': (len(genders) - sum(genders))/len(genders)})

#ethinicity
diversity = [item for sublist in diversity for item in sublist]
temp_list = []
for race in race_outcomes.index:
  try:
    temp_list.append(diversity.count(race)/len(diversity))
  except:
    temp_list.append(0)
race_outcomes['ten'] = temp_list

#################### 11 doc2vec embeddings and one hot with our alternations ####################
stages = []
diversity = []
genders = []
for one_job_id in jobIDs:
  pos_tfidf = Generate_Doc2Vec_Embeddings(one_job_id, job_text, resume_text, vectorSize=doc2_vec_size)
  pos_tfidf['ReqID'] = np.repeat(one_job_id,len(pos_tfidf))
  pos_tfidf_with_hot_ideal = pd.DataFrame(pos_tfidf).merge(all_dummies_ideal, how="left", on=["ID",'ReqID'])
  pos_tfidf_with_hot_ideal = RecommendTopX(jobID=one_job_id, full_df=pos_tfidf_with_hot_ideal.drop('ReqID',axis=1), num_x=K)
  pos_tfidf_with_hot_ideal = pos_tfidf_with_hot_ideal.merge(resume_text[resume_text['Req ID'] == one_job_id], on='Candidate ID',how='left')
  
  stages.append(pos_tfidf_with_hot_ideal['Latest Recruiting Step'].values)
  div_rows = (diversity_df.ReqID == one_job_id) & (diversity_df.CanID.isin(pos_tfidf_with_hot_ideal['Candidate ID']))
  diversity.append(diversity_df.Ethinicity[div_rows].values)
  genders.append(diversity_df.IsMale[div_rows].values)

stages = [item for sublist in stages for item in sublist]
stages = pd.DataFrame(stages, columns=['Stages'])['Stages'].value_counts()
good_picks = stages[stages.index.isin(jobs_reviewed_atleast_once)]
p_at_k11 = sum(good_picks)/sum(stages.values)
p_at_k11 
outcomes.update({11: p_at_k11})

#gender
genders = [item for sublist in genders for item in sublist]
(len(genders) - sum(genders))/len(genders)
gender_outcomes.update({'eleven': (len(genders) - sum(genders))/len(genders)})

#ethinicity
diversity = [item for sublist in diversity for item in sublist]
temp_list = []
for race in race_outcomes.index:
  try:
    temp_list.append(diversity.count(race)/len(diversity))
  except:
    temp_list.append(0)
race_outcomes['eleven'] = temp_list

#################### results #################### 
outcomes
sorted(outcomes.items(), key=lambda item: item[1])

gender_summary = gender_outcomes
gender_summary.update({'OurAvg': np.mean(pd.Series(gender_summary)[2:])})
gender_summary= pd.Series(gender_summary)[['PctWomenApplied', 'PctWomenInterestedIn','OurAvg']]
gender_summary

race_summary = race_outcomes
race_summary['OurAvg'] = np.mean(race_summary[race_summary.columns[2:]].T).values
race_summary = race_summary[['PctApplied', 'PctInterestedIn','OurAvg']]*100
khc_var = sum((race_summary.PctApplied - race_summary.PctInterestedIn)*(race_summary.PctApplied - race_summary.PctInterestedIn))
our_var = sum((race_summary.PctApplied - race_summary.OurAvg)*(race_summary.PctApplied - race_summary.OurAvg))
race_summary
pd.DataFrame([khc_var,our_var], columns=['Variance of Ethinicity - apply vs interest'], index=['KHC','Ours'])































#################### playing around #################### 

#################### PCA(count) + one hot ideal #################### 
### PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

stages = []
diversity = []
genders = []
for one_job_id in jobIDs:
  pos_embedding = GenerateCountEmbedding(one_job_id, job_text, resume_text)
  
  #PCA on those
  features = pos_embedding[pos_embedding.columns[1:]]
  features = StandardScaler().fit_transform(features)
  pca = PCA(n_components=min(len(pos_embedding),100))
  features = pca.fit_transform(features)
  temp = pd.DataFrame(pos_embedding.ID)
  pos_embedding = pd.concat([temp, pd.DataFrame(features)],axis=1)
  
  pos_embedding['ReqID'] = np.repeat(one_job_id,len(pos_embedding))
  pos_embedding_with_hot_ideal = pd.DataFrame(pos_embedding).merge(all_dummies_ideal, how="left", on=["ID",'ReqID'])
  pos_embedding_with_hot_ideal = RecommendTopX(jobID=one_job_id, full_df=pos_embedding_with_hot_ideal.drop('ReqID',axis=1), num_x=K)
  pos_embedding_with_hot_ideal = pos_embedding_with_hot_ideal.merge(resume_text[resume_text['Req ID'] == one_job_id], on='Candidate ID',how='left')
  
  stages.append(pos_embedding_with_hot_ideal['Latest Recruiting Step'].values)
  div_rows = (diversity_df.ReqID == one_job_id) & (diversity_df.CanID.isin(pos_embedding_with_hot_ideal['Candidate ID']))
  diversity.append(diversity_df.Ethinicity[div_rows].values)
  genders.append(diversity_df.IsMale[div_rows].values)

stages = [item for sublist in stages for item in sublist]
stages = pd.DataFrame(stages, columns=['Stages'])['Stages'].value_counts()
good_picks = stages[stages.index.isin(jobs_reviewed_atleast_once)]
p_at_k_pca = sum(good_picks)/sum(stages.values)
p_at_k_pca 
outcomes.update({'pca': p_at_k_pca})

#gender
genders = [item for sublist in genders for item in sublist]
(len(genders) - sum(genders))/len(genders)
gender_outcomes.update({'pca': (len(genders) - sum(genders))/len(genders)})

#ethinicity
diversity = [item for sublist in diversity for item in sublist]
temp_list = []
for race in race_outcomes.index:
  try:
    temp_list.append(diversity.count(race)/len(diversity))
  except:
    temp_list.append(0)
race_outcomes['pca'] = temp_list

#################### has degree to top  ####################
stages = []
diversity = []
genders = []
for one_job_id in jobIDs:

  pos_df = Generate_Doc2Vec_Embeddings(one_job_id, job_text, resume_text)
  pos_df['ReqID'] = np.repeat(one_job_id,len(pos_df))
  pos_df = pd.DataFrame(pos_df).merge(all_dummies_ideal, how="left", on=["ID",'ReqID'])
  
  #if they need a doctor/master/bachelor degree:
  has_degree = []
  if pos_df.Doctors[pos_df.ID == one_job_id].values == 1:
    for item in range(0,len(pos_df)):
      if pos_df.Doctors[item] == 1:
        has_degree.append(item)
    has_degree_df = pos_df[pos_df.index.isin(has_degree)]
    has_degree = has_degree[1:]
    no_degree_df = pos_df[~pos_df.index.isin(has_degree)]
    has_degree_df = RecommendTopX(jobID=one_job_id, full_df=has_degree_df.drop('ReqID',axis=1), num_x=min(K, len(has_degree)))
    no_degree_df = RecommendTopX(jobID=one_job_id, full_df=no_degree_df.drop('ReqID',axis=1), num_x=min(K, len(no_degree_df)))
    
    job_dummy_ideal_FULL = pd.DataFrame(np.concatenate([has_degree_df,no_degree_df]))
    job_dummy_ideal_FULL.columns = ['Candidate ID', 'cosine']
    job_dummy_ideal_FULL = job_dummy_ideal_FULL.head(K)
    
  elif pos_df.Masters[pos_df.ID == one_job_id].values == 1:
    for item in range(0,len(pos_df)):
      if pos_df.Masters[item] == 1:
        has_degree.append(item)
    has_degree_df = pos_df[pos_df.index.isin(has_degree)]
    has_degree = has_degree[1:]
    no_degree_df = pos_df[~pos_df.index.isin(has_degree)]
    has_degree_df = RecommendTopX(jobID=one_job_id, full_df=has_degree_df.drop('ReqID',axis=1), num_x=min(K, len(has_degree)))
    no_degree_df = RecommendTopX(jobID=one_job_id, full_df=no_degree_df.drop('ReqID',axis=1), num_x=min(K, len(no_degree_df)))
    
    job_dummy_ideal_FULL = pd.DataFrame(np.concatenate([has_degree_df,no_degree_df]))
    job_dummy_ideal_FULL.columns = ['Candidate ID', 'cosine']
    job_dummy_ideal_FULL = job_dummy_ideal_FULL.head(K)
    
  elif pos_df.Bachelors[pos_df.ID == one_job_id].values == 1:
    for item in range(0,len(pos_df)):
      if pos_df.Bachelors[item] == 1:
        has_degree.append(item)
    has_degree_df = pos_df[pos_df.index.isin(has_degree)]
    has_degree = has_degree[1:]
    no_degree_df = pos_df[~pos_df.index.isin(has_degree)]
    has_degree_df = RecommendTopX(jobID=one_job_id, full_df=has_degree_df.drop('ReqID',axis=1), num_x=min(K, len(has_degree)))
    no_degree_df = RecommendTopX(jobID=one_job_id, full_df=no_degree_df.drop('ReqID',axis=1), num_x=min(K, len(no_degree_df)))
    
    job_dummy_ideal_FULL = pd.DataFrame(np.concatenate([has_degree_df,no_degree_df]))
    job_dummy_ideal_FULL.columns = ['Candidate ID', 'cosine']
    job_dummy_ideal_FULL = job_dummy_ideal_FULL.head(K)
    
  else:
    job_dummy_ideal_FULL = RecommendTopX(jobID=one_job_id, full_df=pos_df.drop('ReqID',axis=1), num_x=K)
  
  job_dummy_ideal_FULL = job_dummy_ideal_FULL.merge(resume_text[resume_text['Req ID'] == one_job_id], on='Candidate ID',how='left')
  
  stages.append(job_dummy_ideal_FULL['Latest Recruiting Step'].values)  
  div_rows = (diversity_df.ReqID == one_job_id) & (diversity_df.CanID.isin(job_dummy_ideal_FULL['Candidate ID']))
  diversity.append(diversity_df.Ethinicity[div_rows].values)
  genders.append(diversity_df.IsMale[div_rows].values)

stages = [item for sublist in stages for item in sublist]
stages = pd.DataFrame(stages, columns=['Stages'])['Stages'].value_counts()
good_picks = stages[stages.index.isin(jobs_reviewed_atleast_once)]
p_at_k_degree = sum(good_picks)/sum(stages.values)
p_at_k_degree 
outcomes.update({'degree': p_at_k_degree})

#gender
genders = [item for sublist in genders for item in sublist]
(len(genders) - sum(genders))/len(genders)
gender_outcomes.update({'degree': (len(genders) - sum(genders))/len(genders)})

#ethinicity
diversity = [item for sublist in diversity for item in sublist]
temp_list = []
for race in race_outcomes.index:
  try:
    temp_list.append(diversity.count(race)/len(diversity))
  except:
    temp_list.append(0)
race_outcomes['degree'] = temp_list

#################### has years of experience to top ####################
stages = []

job_dummies_ideal_w_experi = pd.concat([job_dummies_ideal, job_features.mos_experience], axis=1)
resume_dummies_w_experi = pd.concat([resume_dummies, resume_features.mos_experience], axis=1)
all_dummies_ideal_w_experi = pd.concat([resume_dummies_w_experi, job_dummies_ideal_w_experi])

for one_job_id in jobIDs:

  pos_df = GenerateCountEmbedding(one_job_id, job_text, resume_text)
  pos_df['ReqID'] = np.repeat(one_job_id,len(pos_df))
  pos_df = pd.DataFrame(pos_df).merge(all_dummies_ideal_w_experi, how="left", on=["ID",'ReqID'])

  has_experi = []
  if pos_df.mos_experience[pos_df.ID == one_job_id].values > 0:
    experi_wanted = pos_df.mos_experience[pos_df.ID == one_job_id].values[0]
    
    for item in range(0,len(pos_df)):
      if pos_df.mos_experience[item] >= experi_wanted:
        has_experi.append(item)
      elif pos_df.mos_experience[item] == 0:
        has_experi.append(item)
      else:
        pass
    
    has_experi_wanted = pos_df[pos_df.index.isin(has_experi)]
    has_experi = has_experi[1:]
    no_experi_wanted = pos_df[~pos_df.index.isin(has_experi)]
    
    has_experi_wanted = RecommendTopX(jobID=one_job_id, full_df=has_experi_wanted.drop('ReqID',axis=1), num_x=min(K, len(has_degree)))
    no_experi_wanted = RecommendTopX(jobID=one_job_id, full_df=no_experi_wanted.drop('ReqID',axis=1), num_x=min(K, len(no_degree_df)))
    
    job_dummy_ideal_FULL = pd.DataFrame(np.concatenate([has_experi_wanted,no_experi_wanted]))
    job_dummy_ideal_FULL.columns = ['Candidate ID', 'cosine']
    job_dummy_ideal_FULL = job_dummy_ideal_FULL.head(K)
    
  else:
    job_dummy_ideal_FULL = RecommendTopX(jobID=one_job_id, full_df=pos_df.drop('ReqID',axis=1), num_x=K)
  
  job_dummy_ideal_FULL = job_dummy_ideal_FULL.merge(resume_text[resume_text['Req ID'] == one_job_id], on='Candidate ID',how='left')
  
  stages.append(job_dummy_ideal_FULL['Latest Recruiting Step'].values)  
  div_rows = (diversity_df.ReqID == one_job_id) & (diversity_df.CanID.isin(job_dummy_ideal_FULL['Candidate ID']))
  diversity.append(diversity_df.Ethinicity[div_rows].values)
  genders.append(diversity_df.IsMale[div_rows].values)

stages = [item for sublist in stages for item in sublist]
stages = pd.DataFrame(stages, columns=['Stages'])['Stages'].value_counts()
good_picks = stages[stages.index.isin(jobs_reviewed_atleast_once)]
p_at_k_experi = sum(good_picks)/sum(stages.values)
p_at_k_experi
outcomes.update({'experience': p_at_k_experi})

#################### results #################### 
outcomes
sorted(outcomes.items(), key=lambda item: item[1])

gender_summary = gender_outcomes
gender_summary.update({'OurAvg': np.mean(pd.Series(gender_summary)[2:])})
gender_summary= pd.Series(gender_summary)[['PctWomenApplied', 'PctWomenInterestedIn','OurAvg']]
gender_summary

race_summary = race_outcomes
race_summary['OurAvg'] = np.mean(race_summary[race_summary.columns[2:]].T).values
race_summary = race_summary[['PctApplied', 'PctInterestedIn','OurAvg']]*100
khc_var = sum((race_summary.PctApplied - race_summary.PctInterestedIn)*(race_summary.PctApplied - race_summary.PctInterestedIn))
our_var = sum((race_summary.PctApplied - race_summary.OurAvg)*(race_summary.PctApplied - race_summary.OurAvg))
race_summary
print(pd.DataFrame([khc_var,our_var], columns=['Variance of Ethinicity - apply vs interest'], index=['KHC','Ours']))

#################### save results #################### 
pd.Series(outcomes).to_csv('~/models/content_based/Results/p_at_k.csv')
pd.Series(gender_outcomes).to_csv('~/models/content_based/Results/gender_diversity.csv')
race_outcomes.to_csv('~/models/content_based/Results/race_diversity.csv')

#outcomes=pd.read_csv('~/models/content_based/Results/p_at_k.csv', names = ['model','p@k'])
#gender_outcomes = pd.read_csv('~/models/content_based/Results/gender_diversity.csv', names = ['model','percent'])
#race_outcomes=pd.read_csv('~/models/content_based/Results/race_diversity.csv',index_col=0)


race_summary = race_outcomes
#race_summary.drop('OurAvg',inplace=True,axis=1)
race_summary['khcVAR'] = abs(race_summary['PctInterestedIn']-race_summary['PctApplied'])
race_summary['oneVAR'] = abs(race_summary['one']-race_summary['PctApplied'])
race_summary['twoVAR'] = abs(race_summary['two']-race_summary['PctApplied'])
race_summary['threeVAR'] = abs(race_summary['three']-race_summary['PctApplied'])
race_summary['fourVAR'] = abs(race_summary['four']-race_summary['PctApplied'])
race_summary['fiveVAR'] = abs(race_summary['five']-race_summary['PctApplied'])
race_summary['sixVAR'] = abs(race_summary['six']-race_summary['PctApplied'])
race_summary['sevenVAR'] = abs(race_summary['seven']-race_summary['PctApplied'])
race_summary['eightVAR'] = abs(race_summary['eight']-race_summary['PctApplied'])
race_summary['nineVAR'] = abs(race_summary['nine']-race_summary['PctApplied'])
race_summary['tenVAR'] = abs(race_summary['ten']-race_summary['PctApplied'])
race_summary['elevenVAR'] = abs(race_summary['eleven']-race_summary['PctApplied'])
race_summary['pcaVAR'] = abs(race_summary['pca']-race_summary['PctApplied'])
race_summary['degreeVAR'] = abs(race_summary['degree']-race_summary['PctApplied'])
race_summary=race_summary*100

race_summary[['khcVAR','degreeVAR','pcaVAR','oneVAR','twoVAR','threeVAR','fourVAR','fiveVAR','sixVAR','sevenVAR','eightVAR','nineVAR','tenVAR','elevenVAR']].apply(lambda x: sum(x**2))

x = race_summary[['PctApplied','PctInterestedIn','nine','khcVAR','nineVAR']]
x = round(x,2)
x.columns = ['Applied','KHC','Model','KHCvar','Modelvar']
x

khc_var = sum((race_summary.PctApplied - race_summary.PctInterestedIn)*(race_summary.PctApplied - race_summary.PctInterestedIn))
our_var = sum((race_summary.PctApplied - race_summary.nine)*(race_summary.PctApplied - race_summary.nine))

khc_var1 = sum(abs(race_summary.PctApplied - race_summary.PctInterestedIn))
our_var1 = sum(abs(race_summary.PctApplied - race_summary.nine))

race_summary
print(pd.DataFrame([khc_var,our_var], columns=['Variance of Ethinicity - apply vs interest'], index=['KHC','Ours']))
print(pd.DataFrame([khc_var1,our_var1], columns=['Variance of Ethinicity - apply vs interest'], index=['KHC','Ours']))

