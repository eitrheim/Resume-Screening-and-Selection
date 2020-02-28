import pandas as pd
import numpy as np
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer

##### Read data for 5 sample positions ###########

req_ids = ["e3625ad", "39ee3f", "45de815"
           ,"40a2c38", "63146c6"]

# read raw text data for embeddings
job_text = pd.read_csv("data/cleaned_job.csv", index_col=0)
resume_text = pd.read_csv("data/cleaned_resume.csv", index_col=0)

# keep only the relevant positions and candidates

job_text = job_text[job_text["Req ID"].isin(req_ids)]
resume_text = resume_text[resume_text["Req ID"].isin(req_ids)]


##### text embedding(Count)#########
### a function to repeat the process
def GenerateCountEmbedding(req_id, job_text_df, resume_text_df):
  pos_jd_text = job_text[job_text["Req ID"]==req_id]
  pos_resume_text = resume_text[resume_text["Req ID"]==req_id]

  pos_jd_text.rename(columns = {'Req ID':'ID',
                             'Job Description':'text'}, inplace=True)
  pos_jd_text.ID = req_id
  pos_jd_text = pos_jd_text[['ID', 'text']]

  pos_resume_text.rename(columns = {'Candidate ID':'ID',
                             'Resume Text':'text'}, inplace=True)
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

### for position e3625ad 
pos1_embedding = GenerateCountEmbedding("e3625ad", job_text
                                        , resume_text)

### for position "39ee3f"
pos2_embedding = GenerateCountEmbedding("39ee3f", job_text
                                        , resume_text)

### for position "45de815"
pos3_embedding = GenerateCountEmbedding("45de815", job_text
                                        , resume_text)

### for position "40a2c38"
pos4_embedding = GenerateCountEmbedding("40a2c38", job_text
                                        , resume_text)

### for position "63146c6"
pos5_embedding = GenerateCountEmbedding("63146c6", job_text
                                        , resume_text)

##### embeddings TFIDF #####
def GenerateTfidfEmbedding(req_id, job_text_df, resume_text_df):
  pos_jd_text = job_text[job_text["Req ID"]==req_id]
  pos_resume_text = resume_text[resume_text["Req ID"]==req_id]

  pos_jd_text.rename(columns = {'Req ID':'ID',
                             'Job Description':'text'}, inplace=True)
  pos_jd_text.ID = req_id
  pos_jd_text = pos_jd_text[['ID', 'text']]

  pos_resume_text.rename(columns = {'Candidate ID':'ID',
                             'Resume Text':'text'}, inplace=True)
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

### for position "e3625ad" 
pos1_tfidf = GenerateTfidfEmbedding("e3625ad", job_text
                                        , resume_text)

### for position "39ee3f"
pos2_tfidf = GenerateTfidfEmbedding("39ee3f", job_text
                                        , resume_text)

### for position "45de815"
pos3_tfidf = GenerateTfidfEmbedding("45de815", job_text
                                        , resume_text)

### for position "40a2c38"
pos4_tfidf = GenerateTfidfEmbedding("40a2c38", job_text
                                        , resume_text)

### for position "63146c6"
pos5_tfidf = GenerateTfidfEmbedding("63146c6", job_text
                                        , resume_text)


##### Run Cos Sim and rank the candidates #####

# define function for returning recommended resume ID's based on Job Description

def RecommendTopCandidates(jobID, full_df, num_candidates):
  
  can_count = len(full_df) - 1
  
  if num_candidates > can_count:
    raise ValueError("Number of recommendations exceeds number of candidates. The number of candidates for this position is :{}".format(can_count))
  
  recommended_candidates = []
  
  full_df.reset_index(inplace=True)
  
  unique_ids = pd.Series(full_df["ID"])
  
  cos_sim_matrix = cosine_similarity(full_df.drop("ID", axis=1)
                                    , full_df.drop("ID", axis=1))
  
  ideal_candidate_index = unique_ids[unique_ids == jobID].index[0]
  
  sorted_scores = pd.Series(cos_sim_matrix[ideal_candidate_index]).sort_values(ascending=False)
  
  top_candidate_index = list(sorted_scores.iloc[1:(num_candidates+1)].index)
  
  for i in top_candidate_index:
    recommended_candidates.append(list(unique_ids)[i])
    
  return recommended_candidates


#### position 1 with count vectors #####

ordered_candidate_list_pos1 = RecommendTopCandidates(jobID='e3625ad'
                                                    , full_df=pos1_embedding
                                                    , num_candidates=10)

df_list = []
for i in ordered_candidate_list_pos1:
  df_list.append(resume_text.loc[resume_text['Candidate ID'] == i])

# show result
job_text.loc[job_text["Req ID"] == "e3625ad"]
pd.concat(df_list)

#### position 1 with tfidf vectors #####

ordered_candidate_list_pos1 = RecommendTopCandidates(jobID='e3625ad'
                                                    , full_df=pos1_tfidf
                                                    , num_candidates=10)

df_list = []
for i in ordered_candidate_list_pos1:
  df_list.append(resume_text.loc[resume_text['Candidate ID'] == i])

# show result
job_text.loc[job_text["Req ID"] == "e3625ad"]
pd.concat(df_list)


#### position 2 with count vectors #####

ordered_candidate_list_pos2 = RecommendTopCandidates(jobID='39ee3f'
                                                    , full_df=pos2_embedding
                                                    , num_candidates=10)

df_list = []
for i in ordered_candidate_list_pos2:
  df_list.append(resume_text.loc[resume_text['Candidate ID'] == i])

# show result
job_text.loc[job_text["Req ID"] == "39ee3f"]
pd.concat(df_list)

#### position 2 with tfidf vectors #####

ordered_candidate_list_pos2 = RecommendTopCandidates(jobID='39ee3f'
                                                    , full_df=pos2_tfidf
                                                    , num_candidates=10)

df_list = []
for i in ordered_candidate_list_pos2:
  df_list.append(resume_text.loc[resume_text['Candidate ID'] == i])

# show result
job_text.loc[job_text["Req ID"] == "39ee3f"]
pd.concat(df_list)

#### position 3 with count vectors #####

ordered_candidate_list_pos3 = RecommendTopCandidates(jobID='45de815'
                                                    , full_df=pos3_embedding
                                                    , num_candidates=10)

df_list = []
for i in ordered_candidate_list_pos3:
  df_list.append(resume_text.loc[resume_text['Candidate ID'] == i])

# show result
job_text.loc[job_text["Req ID"] == "45de815"]
pd.concat(df_list)

#### position 3 with tfidf vectors #####

ordered_candidate_list_pos3 = RecommendTopCandidates(jobID='45de815'
                                                    , full_df=pos3_tfidf
                                                    , num_candidates=10)

df_list = []
for i in ordered_candidate_list_pos3:
  df_list.append(resume_text.loc[resume_text['Candidate ID'] == i])

# show result
job_text.loc[job_text["Req ID"] == "45de815"]
pd.concat(df_list)

#### position 4 with count vectors #####

ordered_candidate_list_pos4 = RecommendTopCandidates(jobID='40a2c38'
                                                    , full_df=pos4_embedding
                                                    , num_candidates=10)

df_list = []
for i in ordered_candidate_list_pos4:
  df_list.append(resume_text.loc[resume_text['Candidate ID'] == i])

# show result
job_text.loc[job_text["Req ID"] == "40a2c38"]
pd.concat(df_list)

#### position 4 with tfidf vectors #####

ordered_candidate_list_pos4 = RecommendTopCandidates(jobID='40a2c38'
                                                    , full_df=pos4_tfidf
                                                    , num_candidates=10)

df_list = []
for i in ordered_candidate_list_pos4:
  df_list.append(resume_text.loc[resume_text['Candidate ID'] == i])

# show result
job_text.loc[job_text["Req ID"] == "40a2c38"]
pd.concat(df_list)

#### position 5 with count vectors #####

ordered_candidate_list_pos5 = RecommendTopCandidates(jobID='63146c6'
                                                    , full_df=pos5_embedding
                                                    , num_candidates=10)

df_list = []
for i in ordered_candidate_list_pos5:
  df_list.append(resume_text.loc[resume_text['Candidate ID'] == i])

# show result
job_text.loc[job_text["Req ID"] == "63146c6"]
pd.concat(df_list)

#### position 5 with tfidf vectors #####

ordered_candidate_list_pos5 = RecommendTopCandidates(jobID='63146c6'
                                                    , full_df=pos5_tfidf
                                                    , num_candidates=10)

df_list = []
for i in ordered_candidate_list_pos5:
  df_list.append(resume_text.loc[resume_text['Candidate ID'] == i])

# show result
job_text.loc[job_text["Req ID"] == "63146c6"]
pd.concat(df_list)