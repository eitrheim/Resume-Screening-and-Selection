import pandas as pd
import numpy as np
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, auc_score

job_df = pd.read_csv("data/cleaned_job.csv", index_col=0)
resume_df = pd.read_csv("data/cleaned_resume.csv", index_col=0)

job_df.head()
resume_df.head()

resume_df[["Req ID", "Candidate ID"]].groupby(["Req ID"]).agg('count').sort_values('Candidate ID')

### try recommender on Req ID "e3625ad" ###
req_id = "e3625ad"
position_df = job_df.loc[job_df["Req ID"] == req_id]
applicant_df = resume_df.loc[resume_df["Req ID"] == req_id]

# bring them to the same format

position_df.rename(columns = {'Req ID':'ID',
                             'Job Description':'text'}, inplace=True)
position_df.ID = "0000"
position_df = position_df[['ID', 'text']]

applicant_df.rename(columns = {'Candidate ID':'ID',
                             'Resume Text':'text'}, inplace=True)
applicant_df = applicant_df[['ID', 'text']]

#append to same df
df = applicant_df.append(position_df)
df.set_index('ID', inplace=True)

# join words and vectorize
tokenizer = RegexpTokenizer(r'\w+')
df['text'] = df['text'].apply(lambda x: tokenizer.tokenize(x))
df['text'] = df['text'].apply(lambda x: ' '.join(x))

count = CountVectorizer()
count_df = count.fit_transform(df['text'])
count_df.shape

########################
model = LightFM()

#fullDF = pd.DataFrame(count_df)
train = count_df[:448]
test = count_df[449:]

model.fit(train, epochs=30, num_threads=2)


indices = pd.Series(df.index)

# define function for returning recommended resume ID's based on Job Description

def RecommendTopTen(model, data, ID = '0000'):
  
  recommended_candidates = []
  
  idx = indices[indices == ID].index[0]
  
  
  score_series = pd.Series(cos_sim[idx]).sort_values(ascending=False)
  
  top_10 = list(score_series.iloc[1:11].index)
  
  for i in top_10:
    recommended_candidates.append(list(df.index)[i])
    
  return recommended_candidates


# run 

#ordered_candidate_list = RecommendTopTen(model, count_df,'0000')

#df_list = []
#for i in ordered_candidate_list:
#  df_list.append(resume_df.loc[resume_df['Candidate ID'] == i])

# show result
#job_df.loc[job_df["Req ID"] == "e3625ad"]

#pd.concat(df_list)

# return top similarity score for validation

#def ReturnScore(ID = '0000', cos_sim = cos_sim_df):
  
 # idx = indices[indices == ID].index[0]
  
  #score_series = pd.Series(cos_sim[idx]).sort_values(ascending=False)
  
  #return score_series.iloc[1:11]

#ReturnScore()

train_precision = precision_at_k(model, train, k=10).mean()
#print(train_precision)
#test_precision = precision_at_k(model, test, k=10).mean()

train_auc = auc_score(model, train).mean()
#print(train_auc)
#test_auc = auc_score(model, test).mean()

print('Precision: Train %.2f.' % (train_precision))
print('Precision: Train %.2f.' % (train_auc))
#print('Precision: Test %.2f.' % (test_precision))
#print('Precision: Test %.2f.' % (test_auc))