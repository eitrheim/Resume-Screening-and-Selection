import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, auc_score
from lightfm.cross_validation import random_train_test_split
import scipy as sp

##### Read data for 5 sample positions ###########

req_ids = ["e3625ad", "39ee3f", "45de815"
           ,"40a2c38", "63146c6"]

# read raw text data for embeddings
job_text = pd.read_csv("data/cleaned_job.csv", index_col=0)

resume_text = pd.read_csv("data/cleaned_resume.csv", index_col=0)




# read structured
job_features = pd.read_csv("Resume-Parser-JOBS/data/output/job_description_summary.csv")
resume_features = pd.read_csv("data/resumes_5jobs.csv")

# keep only the relevant positions and candidates

job_text = job_text[job_text["Req ID"].isin(req_ids)]
resume_text = resume_text[resume_text["Req ID"].isin(req_ids)]
job_features = job_features[job_features.ReqID.isin(req_ids)]
resume_features = resume_features[resume_features.ReqID.isin(req_ids)]

##### one hot encode the structured features #####

### for jobs ###
# drop unused columns
drop_cols = ['GPA', 'courses', 'hobbies', 'email'
             , 'phone','Education', 'Extracurriculars'
             ,'Language', 'Work', 'Summaries', 'Skill'
             , 'Member', 'Writing', 'Researching'
             , 'Honor', 'Activity']

job_features.drop(drop_cols, inplace=True, axis=1)
df = job_features
hot = df[['ReqID']]

#honor_societies
df.honor_societies.fillna('', inplace=True)
hot['HonorSociety'] = df.honor_societies.apply(lambda x: 1 if len(x) > 2 else 0)

#latin_honors
df.latin_honors.fillna('', inplace=True)
hot['LatinHonors'] = df.latin_honors.apply(lambda x: 1 if len(x) > 2 else 0)

#scholarships_awards
df.scholarships_awards.fillna('', inplace=True)
hot['ScholarshipsAward'] = df.scholarships_awards.apply(lambda x: 1 if len(x) > 2 else 0)

#schools
df.community_college.fillna('', inplace=True)
hot['CommCollege'] = df.community_college.apply(lambda x: 1 if len(x) > 2 else 0)
df.other_universities.fillna('', inplace=True)
hot['OtherUni'] = df.other_universities.apply(lambda x: 1 if len(x) > 2 else 0)
df.top_100_universities.fillna('', inplace=True)
hot['Top100Uni'] = df.top_100_universities.apply(lambda x: 1 if len(x) > 2 else 0)
df.top_10_universities.fillna('', inplace=True)
hot['Top10Uni'] = df.top_10_universities.apply(lambda x: 1 if len(x) > 2 else 0)

#degrees
df.associate_education_level.fillna('', inplace=True)
hot['Associates'] = df.associate_education_level.apply(lambda x: 1 if len(x) > 2 else 0)
df.bachelor_education_level.fillna('', inplace=True)
hot['Bachelors'] = df.bachelor_education_level.apply(lambda x: 1 if len(x) > 2 else 0)
df.master_education_level.fillna('', inplace=True)
hot['Masters'] = df.master_education_level.apply(lambda x: 1 if len(x) > 2 else 0)
df.doctor_education_level.fillna('', inplace=True)
hot['Doctors'] = df.doctor_education_level.apply(lambda x: 1 if len(x) > 2 else 0)

#companies
df.company_foodbev.fillna('', inplace=True)
hot['FoodBev'] = df.company_foodbev.apply(lambda x: 1 if len(x) > 2 else 0)
df.company_consumer.fillna('', inplace=True)
hot['Consumer'] = df.company_consumer.apply(lambda x: 1 if len(x) > 2 else 0)
df.company_energychem.fillna('', inplace=True)
hot['EnergyChem'] = df.company_energychem.apply(lambda x: 1 if len(x) > 2 else 0)
df.company_fin.fillna('', inplace=True)
hot['Fin'] = df.company_fin.apply(lambda x: 1 if len(x) > 2 else 0)
df.company_health.fillna('', inplace=True)
hot['HealthMed'] = df.company_health.apply(lambda x: 1 if len(x) > 2 else 0)
df.company_industrial.fillna('', inplace=True)
hot['Industrial'] = df.company_industrial.apply(lambda x: 1 if len(x) > 2 else 0)
df.company_tech.fillna('', inplace=True)
hot['Tech'] = df.company_tech.apply(lambda x: 1 if len(x) > 2 else 0)
df.company_services.fillna('', inplace=True)
hot['Services'] = df.company_services.apply(lambda x: 1 if len(x) > 2 else 0)
df.company_other.fillna('', inplace=True)
hot['OtherCo'] = df.company_other.apply(lambda x: 1 if len(x) > 2 else 0)

# ONE HOT ENCODING - EXPLODING COLUMNS
import yaml
with open('Resume-Parser-master-new/confs/config.yaml', 'r') as stream:
  yaml_file = yaml.safe_load(stream)

#certifications
df.certifications.fillna('', inplace=True)
for item in yaml_file['case_agnostic_whole_resume']['certifications']:
  if type(item) == list:
    search_term = item[0].replace('\\x20','').replace(' ','')
    col_name = item[1].replace('\\x20','').replace(' ','')
  else:
    search_term = item.replace('\\x20','').replace(' ','')
    col_name = search_term
  hot[col_name] = df.certifications.apply(lambda x: 1 if x.find(search_term) >= 0 else 0)

#soft_skills
df.soft_skills.fillna('', inplace=True)
for item in yaml_file['case_agnostic_whole_resume']['soft_skills']:
  if type(item) == list:
    search_term = item[0].replace('\\x20','').replace(' ','')
    col_name = item[1].replace('\\x20','').replace(' ','')
  else:
    search_term = item.replace('\\x20','').replace(' ','')
    col_name = search_term
  hot[col_name] = df.soft_skills.apply(lambda x: 1 if x.find(search_term) >= 0 else 0)

#major_minor
df.major_minor.fillna('', inplace=True)
for item in yaml_file['case_agnostic_education']['major_minor']:
  if type(item) == list:
    search_term = item[0].replace('\\x20','').replace(' ','')
    col_name = item[1].replace('\\x20','').replace(' ','')
  else:
    search_term = item.replace('\\x20','').replace(' ','')
    col_name = search_term
  hot[col_name] = df.major_minor.apply(lambda x: 1 if x.find(search_term) >= 0 else 0)

#languages
df.languages.fillna('', inplace=True)
for item in yaml_file['case_agnostic_languages']['languages']:
  if type(item) == list:
    search_term = item[0].replace('\\x20','').replace(' ','')
    col_name = item[1].replace('\\x20','').replace(' ','')
  else:
    search_term = item.replace('\\x20','').replace(' ','')
    col_name = search_term
  hot[col_name] = df.languages.apply(lambda x: 1 if x.find(search_term) >= 0 else 0)

#technical_skills
df.technical_skills.fillna('', inplace=True)
for item in yaml_file['case_agnostic_skill']['technical_skills']:
  if type(item) == list:
    search_term = item[0].replace('\\x20','').replace(' ','')
    col_name = item[1].replace('\\x20','').replace(' ','')
  else:
    search_term = item.replace('\\x20','').replace(' ','')
    col_name = search_term
  hot[col_name] = df.technical_skills.apply(lambda x: 1 if x.find(search_term) >= 0 else 0)

job_dummies = hot

### for resumes ###

# drop unused columns
drop_cols = ['GPA', 'courses', 'hobbies', 'email', 'phone'
             ,'Education', 'Extracurriculars','Language', 'Work'
             , 'Summaries', 'Skill', 'Member', 'Writing', 'Researching'
             , 'Honor', 'Activity']

resume_features.drop(drop_cols, inplace=True, axis=1)
df = resume_features

#ONE HOT ENCODING
hot = df[['ReqID', 'CanID']]

#honor_societies
df.honor_societies.fillna('', inplace=True)
hot['HonorSociety'] = df.honor_societies.apply(lambda x: 1 if len(x) > 2 else 0)

#latin_honors
df.latin_honors.fillna('', inplace=True)
hot['LatinHonors'] = df.latin_honors.apply(lambda x: 1 if len(x) > 2 else 0)

#scholarships_awards
df.scholarships_awards.fillna('', inplace=True)
hot['ScholarshipsAward'] = df.scholarships_awards.apply(lambda x: 1 if len(x) > 2 else 0)

#schools
df.community_college.fillna('', inplace=True)
hot['CommCollege'] = df.community_college.apply(lambda x: 1 if len(x) > 2 else 0)
df.other_universities.fillna('', inplace=True)
hot['OtherUni'] = df.other_universities.apply(lambda x: 1 if len(x) > 2 else 0)
df.top_100_universities.fillna('', inplace=True)
hot['Top100Uni'] = df.top_100_universities.apply(lambda x: 1 if len(x) > 2 else 0)
df.top_10_universities.fillna('', inplace=True)
hot['Top10Uni'] = df.top_10_universities.apply(lambda x: 1 if len(x) > 2 else 0)

#degrees
df.associate_education_level.fillna('', inplace=True)
hot['Associates'] = df.associate_education_level.apply(lambda x: 1 if len(x) > 2 else 0)
df.bachelor_education_level.fillna('', inplace=True)
hot['Bachelors'] = df.bachelor_education_level.apply(lambda x: 1 if len(x) > 2 else 0)
df.master_education_level.fillna('', inplace=True)
hot['Masters'] = df.master_education_level.apply(lambda x: 1 if len(x) > 2 else 0)
df.doctor_education_level.fillna('', inplace=True)
hot['Doctors'] = df.doctor_education_level.apply(lambda x: 1 if len(x) > 2 else 0)

#companies
df.company_foodbev.fillna('', inplace=True)
hot['FoodBev'] = df.company_foodbev.apply(lambda x: 1 if len(x) > 2 else 0)
df.company_consumer.fillna('', inplace=True)
hot['Consumer'] = df.company_consumer.apply(lambda x: 1 if len(x) > 2 else 0)
df.company_energychem.fillna('', inplace=True)
hot['EnergyChem'] = df.company_energychem.apply(lambda x: 1 if len(x) > 2 else 0)
df.company_fin.fillna('', inplace=True)
hot['Fin'] = df.company_fin.apply(lambda x: 1 if len(x) > 2 else 0)
df.company_health.fillna('', inplace=True)
hot['HealthMed'] = df.company_health.apply(lambda x: 1 if len(x) > 2 else 0)
df.company_industrial.fillna('', inplace=True)
hot['Industrial'] = df.company_industrial.apply(lambda x: 1 if len(x) > 2 else 0)
df.company_tech.fillna('', inplace=True)
hot['Tech'] = df.company_tech.apply(lambda x: 1 if len(x) > 2 else 0)
df.company_services.fillna('', inplace=True)
hot['Services'] = df.company_services.apply(lambda x: 1 if len(x) > 2 else 0)
df.company_other.fillna('', inplace=True)
hot['OtherCo'] = df.company_other.apply(lambda x: 1 if len(x) > 2 else 0)

#ONE HOT ENCODING - EXPLODING COLUMNS
with open('Resume-Parser-master-new/confs/config.yaml', 'r') as stream:
  yaml_file = yaml.safe_load(stream)

#certifications
df.certifications.fillna('', inplace=True)
for item in yaml_file['case_agnostic_whole_resume']['certifications']:
  if type(item) == list:
    search_term = item[0].replace('\\x20','').replace(' ','')
    col_name = item[1].replace('\\x20','').replace(' ','')
  else:
    search_term = item.replace('\\x20','').replace(' ','')
    col_name = search_term
  hot[col_name] = df.certifications.apply(lambda x: 1 if x.find(search_term) >= 0 else 0)

#soft_skills
df.soft_skills.fillna('', inplace=True)
for item in yaml_file['case_agnostic_whole_resume']['soft_skills']:
  if type(item) == list:
    search_term = item[0].replace('\\x20','').replace(' ','')
    col_name = item[1].replace('\\x20','').replace(' ','')
  else:
    search_term = item.replace('\\x20','').replace(' ','')
    col_name = search_term
  hot[col_name] = df.soft_skills.apply(lambda x: 1 if x.find(search_term) >= 0 else 0)

#major_minor
df.major_minor.fillna('', inplace=True)
for item in yaml_file['case_agnostic_education']['major_minor']:
  if type(item) == list:
    search_term = item[0].replace('\\x20','').replace(' ','')
    col_name = item[1].replace('\\x20','').replace(' ','')
  else:
    search_term = item.replace('\\x20','').replace(' ','')
    col_name = search_term
  hot[col_name] = df.major_minor.apply(lambda x: 1 if x.find(search_term) >= 0 else 0)

#languages
df.languages.fillna('', inplace=True)
for item in yaml_file['case_agnostic_languages']['languages']:
  if type(item) == list:
    search_term = item[0].replace('\\x20','').replace(' ','')
    col_name = item[1].replace('\\x20','').replace(' ','')
  else:
    search_term = item.replace('\\x20','').replace(' ','')
    col_name = search_term
  hot[col_name] = df.languages.apply(lambda x: 1 if x.find(search_term) >= 0 else 0)

#technical_skills
df.technical_skills.fillna('', inplace=True)
for item in yaml_file['case_agnostic_skill']['technical_skills']:
  if type(item) == list:
    search_term = item[0].replace('\\x20','').replace(' ','')
    col_name = item[1].replace('\\x20','').replace(' ','')
  else:
    search_term = item.replace('\\x20','').replace(' ','')
    col_name = search_term
  hot[col_name] = df.technical_skills.apply(lambda x: 1 if x.find(search_term) >= 0 else 0)

empty_cols = []
for i in hot.columns[2:]:
  if sum(hot[i]) == 0:
    empty_cols.append(i)

resume_dummies = hot

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


##### combining embedding with dummies ########
# list(set(list(resume_dummies.columns))-set(list(job_dummies.columns)))

#rename their index column
resume_dummies.rename(columns = {'CanID':'ID'}, inplace=True)
resume_dummies.drop(["ReqID"], inplace=True, axis=1)
job_dummies.rename(columns = {'ReqID':'ID'}, inplace=True)
all_dummies = pd.concat([resume_dummies, job_dummies])

### Combine with Count embedding ###
pos1_full_count = pd.DataFrame(pos1_embedding).merge(all_dummies
                                                     , how="left"
                                                     , on="ID")
pos1_full_count.drop_duplicates(subset="ID", inplace=True)
pos1_full_count = pos1_full_count.fillna(value=0)

pos2_full_count = pd.DataFrame(pos2_embedding).merge(all_dummies
                                                     , how="left"
                                                     , on="ID")
pos2_full_count.drop_duplicates(subset="ID", inplace=True)
pos2_full_count = pos2_full_count.fillna(value=0)

pos3_full_count = pd.DataFrame(pos3_embedding).merge(all_dummies
                                                     , how="left"
                                                     , on="ID")
pos3_full_count.drop_duplicates(subset="ID", inplace=True)
pos3_full_count = pos3_full_count.fillna(value=0)

pos4_full_count = pd.DataFrame(pos4_embedding).merge(all_dummies
                                                     , how="left"
                                                     , on="ID")
pos4_full_count.drop_duplicates(subset="ID", inplace=True)
pos4_full_count = pos4_full_count.fillna(value=0)

pos5_full_count = pd.DataFrame(pos5_embedding).merge(all_dummies
                                                     , how="left"
                                                     , on="ID")
pos5_full_count.drop_duplicates(subset="ID", inplace=True)
pos5_full_count = pos5_full_count.fillna(value=0)

### Combine with TFIDF embedding ###
pos1_full_tfidf = pos1_tfidf.merge(all_dummies
                                   , how="left"
                                   , on="ID")
pos1_full_tfidf.drop_duplicates(subset="ID", inplace=True)
pos1_full_tfidf = pos1_full_tfidf.fillna(value=0)

pos2_full_tfidf = pos2_tfidf.merge(all_dummies
                                   , how="left"
                                   , on="ID")
pos2_full_tfidf.drop_duplicates(subset="ID", inplace=True)
pos2_full_tfidf = pos2_full_tfidf.fillna(value=0)

pos3_full_tfidf = pos3_tfidf.merge(all_dummies
                                   , how="left"
                                   , on="ID")
pos3_full_tfidf.drop_duplicates(subset="ID", inplace=True)
pos3_full_tfidf = pos3_full_tfidf.fillna(value=0)

pos4_full_tfidf = pos4_tfidf.merge(all_dummies
                                   , how="left"
                                   , on="ID")
pos4_full_tfidf.drop_duplicates(subset="ID", inplace=True)
pos4_full_tfidf = pos4_full_tfidf.fillna(value=0)

pos5_full_tfidf = pos5_tfidf.merge(all_dummies
                                   , how="left"
                                   , on="ID")
pos5_full_tfidf.drop_duplicates(subset="ID", inplace=True)
pos5_full_tfidf = pos5_full_tfidf.fillna(value=0)

### Convert data to sparse matrix and split for cv###
pos1_spr = sp.sparse.csr_matrix(pos1_full_tfidf.set_index("ID").values)

pos1_train, pos1_test = random_train_test_split(pos1_spr
                                                , test_percentage=0.25
                                                , random_state = None)

### create and train LightFM model ###
NUM_THREADS = 4
NUM_COMPONENTS = 5
NUM_EPOCHS = 30
ITEM_ALPHA = 1e-6

pos1_model = LightFM(loss='warp'
                    , item_alpha=ITEM_ALPHA
                    , no_components=NUM_COMPONENTS)


%time pos1_model = pos1_model.fit(pos1_train, epochs=NUM_EPOCHS, num_threads=NUM_THREADS)

train_auc = auc_score(pos1_model, pos1_train, num_threads=NUM_THREADS).mean()
print('train AUC: %s' %train_auc)
test_auc = auc_score(pos1_model, pos1_test, num_threads=NUM_THREADS).mean()
print('test AUC: %s' %test_auc)

train_precision = precision_at_k(pos1_model, pos1_train, k=10).mean()
print('train precision at k: %s' %train_precision)
test_precision = precision_at_k(pos1_model, pos1_test, k=10).mean()
print('test precision at k: %s' %test_precision)

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

