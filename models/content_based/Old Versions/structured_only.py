import pandas as pd
import numpy as np
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import RegexpTokenizer

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

##### Stitch ideal candidate with resumes together ########

# a function to look up resumes for a position
# and put their structured features in one df

def output_structured(jobID, job_dummies, resume_dummies):
  
  # get dummies for job description
  jd_df = job_dummies[job_dummies.ReqID == jobID]
  # get dummies for resumes 
  resume_df = resume_dummies[resume_dummies.ReqID == jobID]
  # drop req ids from resumes
  resume_df.drop(["ReqID"], inplace=True, axis=1)
  # rename the col names for ID
  jd_df.rename(columns = {'ReqID':'ID'}, inplace=True)
  resume_df.rename(columns = {'CanID':'ID'}, inplace=True)
  # concat together
  df = pd.concat([jd_df, resume_df])
  
  return df

### run function on selected positions ###

pos1_structured = output_structured("e3625ad"
                                    , job_dummies
                                    , resume_dummies)

pos2_structured = output_structured("39ee3f"
                                    , job_dummies
                                    , resume_dummies)

pos3_structured = output_structured("45de815"
                                    , job_dummies
                                    , resume_dummies)

pos4_structured = output_structured("40a2c38"
                                    , job_dummies
                                    , resume_dummies)

pos5_structured = output_structured("63146c6"
                                    , job_dummies
                                    , resume_dummies)

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


#### position 1 #####

ordered_candidate_list_pos1 = RecommendTopCandidates(jobID='e3625ad'
                                                    , full_df=pos1_structured
                                                    , num_candidates=10)

df_list = []
for i in ordered_candidate_list_pos1:
  df_list.append(resume_text.loc[resume_text['Candidate ID'] == i])

# show result
job_text.loc[job_text["Req ID"] == "e3625ad"]
pd.concat(df_list)


#### position 2 #####

ordered_candidate_list_pos2 = RecommendTopCandidates(jobID='39ee3f'
                                                    , full_df=pos2_structured
                                                    , num_candidates=10)

df_list = []
for i in ordered_candidate_list_pos2:
  df_list.append(resume_text.loc[resume_text['Candidate ID'] == i])

# show result
job_text.loc[job_text["Req ID"] == "39ee3f"]
pd.concat(df_list)



#### position 3 #####

ordered_candidate_list_pos3 = RecommendTopCandidates(jobID='45de815'
                                                    , full_df=pos3_structured
                                                    , num_candidates=10)

df_list = []
for i in ordered_candidate_list_pos3:
  df_list.append(resume_text.loc[resume_text['Candidate ID'] == i])

# show result
job_text.loc[job_text["Req ID"] == "45de815"]
pd.concat(df_list)



#### position 4 #####

ordered_candidate_list_pos4 = RecommendTopCandidates(jobID='40a2c38'
                                                    , full_df=pos4_structured
                                                    , num_candidates=10)

df_list = []
for i in ordered_candidate_list_pos4:
  df_list.append(resume_text.loc[resume_text['Candidate ID'] == i])

# show result
job_text.loc[job_text["Req ID"] == "40a2c38"]
pd.concat(df_list)



#### position 5 #####

ordered_candidate_list_pos5 = RecommendTopCandidates(jobID='63146c6'
                                                    , full_df=pos5_structured
                                                    , num_candidates=10)

df_list = []
for i in ordered_candidate_list_pos5:
  df_list.append(resume_text.loc[resume_text['Candidate ID'] == i])

# show result
job_text.loc[job_text["Req ID"] == "63146c6"]
pd.concat(df_list)

