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

stages = [item for sublist in stages for item in sublist]
stages = pd.DataFrame(stages, columns=['Stages'])['Stages'].value_counts()
good_picks = stages[stages.index.isin(jobs_reviewed_atleast_once)]
p_at_k11 = sum(good_picks)/sum(stages.values)
p_at_k11 
outcomes.update({11: p_at_k11})

#################### results #################### 