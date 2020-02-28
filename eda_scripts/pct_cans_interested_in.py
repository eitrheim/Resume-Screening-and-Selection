import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

resume_text = pd.read_csv("~/data/Candidate Report_tokenized.csv").fillna('')
job_text = pd.read_csv("~/data/full_requisition_data_tokenized.csv").fillna('')

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

#how many candidates with NO RESUMES were they interested in?
temp_df = resume_text[resume_text['Resume Text'] ==  '[\'nan\']']
temp_df = temp_df.merge(job_text, how='left',on='Req ID')
temp_df = temp_df['Latest Recruiting Step'].value_counts()
sum(temp_df[temp_df.index.isin(jobs_reviewed_atleast_once)])
####### 860 CANDIDATES?! #######


#interested in atleast one candidate with a resume
temp_df = resume_text[resume_text['Latest Recruiting Step'].isin(jobs_reviewed_atleast_once)]
#temp_df = temp_df[temp_df['Resume Text'] !=  '[\'nan\']']
x = temp_df[['Req ID', 'Candidate ID','Resume Text']]
x = x.merge(job_text, how='left',on='Req ID')
x = x['Req ID'].value_counts()
x = x[x >= 1]
jobIDs = x.index


def pct_reviewed(low_lim, high_lim):
  temp_df = x[x >= low_lim][x < high_lim].index
  jobIDs = temp_df
  if len(temp_df) == 0:
    output = np.nan
  else:
    y = resume_text[resume_text['Req ID'].isin(jobIDs)]
    y = y['Latest Recruiting Step'].value_counts()
    output = sum(y[y.index.isin(jobs_reviewed_atleast_once)])/sum(y)
  return output

pcts = []
for i in list(range(0,245,15)):
  pcts.append(pct_reviewed(i, i+15))

plot_data = pd.DataFrame(pcts,list(range(0,245,15))).reset_index()
plot_data.columns = ['Number of Candidates Applied', 'Percent that Recruiters Were Interested In']
plt.scatter(x=plot_data.iloc[:,0],y=plot_data.iloc[:,1])

PercentInterestedIn = []
for i in x.index:
  y = resume_text[resume_text['Req ID'] == i]
  y = y['Latest Recruiting Step'].value_counts()
  output = sum(y[y.index.isin(jobs_reviewed_atleast_once)])/sum(y)
 # output = round(output*100)
  PercentInterestedIn.append(output)

import seaborn as sns
plot_data = pd.DataFrame(x.values,PercentInterestedIn).reset_index()
plot_data.columns = ['Percent that Recruiters Were Interested In', 'Number of Candidates to Apply']
sns.set()
sns.jointplot(x=plot_data[plot_data.columns[1]],y=plot_data[plot_data.columns[0]],s=3).ax_joint.legend_.remove()


#if a role has a lot of candidates apply to it, they are interested in more candidates
#this seems backwards, say if 1000 people apply, they shouldn't be interested in almost 1000



