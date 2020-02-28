import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x = pd.read_csv("~/data/Candidate Report.csv", encoding = 'latin-1').fillna('')
x = x[['Req ID', 'Req Title', 'Candidate ID', 'Gender', 'Ethinicity',
       'Source', 'Job Application Source', 'Latest Recruiting Step',
       'Last Recruiting Stage','Resume Text']]
x.Gender = x.Gender.map({"Group 1": 1, 'Group 2': 0}).astype(int)
x.columns = ['ReqID', 'ReqTitle', 'CanID', 'IsMale', 'Ethinicity',
             'Source', 'JobSource','LatestStep', 'LastStage','Text']
x.head()
 
jobs_reviewed_atleast_once = ['Review', 'Completion',  'Phone Screen',
                              'Schedule Interview', 'Offer Rejected',
                              'Schedule interview', 
                              'No Show (Interview / First Day)', 'Offer',
                              'Second Round Interview', 
                              'Background Check', 'Revise Offer',
                              'Final Round Interview']


#percent of women applicants
x.IsMale.value_counts()[0]/len(x)
#percent of applicants that recruiters were interested in that are women
temp_df = x[x['LatestStep'].isin(jobs_reviewed_atleast_once)]
len(temp_df[temp_df.IsMale ==  0])/len(temp_df)

#looking at races
races = pd.DataFrame(columns=('pct_overall','pct_interested_in'))
race_label =[]
for race in x.Ethinicity.unique():
  a = x.Ethinicity.value_counts()[race]/len(x)
  #percent of applicants that recruiters were interested in that are women
  temp_df = x[x['LatestStep'].isin(jobs_reviewed_atleast_once)]
  b = len(temp_df[temp_df.Ethinicity ==  race])/len(temp_df)
  races = races.append({'pct_overall': a,'pct_interested_in': b}, ignore_index=True)
  race_label.append(race)
races['advantage'] = races.pct_interested_in - races.pct_overall
races.index = race_label

races.sort_values(by='advantage', ascending=False)
#group 0 is more likely to be a candidate they are interested
#group 9 is less likely to be a reviewed candidate --> black candidates

pd.set_option('max_colwidth',100)
x.Text[x.Ethinicity == 'Group 0'].drop_duplicates()