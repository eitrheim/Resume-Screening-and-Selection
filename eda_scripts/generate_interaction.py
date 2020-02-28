import pandas as pd
import numpy as np
from scipy import sparse
import matplotlib.pylab as plt
import scipy as sp

resume_text = pd.read_csv("data/cleaned_resume.csv", index_col=0)
job_text = pd.read_csv("~/data/full_requisition_data_tokenized.csv").fillna('')

resume_text['Last Recruiting Stage'].value_counts()
resume_text['Latest Recruiting Step'].value_counts()

resume_text.drop_duplicates(subset=['Req ID','Candidate ID'], keep='last', inplace=True)

###
interaction_dict = {'Review': 0
                   , 'Screen': 1
                   , 'Interview': 2
                   , 'Ready for Hire': 3
                   , 'Offer': 3
                   , 'Background Check': 3}

resume_text['interaction'] = resume_text['Last Recruiting Stage'].map(interaction_dict)
interaction_matrix = resume_text.pivot(index='Req ID'
                                      , columns='Candidate ID'
                                      , values='interaction')

interaction_matrix.fillna(0)
interaction_sparse = sparse.csr_matrix(interaction_matrix.values)
sparse.save_npz('data/interaction_v1.npz', interaction_sparse)

##### smaller interaction matrix with NRNC dropped ####

resume_subset = resume_text.drop(resume_text[resume_text['Latest Recruiting Step'] == 'Not Reviewed Not Considered'].index)
resume_subset['interaction'] = resume_subset['Last Recruiting Stage'].map(interaction_dict)
interaction_matrix_small = resume_subset.pivot(index='Req ID'
                                      , columns='Candidate ID'
                                      , values='interaction')

interaction_matrix_small.fillna(0)
interaction_sparse_small = sparse.csr_matrix(interaction_matrix_small.values)
sparse.save_npz('data/interaction_small.npz', interaction_sparse_small)

##### 
interaction_dict_v2 = {'Not Reviewed Not Considered': 0
                       , 'Hiring Restrictions': 0
                       , 'Hiring Policy': 0
                       , 'Voluntary Withdrew' : 0
                       , 'Position Cancelled': 1
                       , 'Selected other more qualified candidate' : 1
                       , 'Basic Qualifications' : 1
                       , 'Salary Expectations too high' : 1
                       , 'Review' : 2
                       , 'Skills or Abilities' : 2
                       , 'Phone Screen' : 3
                       , 'Schedule Interview' : 3
                       , 'Schedule interview' : 3
                       , 'No Show (Interview / First Day)' : 3
                       , 'Second Round Interview' : 4
                       , 'Final Round Interview' : 4
                       , 'Completion' : 5
                       , 'Offer' : 5
                       , 'Offer Rejected' : 5
                       , 'Revise Offer' : 5
                       , 'Background Check' : 5}

resume_text['interaction'] = resume_text['Latest Recruiting Step'].map(interaction_dict_v2)
interaction_matrix = resume_text.pivot(index='Req ID'
                                      , columns='Candidate ID'
                                      , values='interaction')

# interaction_matrix.fillna(0)
interaction_sparse = sparse.csr_matrix(interaction_matrix.values)
sparse.save_npz('data/interaction_v2.npz', interaction_sparse)

#####
resume_text['interaction'] = resume_text['Last Recruiting Stage'].map(interaction_dict)
interaction_matrix = resume_text.pivot(index='Req ID'
                                      , columns='Candidate ID'
                                      , values='interaction')
interaction_sparse = sparse.csr_matrix(interaction_matrix.values)
sparse.save_npz('data/interaction_v3.npz', interaction_sparse)


#####
interaction_dict_v4 = {'Not Reviewed Not Considered': 0
                       , 'Hired For Another Job': 0
                       , 'Hiring Restrictions': 0
                       , 'Hiring Policy': 0
                       , 'Voluntary Withdrew' : 0
                       , 'Position Cancelled': 0
                       , 'Selected other more qualified candidate' : 0
                       , 'Basic Qualifications' : 0
                       , 'Salary Expectations too high' : 0
                       , 'Skills or Abilities' : 0
                       , 'Review' : 1
                       , 'Phone Screen' : 2
                       , 'Schedule Interview' : 3
                       , 'Schedule interview' : 3
                       , 'No Show (Interview / First Day)' : 3
                       , 'Second Round Interview' : 4
                       , 'Final Round Interview' : 4
                       , 'Completion' : 5
                       , 'Offer' : 5
                       , 'Offer Rejected' : 5
                       , 'Revise Offer' : 5
                       , 'Background Check' : 5}

resume_text['interaction'] = resume_text['Latest Recruiting Step'].map(interaction_dict_v4)
resume_text = resume_text.sort_values('Req ID')

interaction_matrix = resume_text.pivot(index='Req ID'
                                      , columns='Candidate ID'
                                      , values='interaction')
interaction_matrix = interaction_matrix.sort_values()

#interaction_sparse = sparse.csr_matrix(interaction_matrix.values)
#interaction_sparse2 = sparse.coo_matrix(interaction_matrix.values)
sparse.save_npz('data/interaction_v4.npz', interaction_sparse)


##### binary
job_text = pd.read_csv("~/data/full_requisition_data_tokenized.csv").fillna('')
job_text.drop('Job Description Clean',axis=1, inplace=True)
resume_text = pd.read_csv("~/data/Candidate Report_tokenized.csv").fillna('')
#drop all rows that do not have a resume
resume_text = resume_text[resume_text['Resume Text'] != '[\'nan\']']
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
x = temp_df.merge(job_text, how='left',on='Req ID')
x = x['Req ID'].value_counts()
x = x[x >= 5]
jobIDs = x.index
temp_df= resume_text[resume_text['Req ID'].isin(jobIDs)]

#drop duplicates
temp_df.drop_duplicates(subset=['Req ID','Candidate ID'], keep='last', inplace=True)

interaction_dict_binary = {'Not Reviewed Not Considered': 0
                       , 'Hiring Restrictions': 0
                       , 'Hiring Policy': 0
                       , 'Voluntary Withdrew' : 0
                       , 'Position Cancelled': 0
                       , 'Skills or Abilities': 0
                       , 'Selected other more qualified candidate' : 0
                       , 'Basic Qualifications' : 0
                       , 'Salary Expectations too high' : 0
                       , 'Hired For Another Job' : 0
                       , 'Review' : 1
                       , 'Phone Screen' : 1
                       , 'Schedule Interview' : 1
                       , 'Schedule interview' : 1
                       , 'No Show (Interview / First Day)' : 1
                       , 'Second Round Interview' : 1
                       , 'Final Round Interview' : 1
                       , 'Completion' : 1
                       , 'Offer' : 1
                       , 'Offer Rejected' : 1
                       , 'Revise Offer' : 1
                       , 'Background Check' : 1}

temp_df['interaction'] = temp_df['Latest Recruiting Step'].map(interaction_dict_binary)
interaction_matrix = temp_df.pivot(index='Req ID', columns='Candidate ID', values='interaction')

interaction_matrix = interaction_matrix.fillna(0).astype(int)
interaction_sparse = sparse.csr_matrix(interaction_matrix.values)
sparse.save_npz('data/interaction_v_binary.npz', interaction_sparse)

interaction_sparse.data = np.nan_to_num(interaction_sparse.data, nan=0, copy=False)

plt.grid(b=None)

plt.spy(interaction_sparse2, aspect='auto', markersize=0.001)

plt.spy(interaction_sparse, aspect='auto', precision=0.1, markersize=1,marker=',')
plt.spy(interaction_sparse, aspect='auto', precision=0.1, markersize=1,marker='_')

##### updated with client feedback
interaction_dict_v5 = {'Not Reviewed Not Considered': 0
                       , 'Hired For Another Job': 0
                       , 'Hiring Restrictions': 0
                       , 'Hiring Policy': 0
                       , 'Voluntary Withdrew' : 1
                       , 'Position Cancelled': 0
                       , 'Selected other more qualified candidate' : 0
                       , 'Basic Qualifications' : 0
                       , 'Salary Expectations too high' : 1
                       , 'Skills or Abilities' : 2
                       , 'Review' : 1
                       , 'Phone Screen' : 2
                       , 'Schedule Interview' : 3
                       , 'Schedule interview' : 3
                       , 'No Show (Interview / First Day)' : 3
                       , 'Second Round Interview' : 4
                       , 'Final Round Interview' : 4
                       , 'Completion' : 5
                       , 'Offer' : 5
                       , 'Offer Rejected' : 5
                       , 'Revise Offer' : 5
                       , 'Background Check' : 5}
resume_text['interaction'] = resume_text['Last Recruiting Stage'].map(interaction_dict_v5)
interaction_matrix = resume_text.pivot(index='Req ID'
                                      , columns='Candidate ID'
                                      , values='interaction')
interaction_sparse = sparse.csr_matrix(interaction_matrix.values)
sparse.save_npz('data/interaction_v5.npz', interaction_sparse)

plt.grid(b=None)
plt.spy(interaction_sparse, aspect='auto', precision=0.1, markersize=1,marker=',')