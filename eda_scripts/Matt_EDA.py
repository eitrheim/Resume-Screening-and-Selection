import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
from rake_nltk import Rake

candid = pd.read_csv('data/Candidate Report.csv', encoding = 'latin-1')
candid.columns

#checking null values within columns for candidate data
total = candid.isnull().sum().sort_values(ascending=False)
percent_1 = candid.isnull().sum()/candid.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis = 1, keys=['Total', '%'], sort=False)
missing_data.head(7)
candid.shape

candid.Country.value_counts()
candid["Job Application Source"].value_counts()
uniqueCandidates = candid["Candidate ID"].unique()
uniqueCandidates.shape
#Parsing involving skills fails to pickup important features, 
#will have to engineer our own parsing framework for this.

pd.options.display.max_colwidth = 2000

reqRepo = pd.read_csv('Req Report.csv')
reqRepo.columns

candid = pd.read_csv('data/full_requisition_data.csv', encoding = 'latin-1')
candid.columns
candid.shape
candid["Candidate ID"].value_counts()
uniqueCandidates = candid["Candidate ID"].unique()
uniqueCandidates.shape
candid["Job Description"][4]

obs = pd.read_csv("JobDescriptionParser-master/JobDescriptionParser-master/DescriptionParser/output/JOB_DESCRIPTION_summary.csv")
obs["Job Description"][61]
obs["Skills"][61]
obs[obs["Skills"] == '[]'].index

MBO = pd.read_excel('data/MBO_Scores.xlsx')
MBO.columns



parsed_resume = pd.read_csv("Resume-Parser-master-new/data/schema/transform.csv")
parsed_resume
    
transformCSVJD pd.read_csv("JobDescriptionParser-master/JobDescriptionParser-master/DescriptionParser/schema/transform.csv")
transformCSVJD

transformCSVJD = pd.read_csv("JobDescriptionParser-master/JobDescriptionParser-master/DescriptionParser/schema/transform.csv")
transformCSVJD

JOBPARSERMINI = pd.read_csv("Resume-Parser-FOR-DESCRIPTIONS/data/output/resume_summary.csv")
JOBPARSERMINI.soft_skills
JOBPARSERMINI.technical_skills
JOBPARSERMINI.head(5)


from scipy import sparse
interaction_sparse_V4 = sparse.load_npz('data/interaction_v4.npz')
interaction_sparse_V4
interaction_sparse_V3 = sparse.load_npz('data/interaction_v3.npz')
interaction_sparse_V3
