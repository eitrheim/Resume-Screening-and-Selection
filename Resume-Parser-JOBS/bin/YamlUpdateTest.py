import pandas as pd
import numpy as np
PATH = "Resume-Parser-JOBS/data/output/"
jobs = pd.read_csv(PATH + "job_description_summary.csv")

jobs = jobs.replace('\[', "", regex=True)
jobs = jobs.replace('\]', "", regex=True)
jobs = jobs.replace(r'^\s*$', np.nan, regex=True)
total = jobs.isnull().sum().sort_values(ascending=False)
percent_1 = jobs.isnull().sum()/jobs.isnull().count()*100
percent_2 = (round(percent_1,1)).sort_values(ascending=False)
missing_data=pd.concat([total, percent_2], axis = 1, keys=["Total", "%"], sort=False)
missing_data.head(46)

jobs.columns

jobs.bachelor_education_level
jobs.master_education_level
jobs.text[497]
jobs[jobs.mos_experience != 0]