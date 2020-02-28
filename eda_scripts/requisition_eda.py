import pandas as pd
import numpy as np
import nltk
from gensim.models import Word2Vec
import multiprocessing
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import sys
import spacy

# read all requisition files and compare
req_report = pd.read_csv('Req Report.csv')
req_csv = pd.read_csv('Requisition Data.csv')
# req_xlsx = pd.read_excel('Requisition Data.xlsx')

req_report.describe()
req_csv.describe()
# req_xlsx.describe()

# display column names
list(req_report.columns)
list(req_csv.columns)
# list(req_xlsx.columns)

# merge the two dfs
req_all = pd.merge(req_csv, req_report, 
                   on="Req ID", how="left")
req_all.head()
list(req_all.columns)

#right join so only keep positions from both df
req_both = pd.merge(req_csv, req_report, 
                   on="Req ID", how="right")
list(req_both.columns)

req_all.describe()
req_both.describe()

#compare the columns to see if they are equal
req_all['Req Title_x'].equals(req_all['Req Title_y'].dropna())
req_both['Req Title_x'].equals(req_both['Req Title_y'])
req_both['Job Requisition Status_x'].equals(req_both['Job Requisition Status_y'])
req_both['Candidate ID'].equals(req_both['Candidate ID(Hired)'])



# display the entries that are different from one sheet to another
df_compare = req_both[req_both['Req Title_x'] != req_both['Req Title_y']]
df_compare
df_compare2 = req_all[req_all['Req Title_x'] != req_all['Req Title_y']]
df_compare2

req_csv['Req ID'].isna().sum()
req_report['Req ID'].isna().sum()

# so it seems that both df have records unique to them, thus we need
# to outer join them in order to preserve all information
req_full = pd.merge(req_csv, req_report,
                   on="Req ID", how="outer")
req_full.describe()
list(req_full.columns)

# Fill x column with NA with data from y column

req_full['Req Title_x'].isna().sum()
req_full['Req Title_x'] = req_full.apply(
                          lambda row: row["Req Title_y"] 
                                 if np.isnan(row['Req Title_x'])
                                 else row['Req Title_x'])

req_full['Req Title_x'].fillna(req_full['Req Title_y'], inplace=True)

req_full['Job Requisition Status_x'].isna().sum()
req_full['Job Requisition Status_x'].fillna(req_full['Job Requisition Status_y'], inplace=True)

req_full['Job Description_x'].isna().sum()
req_full['Job Description_x'].fillna(req_full['Job Description_y'], inplace=True)
req_full['Job Description_y'].fillna(req_full['Job Description_x'], inplace=True)
req_full['Job Description_x'].equals(req_full['Job Description_y'])
df_compare = req_full[req_full['Job Description_x'] != req_full['Job Description_y']]
df_compare # the remaing different entries are the ones have both NA's from x and y

req_full['Date Requisition Opened_x'].isna().sum()
req_full['Date Requisition Opened_x'].fillna(req_full['Date Requisition Opened_y'], inplace=True)

req_full['offer completed/accepted date_x'].isna().sum()
req_full['offer completed/accepted date_x'].fillna(req_full['offer completed/accepted date_y'], inplace=True)

req_full['Candidate ID'].isna().sum()
req_full['Candidate ID'].fillna(req_full['Candidate ID(Hired)'], inplace=True)
req_full['Candidate ID(Hired)'].fillna(req_full['Candidate ID'], inplace=True)
req_full['Candidate ID'].equals(req_full['Candidate ID(Hired)']) # True

# drop duplicated columns
req_full.drop(["Req Title_y",
              "Job Requisition Status_y",
              "Job Description_y",
              "Date Requisition Opened_y",
              "offer completed/accepted date_y",
              "Candidate ID(Hired)"], axis=1, inplace=True)
list(req_full.columns)
# rename columns
req_full.rename(columns = {'Req Title_x':'Req Title',
                          'Job Requisition Status_x':'Job Requisition Status',
                          'Job Description_x':'Job Description',
                          'Date Requisition Opened_x':'Date Requisition Opened',
                          'offer completed/accepted date_x':'offer completed/accepted date'
                          }, inplace=True)

req_full.describe()

# save merged df to csv
req_full.to_csv('full_requisition_data.csv')

# calculate numbers of missing values
req_full.isnull().sum()
req_full.isnull().sum()/len(req_full)*100

# read all requisition files and compare
can_report = pd.read_csv('Candidate Report.csv', encoding="ISO-8859-1")
can_csv = pd.read_csv('Candidate Data.csv', encoding="ISO-8859-1")
can_xlsx = pd.read_excel('Candidate Data.xlsx')

can_report.describe()
can_csv.describe()
can_xlsx.describe()

# display column names
list(can_report.columns)
list(can_csv.columns)

can_report.isnull().sum()
can_report.isnull().sum()/len(can_report)*100



