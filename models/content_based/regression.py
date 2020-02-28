import pandas as pd
import numpy as np
#import nltk
#from nltk.tokenize import word_tokenize
#from spacy.lang.en.stop_words import STOP_WORDS
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from gensim.models import doc2vec
from collections import namedtuple
# hide settingwithcopywarning
pd.options.mode.chained_assignment = None

#################### Read data ####################
# read structure + one hot encoded dfs
resume_dummies = pd.read_csv("~/data/resume_summary_one_hot.csv")
# just the structured
resume_features = resume_dummies[resume_dummies.columns[:47]]
# just the one hot
resume_dummies.drop(resume_dummies.columns[2:47], axis=1, inplace=True)
resume_text = pd.read_csv("~/data/Candidate Report_tokenized.csv").fillna('')
resume_dummies.rename(columns = {'CanID':'ID'}, inplace=True)
#drop all rows that do not have a resume
resume_text = resume_text[resume_text['Resume Text'] != '[\'nan\']']
resume_dummies = resume_dummies[resume_dummies.ID.isin(resume_text['Candidate ID'])]



jobs_reviewed_atleast_once = {'Review':1,
                              'Completion':1,
                              'Phone Screen':1,
                              'Schedule Interview':1,
                              'Offer Rejected':1,
                              'Schedule interview':1, 
                              'No Show (Interview / First Day)':1,
                              'Offer':1,
                              'Second Round Interview':1, 
                              'Background Check':1,
                              'Revise Offer':1,
                              'Final Round Interview':1,
                              'Voluntary Withdrew':1, # NEW ADDT
                              'Salary Expectations too high':1, # NEW ADDT
                              'Skills or Abilities':1} # NEW ADDT
resume_text['Latest Recruiting Step']=resume_text['Latest Recruiting Step'].map(jobs_reviewed_atleast_once)
resume_text['Latest Recruiting Step']=resume_text['Latest Recruiting Step'].fillna(0)

df = resume_text[['Req ID','Candidate ID','Latest Recruiting Step']]

df = df.merge(resume_dummies, left_on=['Req ID','Candidate ID'],right_on=['ReqID','ID'])

y = df['Latest Recruiting Step']

x = df[df.columns[5:]]

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=.2)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(xtrain,ytrain)
ypred = dt.predict(xtest)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(ytest,ypred))
print(classification_report(ytest, ypred))

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100)
rf.fit(xtrain,ytrain)
rf_ypred = rf.predict(xtest)

print(confusion_matrix(ytest,rf_ypred))
print(classification_report(ytest, rf_ypred))

temp = pd.DataFrame({'feature':xtest.columns,
             'importance':rf.feature_importances_})

temp = temp.sort_values(by='importance',ascending=False)
temp = temp[temp.importance !=0]
temp.head(10)

