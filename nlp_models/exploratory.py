!pip3 install rake_nltk
import pandas as pd
from rake_nltk import Rake
import numpy as np
import nltk

from spacy.lang.en import English
nlp = English()
from spacy.lang.en.stop_words import STOP_WORDS

from nltk.tokenize import RegexpTokenizer
obs = pd.read_csv("JobDescriptionParser-master/JobDescriptionParser-master/DescriptionParser/output/JOB_DESCRIPTION_summary.csv")

r = Rake(stopwords=STOP_WORDS)
a=r.extract_keywords_from_text(obs["Job Description"][1])
b=r.get_ranked_phrases_with_scores()
b
# process resume data
df = pd.read_csv("/home/cdsw/ReqTokenizedWords.csv", header=None)
df = df[[1]]
df.columns = ["bag_of_words"]
df.head()

can_df = pd.read_csv('/home/cdsw/data/Candidate Report.csv', encoding = 'latin-1')
list(can_df.columns)
can_df.head()

can_df.drop(["Req Title",
              "Requisition Open Date",
              "Gender",
              "Ethinicity",
              "Country",
              "Region",
              "Source",
              "Job Application Source",
              "Skills",
              "Work Experience",
              "Resume Text",
              "Skills as Text for Job Application"], axis=1, inplace=True)

len(df)
len(can_df)

# glue them together
resume_df = pd.concat([can_df, df], axis=1, ignore_index=True)
len(resume_df)
list(resume_df.columns)
resume_df.columns = ["Req ID","Candidate ID","Latest Recruiting Step",
                     "Last Recruiting Stage", "Resume Text"]
resume_df.head()

# process JD data
jd_df = pd.read_csv("data/full_requisition_data.csv")
req_df = pd.read_csv("ReqTokenizedWords.csv", header=None)
len(jd_df)
len(req_df)
list(jd_df.columns)
list(req_df)
jd_df.head()
req_df.head()
req_df = req_df[[1]]
jd_df = jd_df[["Req ID", "Req Title",
               "Job Requisition Status", "Candidate ID",
              "Division", "Function"]]
# glue
job_df = pd.concat([jd_df, req_df], axis=1, ignore_index=True)
job_df.columns = ["Req ID","Req Title",
               "Job Requisition Status", "Candidate ID",
              "Division", "Function", "Job Description"]
job_df.head()



### Clean text ###

# tokenize every text
tokenizer = RegexpTokenizer(r'\w+')


# remove numbers
resume_df["Resume Text"].replace(r'[\d]','',regex=True, inplace=True)
job_df["Job Description"].replace(r'[\d]','',regex=True, inplace=True)

# lower case all words
resume_df["Resume Text"] = resume_df["Resume Text"].str.lower()
job_df["Job Description"] = job_df["Job Description"].str.lower()

# remove stopwords
STOP_WORDS.add('')
# try_df["Resume Text"] = resume_df["Resume Text"].apply(lambda x: [str(word) for word in x if word not in STOP_WORDS])
# try_df["Job Description"] = job_df["Job Description"].apply(lambda x: [word for word in x if word not in STOP_WORDS])


resume_df.to_csv('data/cleaned_resume.csv', index=False)
job_df.to_csv('data/cleaned_job.csv', index=False)