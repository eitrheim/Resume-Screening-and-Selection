import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from spacy.lang.en.stop_words import STOP_WORDS
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from gensim.models import doc2vec
from collections import namedtuple
# hide settingwithcopywarning
pd.options.mode.chained_assignment = None

def rank(jobID, topX):

    # read structure + one hot encoded dfs
    job_dummies_ideal = pd.read_csv("~/PycharmProjects/Resume-Screening-and-Selection/Resume-Parser-JOBS/data/job_description_one_hot_ideal_FULL.csv")
    resume_dummies = pd.read_csv("~/PycharmProjects/Resume-Screening-and-Selection/Resume-Parser-master-new/data/output/resume_summary_one_hot.csv")

    job_features_ideal = job_dummies_ideal[job_dummies_ideal.columns[:44]]
    resume_features = resume_dummies[resume_dummies.columns[:46]]
    job_dummies_ideal.drop(job_dummies_ideal.columns[1:44], axis=1, inplace=True)
    resume_dummies.drop(resume_dummies.columns[2:46], axis=1, inplace=True)

    # read raw df text data and vectorize for embeddings
    EMAIL_REGEX = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,4}"


    def remove_stop_words(x):
        try:
            x = re.sub('(\w)(/)(\w)',r'\1 \3', x)  # for responsibilities/accountabilities
            x = re.sub(EMAIL_REGEX, "", x)  # for email
            x = re.sub('[^a-z- ]', '', x)  # for punctuation
            x = re.sub(' - ', ' ', x)  # for dashes
            x = re.sub('-', ' ', x)  # for dashes
            x = re.sub(' \w ', ' ', x)  # for single letters
            x = re.sub('\s\s+', ' ', x)  # for multiple spaces

            word_tokens = word_tokenize(x)
            filtered_sentence = [w for w in word_tokens if not w in STOP_WORDS]
            return filtered_sentence
        except Exception as e:
            print("error:", e)
            print('string:', x)
            sys.exit(1)


    job_text = job_features_ideal[['ReqID', 'text']]
    job_text["text"].replace(r'[\d]', '', regex=True, inplace=True)
    job_text["text"] = job_text["text"].astype(str).apply(lambda x: x.lower().replace('\r', ' ').replace('\n', ' ').replace('\t', ' '))
    job_text["text"] = job_text["text"].apply(remove_stop_words)

    resume_text = resume_features[['ReqID', 'CanID', 'text']]
    resume_text["text"].replace(r'[\d]', '' ,regex=True, inplace=True) # remove numbers
    resume_text["text"] = resume_text["text"].astype(str).apply(lambda x: x.lower().replace('\r', ' ').replace('\n', ' ').replace('\t', ' '))
    # removing 1 and 2 letter words
    resume_text["text"] = resume_text["text"].apply(lambda x: re.sub('\s\w{1,2}\s', ' ', x))
    resume_text["text"] = resume_text["text"].apply(remove_stop_words)

    resume_dummies.rename(columns={'CanID': 'ID'}, inplace=True)
    job_dummies_ideal['ID'] = job_dummies_ideal['ReqID']
    job_dummies_ideal = job_dummies_ideal[resume_dummies.columns]
    all_dummies_ideal = pd.concat([resume_dummies, job_dummies_ideal])


    def GenerateCountEmbedding(req_id, job_text_df, resume_text_df):
        pos_jd_text = job_text[job_text["ReqID"] == req_id]
        pos_resume_text = resume_text[resume_text["ReqID"] == req_id]
        pos_jd_text.rename(columns={'ReqID': 'ID', 'Job Description': 'text'}, inplace=True)
        pos_jd_text.ID = req_id
        pos_jd_text = pos_jd_text[['ID', 'text']]
        pos_resume_text.rename(columns={'CanID': 'ID', 'Resume Text': 'text'}, inplace=True)
        pos_resume_text = pos_resume_text[['ID', 'text']]

        df = pos_jd_text.append(pos_resume_text)
        df.set_index('ID', inplace=True)

        tokenizer = RegexpTokenizer(r'\w+')
        df['text'] = df['text'].apply(lambda x: [''] if x[0] == 'nan' else x)
        #df['text'] = df['text'].apply(lambda x: tokenizer.tokenize(x))
        df['text'] = df['text'].apply(lambda x: ' '.join(x))
        count = CountVectorizer()
        pos_embedding = count.fit_transform(df['text'])
        pos_embedding = pd.DataFrame(pos_embedding.toarray())
        pos_embedding.insert(loc=0, column="ID", value=df.index)

        return pos_embedding


    def RecommendTop(jobID, full_df):  # Cos Sim and rank candidates
        # returns x recommended resume ID's based on Job Description
        recommended_candidates = []
        full_df.fillna(0, inplace=True)
        indices = pd.Series(full_df["ID"])
        cos_sim = cosine_similarity(full_df.drop("ID", axis=1))  # pairwise similarities for all samples in the df
        try:
            idx = indices[indices == jobID].index[0]
            score_series = pd.Series(cos_sim[idx]).sort_values(ascending=False)
            for i in score_series.index:
                recommended_candidates.append(list(indices)[i])
        except IndexError:
            print(jobID, 'had and index error')
        output = pd.DataFrame({'Candidate ID': recommended_candidates, 'cosine': score_series})
        output = output[output['Candidate ID'] != jobID]
        return output


    count_embeddings = GenerateCountEmbedding(jobID, job_text, resume_text)
    count_embeddings['ReqID'] = np.repeat(jobID, len(count_embeddings))
    all_features = pd.DataFrame(count_embeddings).merge(all_dummies_ideal, how="left", on=["ID", 'ReqID'])
    rankings = RecommendTop(jobID=jobID, full_df=all_features.drop('ReqID', axis=1))
    try:
        rankings = rankings[:topX]
    except:
        pass
    return rankings
