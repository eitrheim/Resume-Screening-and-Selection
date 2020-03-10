from __future__ import absolute_import
import pandas as pd
import numpy as np
import csv
import sys

root_path = '/Users/matthewechols/PycharmProjects/Resume-Screening-and-Selection/Resume-Parser-JOBS/data/'
job_description = 'job_descriptions.csv'


def get_Job(JobID):
    sys.path.append(root_path)
    df_ = pd.read_csv(root_path+job_description)
    df_ = df_[df_.ReqID == JobID]
    return(df_.iloc[0][1])


def set_Job(JobID=np.nan, Descrip=np.nan):
    sys.path.append(root_path)
    with open(root_path + job_description, 'a', newline='\n') as newFile:
        newFileWriter = csv.writer(newFile)
        newFileWriter.writerow([])
        newFileWriter.writerow([JobID, Descrip])


def get_Index():
    sys.path.append(root_path)
    df_ = pd.read_csv(root_path+job_description)
    return df_.ReqID
