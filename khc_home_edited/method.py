import pandas as pd
import numpy as np
import csv
import pickle

Jobs_path = "TestDescriptions.csv"
Jobs = pd.read_csv(Jobs_path, delimiter=',')

def get_JobID():
    IDs = np.array(Jobs.index.values.tolist())
    IDs = np.unique(IDs)
    IDs = IDs.tolist()
    return(IDs)

def get_Info(ID):
    return Jobs[Jobs.index == ID]

pickle.dump(Jobs, open('data.pkl','wb'))