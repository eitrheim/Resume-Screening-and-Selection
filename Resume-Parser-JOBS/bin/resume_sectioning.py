# coding: utf-8

import numpy as np

def create_columns(observations):
    df = observations

    df['Language'] = np.repeat("", len(df))
    df['Work'] = np.repeat("", len(df))
    df['Summaries'] = np.repeat("", len(df))
    df['Skill'] = np.repeat("", len(df))
    df['Member'] = np.repeat("", len(df))
    df['Writing'] = np.repeat("", len(df))
    df['Researching'] = np.repeat("", len(df))
    df['Honor'] = np.repeat("", len(df))
    df['Activity'] = np.repeat("", len(df))
    df['Education'] = np.repeat("", len(df))
    df['Extracurriculars'] = np.repeat("", len(df))

    return df
