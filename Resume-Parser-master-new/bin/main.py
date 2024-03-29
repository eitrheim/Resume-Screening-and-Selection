import sys

# importing user defined modules
import field_extraction
import lib
import resume_sectioning

import inspect
import logging
import os
import pandas as pd
import numpy as np
import re
import spacy
#import en_core_web_sm


# hide settingwithcopywarning
pd.options.mode.chained_assignment = None

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  # get location of main.py
parentdir = os.path.dirname(currentdir)  # get parent directory of main.py (where repository is on local)
sys.path.insert(0, parentdir)  # sys.path is the module search path


def main(root_file_path, job_id):

    logging.getLogger().setLevel(logging.WARNING)  # essentially does print statements to help debug (WARNING)
    # logging explained https://appdividend.com/2019/06/08/python-logging-tutorial-with-example-logging-in-python/

    observations = extract(root_file_path)  # get text from pdf resumes

    # to make it like Kraft's
    observations['ReqID'] = np.repeat(job_id, len(observations))
    observations.dropna(inplace=True)

    # to compare it to already parsed info to not have to reparse
    current_observations = pd.read_csv(root_file_path + "Resume-Parser-master-new/data/output/resume_summary.csv")
    for i in observations.index:
        if observations.text[i] in current_observations.text[current_observations.ReqID == observations.ReqID[i]].values:
            observations.drop(i, axis=0, inplace=True)

    # generating a candidate ID based off of the filepath (i.e. the resume name)
    observations['CanID'] = observations.file_path.apply(lambda x: x.split('/')[-1].lower().replace(' ', '')[:4] +
                                                                   str(ord(str(x.split('.')[-2].lower().replace(' ', '')[-1]))))

    print('New resumes being parsed:\n', observations['CanID'].values)

    if len(observations['CanID'].values) == 0:
        observations = current_observations
    else:
        observations = observations[['ReqID', 'CanID', 'text']]
        observations.drop_duplicates(inplace=True, keep='first')
        observations.reset_index(drop=True, inplace=True)
        observations.text.fillna('', inplace=True)
        observations.dropna(how='all', inplace=True)  # drop rows that are all na

        # to get the start (as an int) of resume sections
        observations = resume_sectioning.section_into_columns(observations)

        # get only words pertaining each sub-section
        observations = resume_sectioning.word_put_in_sections(observations)

        # to combine the sub-sections
        observations = resume_sectioning.combine_sections_preparse(observations)
        observations = observations[observations.text == observations.text]

        print("Loading Spacy Corpus")
        nlp = spacy.load('en_core_web_sm')
        # nlp = en_core_web_sm.load()
        print("Spacy Corpus Loaded \n")

        observations = transform(observations, root_file_path, nlp)  # extract data from resume sections

        # to combine the sub-sections one last time
        observations = resume_sectioning.combine_sections_postparse(observations)
        observations = observations[observations.text == observations.text]

        # merge with the already parsed information
        observations = pd.concat([current_observations, observations])
        observations = observations[observations.text == observations.text]
        observations = observations[observations.text != '']
        observations = observations[observations.ReqID != '']
        observations = observations[observations.CanID != '']

    load(observations, root_file_path)  # save to csv to finish

    pass


def extract(root_file_path):
    logging.info('Begin extract')

    candidate_file_agg = list()  # for creating list of resume file paths
    for root, subdirs, files in os.walk(root_file_path + 'Resume-Parser-master-new/data/input/resumes'):
        files = filter(lambda f: f.endswith(('.pdf', '.PDF')), files)  # only read pdf
        folder_files = map(lambda x: os.path.join(root, x), files)
        candidate_file_agg.extend(folder_files)

    observations = pd.DataFrame(data=candidate_file_agg, columns=['file_path'])  # convert to df

    logging.info('Found {} candidate files'.format(len(observations.index)))
    observations['text'] = observations['file_path'].apply(lib.convert_pdf, root_file_path=root_file_path)  # get text from .pdf files

    observations['text'] = observations['text'].apply(lambda x: x.replace(chr(8217), '\''))
    # to get rid of characters that are not in utf-8
    observations['text'] = observations['text'].apply(lambda x: x.encode('utf-8', errors='ignore').decode('utf-8'))
    # to get rid of extra whitespace
    observations['text'] = observations['text'].apply(lambda x: re.sub(' +', ' ', x))

    logging.info('End extract')
    return observations


def transform(observations, root_file_path, nlp):
    logging.info("Extracting email, phone, GPA, and dates of work experience")
    observations = observations.fillna('')
    observations['candidate_name'] = observations['text'].apply(lambda x: field_extraction.candidate_name_extractor(x, nlp))
    observations['email'] = observations['text'].apply(lambda x: lib.term_match(x, field_extraction.EMAIL_REGEX))
    observations['phone'] = observations['text'].apply(lambda x: lib.term_match(x, field_extraction.PHONE_REGEX))
    observations['GPA'] = observations['text'].apply(lambda x: field_extraction.gpa_extractor(x))
    observations['years_experience'] = observations['Work'].apply(lambda x: field_extraction.years_of_experience(x))
    observations['mos_experience'] = field_extraction.months_of_experience(observations['years_experience'])

    # convert GPA to a single number
    GPA_REGEX = r"[01234]{1}\.[0-9]{1,3}"
    observations.GPA.fillna('[]', inplace=True)
    observations['GPAnum'] = observations.GPA.apply(lambda x: re.findall(re.compile(GPA_REGEX), str(x)))

    def getmax(x):
        try:
            y = max(x)
        except:
            y = 0
        return y

    observations['GPAmax'] = observations['GPAnum'].apply(lambda x: getmax(x))
    observations['GPAmax'] = observations['GPAmax'].apply(lambda x: np.nan if x == 0 else x)
    observations.filter(like='GPA')
    np.mean(observations['GPAmax'].astype('float'))
    observations.drop('GPAnum', axis=1, inplace=True)

    observations = field_extraction.extract_fields(observations, root_file_path)  # search for terms in whole resume

    return observations


def load(observations, root_file_path):
    output_path = root_file_path + 'Resume-Parser-master-new/data/output/resume_summary.csv'
    logging.info('Results being output to {}'.format(output_path))
    observations.to_csv(path_or_buf=output_path, index=False, encoding='utf-8')
    logging.info('End transform')
    
    pass


if __name__ == '__main__':
    main()
