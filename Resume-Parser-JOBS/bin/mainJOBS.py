import sys

# importing user defined modules
import jd_field_extraction
import job_lib
import jd_sectioning

import inspect
import logging
import os
import pandas as pd


# hide settingwithcopywarning
pd.options.mode.chained_assignment = None


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  # get location of main.py
parentdir = os.path.dirname(currentdir)  # get parent directory of main.py (where repository is on local)
sys.path.insert(0, parentdir)  # sys.path is the module search path


def main(root_file_path):

    logging.getLogger().setLevel(logging.WARNING)  # essentially does print statements to help debug (WARNING)
    # logging explained https://appdividend.com/2019/06/08/python-logging-tutorial-with-example-logging-in-python/

    # read in job descriptions
    observations = pd.read_csv(root_file_path + "Resume-Parser-JOBS/data/job_descriptions.csv")
    observations.columns = ['ReqID', 'text']

    # already parsed stuff
    current_observations = pd.read_csv(root_file_path + "Resume-Parser-JOBS/data/job_description_one_hot_ideal_FULL.csv")
    print(current_observations.columns)
    for i in observations.index:
        if observations.text[i] in current_observations.text[current_observations.ReqID == observations.ReqID[i]].values:
            observations.drop(i, axis=0, inplace=True)

    print('New job descriptions being parsed:\n', observations['ReqID'].values)
    observations.drop_duplicates(inplace=True, keep='first')
    observations.reset_index(drop=True, inplace=True)
    observations.text.fillna('', inplace=True)
    observations.dropna(how='all', inplace=True) # drop rows that are all na
    observations['text'] = observations['text'].apply(lambda x: x.replace('Kraft Heinz is an EO employer', ''))
    observations['text'] = observations['text'].apply(lambda x: x.replace('Minorities/Women/Vets/Disabled and other protected categories', ''))
    
    observations = jd_sectioning.create_columns(observations)  # add columns to match resume df
  
    observations = transform(observations, root_file_path)  # extract data from resume sections

    # merge with the already parsed information
    observations = pd.concat([current_observations, observations])

    load(observations, root_file_path)  # save to csv to finish

    pass


def transform(observations, root_file_path):
    logging.info('Begin transform')
    
    logging.info("Extracting years of experience wanted")
    observations = observations.fillna('')
    observations['email'] = ''
    observations['phone'] = ''
    observations['GPA'] = ''
    observations['years_experience'] = observations['text'].apply(lambda x: jd_field_extraction.years_of_experience(x))
    observations['mos_experience'] = jd_field_extraction.months_of_experience(observations['years_experience'])

    observations = jd_field_extraction.extract_fields(observations, root_file_path)  # search for terms in whole resume

    logging.info('End transform')
    return observations


def load(observations, root_file_path):
    output_path = root_file_path + 'Resume-Parser-JOBS/data/output/job_description_summary_FULL.csv'
    logging.info('Results being output to {}'.format(output_path))
    observations.to_csv(path_or_buf=output_path, index=False, encoding='utf-8')

    pass


if __name__ == '__main__':
    main()
