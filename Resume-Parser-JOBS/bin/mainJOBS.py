# coding: utf-8

from __future__ import absolute_import
import sys
sys.path.append('../Resume-Parser-JOBS/bin')

# importing user defined modules
import field_extraction
import lib
import resume_sectioning

import inspect
import logging
import os
import pandas as pd


# hide settingwithcopywarning
pd.options.mode.chained_assignment = None


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  # get location of main.py
parentdir = os.path.dirname(currentdir)  # get parent directory of main.py (where repository is on local)
sys.path.insert(0, parentdir)  # sys.path is the module search path


def main():

    logging.getLogger().setLevel(logging.WARNING)  # essentially does print statements to help debug (WARNING)
    # logging explained https://appdividend.com/2019/06/08/python-logging-tutorial-with-example-logging-in-python/

    # read in job descriptions
    observations = pd.read_csv("../data/job_descriptions.csv")
    observations.columns = ['ReqID', 'text']
    
    observations.drop_duplicates(inplace=True)
    observations.reset_index(drop=True, inplace=True)
    observations.text.fillna('', inplace=True)
    observations.dropna(how='all', inplace=True) # drop rows that are all na
    observations['text'] = observations['text'].apply(lambda x: x.replace('Kraft Heinz is an EO employer', ''))
    observations['text'] = observations['text'].apply(lambda x: x.replace('Minorities/Women/Vets/Disabled and other protected categories', ''))
    
    observations = resume_sectioning.create_columns(observations) # add columns to match resume df
  
    observations = transform(observations)  # extract data from resume sections

    load(observations)  # save to csv to finish
    
    print("Finished.")
    pass


def extract():
    logging.info('Begin extract')

    candidate_file_agg = list()  # for creating list of resume file paths
    for root, subdirs, files in os.walk(lib.get_conf('resume_directory')):  # gets path to resumes from yaml file
        # os.walk(parentdir + '/data/input/example_resumes'): would do the same thing
        files = filter(lambda f: f.endswith(('.pdf', '.PDF')), files)  # only read pdfs
        folder_files = map(lambda x: os.path.join(root, x), files)
        candidate_file_agg.extend(folder_files)

    observations = pd.DataFrame(data=candidate_file_agg, columns=['file_path'])  # convert to df
    logging.info('Found {} candidate files'.format(len(observations.index)))
    observations['extension'] = observations['file_path'].apply(lambda x: os.path.splitext(x)[1])  # e.g. pdf or doc
    observations = observations[observations['extension'].isin(lib.AVAILABLE_EXTENSIONS)]
    logging.info('Subset candidate files to extensions w/ available parsers. {} files remain'.
                 format(len(observations.index)))
    observations['text'] = observations['file_path'].apply(lib.convert_pdf)  # get text from .pdf files

    logging.info('End extract')
    return observations


def transform(observations):
    logging.info('Begin transform')
    
    print("Extracting years of experience wanted")
    observations = observations.fillna('')
    observations['email'] = ''
    observations['phone'] = ''
    observations['GPA'] = ''
    observations['years_experience'] = observations['text'].apply(lambda x: field_extraction.years_of_experience(x))
    observations['mos_experience'] = field_extraction.months_of_experience(observations['years_experience'])

    observations = field_extraction.extract_fields(observations)  # search for terms in whole resume

    logging.info('End transform')
    return observations


def load(observations):
    logging.info('Begin load')
    output_path = os.path.join(lib.get_conf('summary_output_directory'), 'job_description_summary_FULL.csv')

    logging.info('Results being output to {}'.format(output_path))
    print('Results output to {}'.format(output_path))
    
    observations.to_csv(path_or_buf=output_path, index=False, encoding='utf-8')
    logging.info('End transform')
    pass


if __name__ == '__main__':
    main()
