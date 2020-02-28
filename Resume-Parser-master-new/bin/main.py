# coding: utf-8

from __future__ import absolute_import
import sys
sys.path.append('/home/cdsw/Resume-Parser-master-new/bin')

# importing user defined modules
import field_extraction
import lib
import resume_sectioning

import inspect
import logging
import os
import pandas as pd
import time
#import spacy
#import en_core_web_sm


# hide settingwithcopywarning
pd.options.mode.chained_assignment = None



currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  # get location of main.py
parentdir = os.path.dirname(currentdir)  # get parent directory of main.py (where repository is on local)
sys.path.insert(0, parentdir)  # sys.path is the module search path


def main():

    logging.getLogger().setLevel(logging.WARNING)  # essentially does print statements to help debug (WARNING)
    # logging explained https://appdividend.com/2019/06/08/python-logging-tutorial-with-example-logging-in-python/

#    observations = extract()  # get text from pdf resumes
#
    # read in Kraft's resume text
    observations = pd.read_csv('~/data/Candidate Report.csv', usecols=[3,0,12], encoding = 'latin-1')
    observations.columns = ['ReqID','CanID','text']
    observations.drop_duplicates(inplace=True)
    observations.reset_index(drop=True, inplace=True)
    observations.text.fillna('', inplace=True)
    observations.dropna(how='all', inplace=True) # drop rows that are all na
    
#    observations = observations.sample(100)
    
#    #ones michael wants
#    jobz = ['e3625ad', '39ee3f', '45de815','40a2c38','63146c6']
#    observations = observations[observations.ReqID.isin(jobz)]
          
    # to get the start (as an int) of resume sections
    observations = resume_sectioning.section_into_columns(observations)
    
#    # save it so we dont have to do it again
#    observations.to_csv('~/Resume-Parser-master-new/data/output/resume_int_pos.csv', index=False, encoding='utf-8')
#    print(len(observations))
#
#    # reading part above
#    observations = pd.read_csv('~/Resume-Parser-master-new/data/output/resume_int_pos.csv', encoding = 'utf-8')
#    observations.reset_index(inplace=True, drop=True)
#    # 107,326 resumes, 0:107325
    
    # get only words pertaining each sub-section
    observations = resume_sectioning.word_put_in_sections(observations)
  
    # to combine the sub-sections
    observations = resume_sectioning.combine_sections_preparse(observations)
    observations = observations[observations.text == observations.text]

#    observations.to_csv('~/Resume-Parser-master-new/data/output/resume_sections.csv', index=False, encoding='utf-8')
#    print("Saved resume_sections.csv")
#    
#    # reading part above
#    observations = pd.read_csv('~/Resume-Parser-master-new/data/output/resume_sections.csv')
#
#    print("Loading Spacy Corpus")
#    #nlp = spacy.load('en_core_web_sm')
#    nlp = en_core_web_sm.load()
#    print("Spacy Corpus Loaded \n")

    observations = transform(observations) #, nlp)  # extract data from resume sections

    # to combine the sub-sections one last time
    observations = resume_sectioning.combine_sections_postparse(observations)
    observations = observations[observations.text == observations.text]
    print(len(observations), "... should be 107,326")
    
    #load(observations)  # save to csv to finish
    
    print("Date time: ", time.strftime('%m-%d %H:%M:%S', time.gmtime()))
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

    # Archive schema and return
    lib.archive_dataset_schemas('extract', locals(), globals())  # saving the schema
    logging.info('End extract')
    return observations


def transform(observations):  #, nlp):
    logging.info('Begin transform')

    print("Date time: ", time.strftime('%m-%d %H:%M:%S', time.gmtime()))
    print("Extracting email, phone, GPA, and dates of work experience")
    observations = observations.fillna('')
    # observations['candidate_name'] = observations['text'].apply(lambda x: field_extraction.candidate_name_extractor(x, nlp))
    observations['email'] = observations['text'].apply(lambda x: lib.term_match(x, field_extraction.EMAIL_REGEX))
    observations['phone'] = observations['text'].apply(lambda x: lib.term_match(x, field_extraction.PHONE_REGEX))
    observations['GPA'] = observations['text'].apply(lambda x: field_extraction.gpa_extractor(x))
    observations['years_experience'] = observations['Work'].apply(lambda x: field_extraction.years_of_experience(x))
    observations['mos_experience'] = field_extraction.months_of_experience(observations['years_experience'])

    observations = field_extraction.extract_fields(observations)  # search for terms in whole resume

    # Archive schema and return
    lib.archive_dataset_schemas('transform', locals(), globals())
    logging.info('End transform')
    return observations


def load(observations):
    logging.info('Begin load')
    output_path = os.path.join(lib.get_conf('summary_output_directory'), 'resume_summary.csv')

    logging.info('Results being output to {}'.format(output_path))
    print('Results output to {}'.format(output_path))
    
    observations.to_csv(path_or_buf=output_path, index=False, encoding='utf-8')
    logging.info('End transform')
     
#    observations.to_csv('~/data/resumes_5jobs.csv', index=False, encoding='utf-8')  
    
    pass


if __name__ == '__main__':
    main()