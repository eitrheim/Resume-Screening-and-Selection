# coding: utf-8

import logging
import os
import re

import pandas as pd
from ruamel.yaml import YAML

import pdf2text
import pdf2textNEWER

CONFS = None


def load_confs(confs_path='/Users/anneitrheim/PycharmProjects/Resume-Screening-and-Selection/Resume-Parser-master-new/confs/config.yaml'):
    """
    Load the .yaml file
    """
    global CONFS

    if CONFS is None:
        yaml = YAML()
        yaml.preserve_quotes = True
        yaml.indent(mapping=4, sequence=4, offset=2)
        try:
            CONFS = yaml.load(open(confs_path))
        except IOError:
            confs_template_path = confs_path
            logging.warn(
                'Confs path: {} does not exist. Attempting to load confs template, '
                'from path: {}'.format(confs_path, confs_template_path))
            CONFS = yaml.load(open(confs_template_path))
    return CONFS


def get_conf(conf_name):
    return load_confs()[conf_name]


def archive_dataset_schemas(step_name, local_dict, global_dict):
    """
    Archive the schema for all available Pandas DataFrames
     - Determine which objects in namespace are Pandas DataFrames
     - Pull schema for all available Pandas DataFrames
     - Write schemas to file
    """
    logging.info('Archiving data set schema(s) for step name: {}'.format(step_name))

    data_schema_dir = get_conf('data_schema_dir')  # get location (i.e. /data/schema)
    schema_output_path = os.path.join(data_schema_dir, step_name + '.csv')  # path to /data/schema/extract.csv
    schema_agg = list()

    env_variables = dict()
    env_variables.update(local_dict)
    env_variables.update(global_dict)

    data_sets = filter(lambda x: type(x[1]) == pd.DataFrame, env_variables.items())  # filter down to Pandas dfs
    data_sets = dict(data_sets)  # dictionary of the dfs in local & global environments

    for (data_set_name, data_set) in data_sets.items():
        logging.info('Working data_set: {}'.format(data_set_name))  # extract variable names
        local_schema_df = pd.DataFrame(data_set.dtypes, columns=['type'])
        local_schema_df['data_set'] = data_set_name
        schema_agg.append(local_schema_df)

    agg_schema_df = pd.concat(schema_agg)  # aggregate schema list into one df
    agg_schema_df.to_csv(schema_output_path, index_label='variable')  # save as csv


def term_count(string_to_search, term):
    """
    A utility function which counts the number of times `term` occurs in `string_to_search`
    """
    try:
        regular_expression = re.compile(term, re.IGNORECASE)
        result = re.findall(regular_expression, string_to_search)
        return len(result)
    except Exception:
        logging.error('Error occurred during regex search: {}'.format(term))
        return 0


def term_count_case_sensitive(string_to_search, term):
    """
    A utility function which counts the number of times `term` occurs in `string_to_search`
    """
    try:
        regular_expression = re.compile(term)
        result = re.findall(regular_expression, string_to_search)
        return len(result)
    except Exception:
        logging.error('Error occurred during regex search: {}'.format(term))
        return 0


def term_match(string_to_search, term):
    """
    A utility function which return the first match to the `regex_pattern` in the `string_to_search`
    """
    try:
        regular_expression = re.compile(term, re.IGNORECASE)
        result = re.findall(regular_expression, string_to_search)
        if len(result) > 0:
            return result[0]
        else:
            return None
    except Exception:
        logging.error('Error occurred during regex search')
        return None


def convert_pdf(f):
    output_filename = os.path.basename(os.path.splitext(f)[0]) + '.txt'  # get file name (e.g. Ann Resume.pdf)
    output_filepath = os.path.join('/Users/anneitrheim/PycharmProjects/Resume-Screening-and-Selection/Resume-Parser-master-new/data/output', output_filename)  # creating the path for the output
    logging.info('Writing text from {} to {}'.format(f, output_filepath))
    pdf2text.main(args=[f, '--outfile', output_filepath])  # convert pdf to text & place in output .txt file
    try:
        pdf2textNEWER.main(argv=[f, '-o', output_filepath, f])
        print('Newer version of pdf2text used for ', output_filename)
    except:
        pass
        print('Older version of pdf2text used for ', output_filename)
    return open(output_filepath).read()  # return contents of intermediate output file
