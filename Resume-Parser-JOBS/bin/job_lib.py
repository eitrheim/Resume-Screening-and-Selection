import logging
import re

import pandas as pd
from ruamel.yaml import YAML


CONFS = None

def load_confs(confs_path='/Users/anneitrheim/PycharmProjects/Resume-Screening-and-Selection/Resume-Parser-JOBS/confs/config.yaml'):
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

