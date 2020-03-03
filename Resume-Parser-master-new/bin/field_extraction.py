# coding: utf-8

import logging
import re
import time
from datetime import datetime
from dateutil import relativedelta
from gensim.utils import simple_preprocess
import lib

EMAIL_REGEX = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,4}"
PHONE_REGEX = r"\(?(\d{3})?\)?[\s\.-]{0,2}?(\d{3})[\s\.-]{0,2}(\d{4})"
GPA_REGEX1 = r"GPA[ .:-]+[of ]{0,3}[01234]{1}\.[0-9]{1,3}"
GPA_REGEX2 = r"[01234]{1}\.[0-9]{1,3}[ .:-]+GPA"
months = ['january', 'jan', 'february', 'feb', 'march', 'mar', 'april', 'apr', 'may', 'june', 'jun', 'july', 'jul',
          'august', 'aug', 'september', 'sept', 'sep', 'october', 'oct', 'november', 'nov', 'december', 'dec']


def candidate_name_extractor(input_string, nlp):

    doc = nlp(input_string.replace('\n', '.'))

    doc_entities = doc.ents  # extract entities

    # Subset to person type entities
    doc_persons = filter(lambda x: x.label_ == 'PERSON', doc_entities)
    doc_persons = filter(lambda x: len(x.text.strip().split()) >= 2, doc_persons)
    doc_persons = map(lambda x: x.text.strip(), doc_persons)
    doc_persons = list(doc_persons)

    if len(doc_persons) > 0:  # assume that the first Person entity is the candidate's name
        return doc_persons[0]
    return "NOT FOUND"


def gpa_extractor(input_string):
    result = re.findall(re.compile(GPA_REGEX1), input_string.replace('\t', ' ').replace('\r', ' '))
    result += re.findall(re.compile(GPA_REGEX2), input_string.replace('\t', ' ').replace('\r', ' '))
    return result


def extract_fields(df):
    # note all commas and apostrophes are removed at this point from the extract_skills_case_ functions
    print("Extracting certifications, latin honors, honor societies, scholarships/awards")
    for extractor, items_of_interest in lib.get_conf('case_agnostic_whole_resume').items():
        # column name is title of the sections in the yaml file
        df[extractor] = df['text'].apply(lambda x: extract_skills_case_agnostic(x, items_of_interest))
    # drop cum laude if summa cum laude or magna cum laude are present
    x = df[df.latin_honors == df.latin_honors]  # so it doesn't look at nans
    for i in x.index:
        if 'summa cum laude' in x.latin_honors.loc[i]:
            df.latin_honors.loc[i].remove('cum laude')
        elif 'magna cum laude' in x.latin_honors.loc[i]:
            df.latin_honors.loc[i].remove('cum laude')
        else:
            pass
    x = df[df.honor_societies == df.honor_societies]  # so it doesn't look at nans
    for i in x.index:
        if 'Alpha Epsilon Delta' in x.honor_societies.loc[i]:
            df.honor_societies.loc[i].remove('Alpha Epsilon')
        if 'Alpha Epsilon Rho' in x.honor_societies.loc[i]:
            df.honor_societies.loc[i].remove('Alpha Epsilon')
        if 'Phi Alpha Epsilon' in x.honor_societies.loc[i]:
            df.honor_societies.loc[i].remove('Alpha Epsilon')
        if 'Omega Chi Epsilon' in x.honor_societies.loc[i]:
            df.honor_societies.loc[i].remove('Chi Epsilon')
        if 'Sigma Lambda Alpha' in x.honor_societies.loc[i]:
            df.honor_societies.loc[i].remove('Lambda Alpha')
        if 'Phi Lambda Sigma' in x.honor_societies.loc[i]:
            df.honor_societies.loc[i].remove('Lambda Sigma')
        if 'Delta Phi Alpha' in x.honor_societies.loc[i]:
            df.honor_societies.loc[i].remove('Phi Alpha')
        if 'Phi Alpha Epsilon' in x.honor_societies.loc[i]:
            df.honor_societies.loc[i].remove('Phi Alpha')
        if 'Phi Alpha Theta' in x.honor_societies.loc[i]:
            df.honor_societies.loc[i].remove('Phi Alpha')
        if 'Sigma Phi Alpha' in x.honor_societies.loc[i]:
            df.honor_societies.loc[i].remove('Phi Alpha')
        if 'Alpha Phi Sigma' in x.honor_societies.loc[i]:
            df.honor_societies.loc[i].remove('Phi Sigma')
        if 'Phi Sigma Iota' in x.honor_societies.loc[i]:
            df.honor_societies.loc[i].remove('Phi Sigma')
        if 'Phi Sigma Pi' in x.honor_societies.loc[i]:
            df.honor_societies.loc[i].remove('Phi Sigma')
        if 'Phi Sigma Tau' in x.honor_societies.loc[i]:
            df.honor_societies.loc[i].remove('Phi Sigma')
        if 'Pi Tau Sigma' in x.honor_societies.loc[i]:
            df.honor_societies.loc[i].remove('Tau Sigma')
        if 'Tau Sigma Delta' in x.honor_societies.loc[i]:
            df.honor_societies.loc[i].remove('Tau Sigma')

    print("Extracting universities and majors/minors")
    for extractor, items_of_interest in lib.get_conf('case_agnostic_education').items():
        df[extractor] = df['Edu'].apply(lambda x: extract_skills_case_agnostic(str(x.encode('utf-8', 'replace')), items_of_interest)) #.replace(' & ', ' and ')

    print("Extracting level of education")
    for extractor, items_of_interest in lib.get_conf('case_sensitive_education').items():
        df[extractor] = df['Edu'].apply(lambda x: extract_skills_case_sensitive(x, items_of_interest))

    print("Extracting coursework")
    for extractor, items_of_interest in lib.get_conf('case_agnostic_courses').items():
        df[extractor] = df['Course'].apply(lambda x: extract_skills_case_agnostic(x, items_of_interest))

    print("Extracting languages spoken")
    for extractor, items_of_interest in lib.get_conf('case_agnostic_languages').items():
        df[extractor] = df['Language'].apply(lambda x: extract_skills_case_agnostic(x, items_of_interest))

    print("Extracting hobbies and interests")
    for extractor, items_of_interest in lib.get_conf('case_agnostic_hobbies').items():
        df[extractor] = df['Hobby'].apply(lambda x: extract_skills_case_agnostic(x, items_of_interest))

    print("Extracting technical skills")
    for extractor, items_of_interest in lib.get_conf('case_agnostic_skill').items():
        df[extractor] = df['Skill'].apply(lambda x: extract_skills_case_agnostic(x.replace('.', ''), items_of_interest))

    print("Extracting companies worked at")
    for extractor, items_of_interest in lib.get_conf('case_agnostic_work').items():
        df[extractor] = df['Work'].apply(lambda x: extract_skills_case_agnostic(x.replace('.', ''), items_of_interest))

    return df


def extract_skills_case_agnostic(resume_text, items_of_interest):
    potential_skills_dict = dict()
    matched_skills = set()

    for skill_input in items_of_interest:
        # Format list of strings inputs
        if type(skill_input) is not str and len(skill_input) >= 1:
            potential_skills_dict[skill_input[0]] = skill_input
        # Format string inputs
        if type(skill_input) is str:
            potential_skills_dict[skill_input] = [skill_input]
        else:
            pass
            #logging.warning('Unknown skill listing type: {}. Please format as a string or a list of strings'.format(skill_input))

    for (skill_name, skill_alias_list) in potential_skills_dict.items():

        skill_matches = 0
        # iterate through each string in the list of equivalent words (i.e. a line in the yaml file)
        # TODO incorporate word2vec here?
        for skill_alias in skill_alias_list:
            skill_matches += lib.term_count(resume_text.lower().replace(' and ', ' & ').replace('-', ' ').replace(':', '').replace(',', '').replace('\'', ''), skill_alias.lower())  # add the # of matches for each alias

        if skill_matches > 0:  # if at least one alias is found, add skill name to set of skills
            matched_skills.add(skill_name.replace('\x20', ''))

    if len(matched_skills) == 0:  # so it doesn't save 'set()' in the csv when it's empty
        matched_skills = ''

    return list(matched_skills)


def extract_skills_case_sensitive(resume_text, items_of_interest):
    potential_skills_dict = dict()
    matched_skills = set()

    for skill_input in items_of_interest:
        if type(skill_input) is not str and len(skill_input) >= 1:
            potential_skills_dict[skill_input[0]] = skill_input
        elif type(skill_input) is str:
            potential_skills_dict[skill_input] = [skill_input]
        else:
            pass
            #logging.warning('Unknown skill listing type: {}.'.format(skill_input))

    for (skill_name, skill_alias_list) in potential_skills_dict.items():

        skill_matches = 0
        # TODO incorporate word2vec here?
        for skill_alias in skill_alias_list:
            skill_matches += lib.term_count(resume_text.replace('-', ' ').replace(':', '').replace(',', '').replace('\'', ''), skill_alias.lower())  # add the # of matches for each alias

        if skill_matches > 0:
            matched_skills.add(skill_name.replace('\x20', ''))

    if len(matched_skills) == 0:
        matched_skills = ''

    return list(matched_skills)

def years_of_experience(input_string):
    list_of_dates = []
    #input_string = input_string.encode('utf-8', 'replace')
    input_string = input_string.lower()
    input_string = input_string.replace(')', ' - ')
    # input_string = input_string.replace('\xe2', ' - ')
    input_string = input_string.replace('.', '')
    input_string = input_string.replace('\n', ' ')
    input_string = input_string.replace('.', '')
    input_string = input_string.replace('\t', ' ')
    input_string = input_string.replace('\r', ' ')
    input_string = input_string.replace('\'', ' ')
    input_string = re.sub('[ ]+', " ", input_string)

    tokens = filter(None, re.split(r'(\S+|\W+)', input_string))
    tokens = list(tokens)
    for i in range(len(tokens)):
        if tokens[i] in months:
            try:
                if int(tokens[i+2]) in range(1970, 2025):
                    if tokens[i+3] == ' - ':
                        if tokens[i+4] in ['current', 'present', 'today']:
                            list_of_dates.append(''.join(tokens[i:i+5]))
                        elif tokens[i+4] in months:
                            try:
                                if int(tokens[i+6]) in range(1970, 2025):
                                    list_of_dates.append(''.join(tokens[i:i+7]))
                                elif int(tokens[i+6]) in range(0, 100):
                                    list_of_dates.append(''.join(tokens[i:i+7]))
                                else:
                                    pass
                            except:
                                pass
                        else:
                            pass
                    elif tokens[i+4] == 'to':
                        if tokens[i+6] in ['current', 'present', 'today']:
                            list_of_dates.append(''.join(tokens[i:i+7]))
                        elif tokens[i+6] in months:
                            try:
                                if int(tokens[i+8]) in range(1970, 2025):
                                    list_of_dates.append(''.join(tokens[i:i+9]))
                                elif int(tokens[i+8]) in range(0, 100):
                                    list_of_dates.append(''.join(tokens[i:i+9]))
                                else:
                                    pass
                            except:
                                pass
                        else:
                            pass

                elif int(tokens[i+2]) in range(0, 100):
                    if tokens[i+3] == ' - ':
                        if tokens[i+4] in ['current', 'present', 'today']:
                            list_of_dates.append(''.join(tokens[i:i+5]))
                        elif tokens[i+4] in months:
                            try:
                                if int(tokens[i+6]) in range(1970, 2025):
                                    list_of_dates.append(''.join(tokens[i:i+7]))
                                elif int(tokens[i+6]) in range(0, 100):
                                    list_of_dates.append(''.join(tokens[i:i+7]))
                                else:
                                    pass
                            except:
                                pass
                        else:
                            pass
                    elif tokens[i+4] == 'to':
                        if tokens[i+6] in ['current', 'present', 'today']:
                            list_of_dates.append(''.join(tokens[i:i+7]))
                        elif tokens[i+6] in months:
                            try:
                                if int(tokens[i+8]) in range(1970, 2025):
                                    list_of_dates.append(''.join(tokens[i:i+9]))
                                elif int(tokens[i+8]) in range(0, 100):
                                    list_of_dates.append(''.join(tokens[i:i+9]))
                                else:
                                    pass
                            except:
                                pass
                        else:
                            pass

            except:
                pass
    return str(list_of_dates)


def months_of_experience(list_of_dates):
    yrs = list_of_dates.apply(lambda x: x.replace('january', 'jan').replace('february', 'feb').replace('march', 'mar')
                              .replace('april', 'apr').replace('june', 'jun').replace('july', 'jul').replace('august', 'aug')
                              .replace('september', 'sep').replace('sept', 'sep').replace('october', 'oct').
                              replace('november', 'nov').replace('december', 'dec').replace('to', '-'))

    today = str(datetime.today().strftime('%b %Y'))
    yrs = yrs.apply(lambda x: x.replace('current', today).replace('present', today).replace('today', today))
    yrs = yrs.apply(lambda x: re.sub("[\[\]]", '', x).split(', '))

    # this assumes they list jobs in reverse chronological order
    mos_of_experience = []
    for row in yrs:
        if row == ['']:
            experience = relativedelta.relativedelta(0, 0)
        else:      
            try:

              try:
                  a = datetime.strptime(row[0].split(' - ')[1][:-1], '%b %Y')
              except ValueError:
                  a = datetime.strptime(row[0].split(' - ')[1][:-1], '%b %y')
              try:
                  b = datetime.strptime(row[-1].split(' - ')[0][1:], '%b %Y')
              except ValueError:
                  b = datetime.strptime(row[-1].split(' - ')[0][1:], '%b %y')
              experience = relativedelta.relativedelta(a, b)
            except:
              experience = relativedelta.relativedelta(0, 0)
        experience = experience.months + experience.years * 12

        if len(row) < 2:
            z = relativedelta.relativedelta(0, 0)
        elif experience == relativedelta.relativedelta(0, 0):
            pass
        else:
            z = relativedelta.relativedelta(0, 0)
            for item in range(1, len(row)):
              try:
                try:
                    a = datetime.strptime(row[item].split(' - ')[1][:-1], '%b %Y')
                except ValueError:
                    a = datetime.strptime(row[item].split(' - ')[1][:-1], '%b %y')
                try:
                    b = datetime.strptime(row[item - 1].split(' - ')[0][1:], '%b %Y')
                except ValueError:
                    b = datetime.strptime(row[item - 1].split(' - ')[0][1:], '%b %y')
                if relativedelta.relativedelta(b, a).months + relativedelta.relativedelta(b, a).years * 12 > 0:
                    z = z + relativedelta.relativedelta(b, a)
              except:
                pass
        experience_gaps = z
        experience_gaps = experience_gaps.months + experience_gaps.years * 12

        if experience_gaps > experience:
            mos_of_experience.append(0)
        else:
            mos_of_experience.append(int(experience - experience_gaps))
    print('\n')
    return mos_of_experience
