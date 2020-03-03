import logging
import re
import job_lib

EMAIL_REGEX = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,4}"
PHONE_REGEX = r"\(?(\d{3})?\)?[\s\.-]{0,2}?(\d{3})[\s\.-]{0,2}(\d{4})"
GPA_REGEX1 = r"GPA[ .:-]+[of ]{0,3}[01234]{1}\.[0-9]{1,3}"
GPA_REGEX2 = r"[01234]{1}\.[0-9]{1,3}[ .:-]+GPA"


def gpa_extractor(input_string):
    result = re.findall(re.compile(GPA_REGEX1), input_string.replace('\t', ' ').replace('\r', ' '))
    result += re.findall(re.compile(GPA_REGEX2), input_string.replace('\t', ' ').replace('\r', ' '))
    return result


def extract_fields(df):
    # note all commas and apostrophes are removed at this point from the extract_skills_case_ functions
    print("Extracting certifications")
    for extractor, items_of_interest in job_lib.get_conf('case_agnostic_whole_resume').items():
        # column name is title of the sections in the yaml file
        df[extractor] = df['text'].apply(lambda x: extract_skills_case_agnostic(x, items_of_interest))
 
    print("Extracting universities and majors/minors")
    for extractor, items_of_interest in job_lib.get_conf('case_agnostic_education').items():
        df[extractor] = df['text'].apply(lambda x: extract_skills_case_agnostic(str(x.encode('utf-8', 'replace')), items_of_interest)) #.replace(' & ', ' and ')

    print("Extracting level of education")
    for extractor, items_of_interest in job_lib.get_conf('case_sensitive_education').items():
        df[extractor] = df['text'].apply(lambda x: extract_skills_case_sensitive(x, items_of_interest))

#    print("Extracting coursework")
    for extractor, items_of_interest in job_lib.get_conf('case_agnostic_courses').items():
        df[extractor] = df['text'].apply(lambda x: extract_skills_case_agnostic(x, items_of_interest))

    print("Extracting languages spoken")
    for extractor, items_of_interest in job_lib.get_conf('case_agnostic_languages').items():
        df[extractor] = df['text'].apply(lambda x: extract_skills_case_agnostic(x, items_of_interest))

#    print("Extracting hobbies and interests")
    for extractor, items_of_interest in job_lib.get_conf('case_agnostic_hobbies').items():
        df[extractor] = df['text'].apply(lambda x: extract_skills_case_agnostic(x, items_of_interest))

    print("Extracting technical skills")
    for extractor, items_of_interest in job_lib.get_conf('case_agnostic_skill').items():
        df[extractor] = df['text'].apply(lambda x: extract_skills_case_agnostic(x.replace('.', ''), items_of_interest))

    print("Extracting preferred industries")
    for extractor, items_of_interest in job_lib.get_conf('case_agnostic_work').items():
        df[extractor] = df['text'].apply(lambda x: extract_skills_case_agnostic(x.replace('.', ''), items_of_interest))

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
            skill_matches += job_lib.term_count(resume_text.lower().replace(' and ', ' & ').replace('-', ' ').replace(':', '').replace(',', '').replace('\'', ''), skill_alias.lower())  # add the # of matches for each alias

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
            skill_matches += job_lib.term_count(resume_text.replace('-', ' ').replace(':', '').replace(',', '').replace('\'', ''), skill_alias.lower())  # add the # of matches for each alias

        if skill_matches > 0:
            matched_skills.add(skill_name.replace('\x20', ''))

    if len(matched_skills) == 0:
        matched_skills = ''

    return list(matched_skills)

  
def years_of_experience(input_string):
    yr_list = []
    #input_string = input_string.encode('utf-8', 'replace')
    input_string = input_string.lower()
    input_string = input_string.replace(')', ' - ')
    input_string = input_string.replace('.', '')
    input_string = input_string.replace('\n', ' ')
    input_string = input_string.replace('\t', ' ')
    input_string = input_string.replace('\r', ' ')
    input_string = input_string.replace('\'', ' ')
    input_string = re.sub('[ ]+', " ", input_string)

    YOE_REGEX = r"[0-9]{1,2}[-]{0,1}[0-9]{0,2}[+]{0,1}[ ]{0,1}year[s]{0,1}[\’]{0,1} [of|in]{0,2}.*?experience"
    years_experience = re.findall(re.compile(YOE_REGEX), input_string)
    
    if len(years_experience) == 0:
        YOE_REGEX = r"minimum [of ]{0,3}[0-9]{1,2}[-]{0,1}[0-9]{0,2}[+]{0,1}[ ]{0,1}year.*?"
        years_experience = re.findall(re.compile(YOE_REGEX), input_string)
        if len(years_experience) == 0:
            years_experience = 0
        else:
            try:
                years_experience = re.findall(re.compile(r"[0-9]{1,2}"), years_experience)
                years_experience = min(years_experience)
            except:
                for i in years_experience:
                    yr_list.append(re.findall(re.compile(r"[0-9]{1,2}"), i)[0])
                years_experience = min(yr_list)
    elif len(years_experience) == 1:
        years_experience = re.findall(re.compile(r"[0-9]{1,2}"), years_experience[0])[0]
        years_experience = min(years_experience)
    else:
        for i in years_experience:
            yr_list.append(re.findall(re.compile(r"[0-9]{1,2}"), i)[0])
        years_experience = min(yr_list)
    years_experience = int(years_experience)
    
    return years_experience
  

def months_of_experience(years_experience):
    
    try:
      mos_of_experience = years_experience * 12
    except:
      mos_of_experience = 0
    
    return mos_of_experience

