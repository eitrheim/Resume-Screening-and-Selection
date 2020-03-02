import pandas as pd
import numpy as np
import re

def onehot():
    # read in parsed data
    df = pd.read_csv("/Users/anneitrheim/PycharmProjects/Resume-Screening-and-Selection/Resume-Parser-master-new/data/output/resume_summary.csv", error_bad_lines=False)
    df = df[df.CanID == df.CanID]
    df = df[df.ReqID == df.ReqID]
    df.reset_index(drop=True, inplace=True)


    ################################################################
    #ONE HOT ENCODING
    hot = df

    #honor_societies
    df.honor_societies.fillna('', inplace=True)
    hot['HonorSociety'] = df.honor_societies.apply(lambda x: 1 if len(x) > 2 else 0)

    #latin_honors
    df.latin_honors.fillna('', inplace=True)
    hot['LatinHonors'] = df.latin_honors.apply(lambda x: 1 if len(x) > 2 else 0)

    #scholarships_awards
    df.scholarships_awards.fillna('', inplace=True)
    hot['ScholarshipsAward'] = df.scholarships_awards.apply(lambda x: 1 if len(x) > 2 else 0)

    #schools
    df.community_college.fillna('', inplace=True)
    hot['CommCollege'] = df.community_college.apply(lambda x: 1 if len(x) > 2 else 0)
    df.other_universities.fillna('', inplace=True)
    hot['OtherUni'] = df.other_universities.apply(lambda x: 1 if len(x) > 2 else 0)
    df.top_100_universities.fillna('', inplace=True)
    hot['Top100Uni'] = df.top_100_universities.apply(lambda x: 1 if len(x) > 2 else 0)
    df.top_10_universities.fillna('', inplace=True)
    hot['Top10Uni'] = df.top_10_universities.apply(lambda x: 1 if len(x) > 2 else 0)

    #degrees
    df.associate_education_level.fillna('', inplace=True)
    hot['Associates'] = df.associate_education_level.apply(lambda x: 1 if len(x) > 2 else 0)
    df.bachelor_education_level.fillna('', inplace=True)
    hot['Bachelors'] = df.bachelor_education_level.apply(lambda x: 1 if len(x) > 2 else 0)
    df.master_education_level.fillna('', inplace=True)
    hot['Masters'] = df.master_education_level.apply(lambda x: 1 if len(x) > 2 else 0)
    df.doctor_education_level.fillna('', inplace=True)
    hot['Doctors'] = df.doctor_education_level.apply(lambda x: 1 if len(x) > 2 else 0)

    #companies
    df.company_foodbev.fillna('', inplace=True)
    hot['FoodBev'] = df.company_foodbev.apply(lambda x: 1 if len(x) > 2 else 0)
    df.company_consumer.fillna('', inplace=True)
    hot['Consumer'] = df.company_consumer.apply(lambda x: 1 if len(x) > 2 else 0)
    df.company_energychem.fillna('', inplace=True)
    hot['EnergyChem'] = df.company_energychem.apply(lambda x: 1 if len(x) > 2 else 0)
    df.company_fin.fillna('', inplace=True)
    hot['Fin'] = df.company_fin.apply(lambda x: 1 if len(x) > 2 else 0)
    df.company_health.fillna('', inplace=True)
    hot['HealthMed'] = df.company_health.apply(lambda x: 1 if len(x) > 2 else 0)
    df.company_industrial.fillna('', inplace=True)
    hot['Industrial'] = df.company_industrial.apply(lambda x: 1 if len(x) > 2 else 0)
    df.company_tech.fillna('', inplace=True)
    hot['Tech'] = df.company_tech.apply(lambda x: 1 if len(x) > 2 else 0)
    df.company_services.fillna('', inplace=True)
    hot['Services'] = df.company_services.apply(lambda x: 1 if len(x) > 2 else 0)
    df.company_other.fillna('', inplace=True)
    hot['OtherCo'] = df.company_other.apply(lambda x: 1 if len(x) > 2 else 0)

    ################################################################
    #ONE HOT ENCODING - EXPLODING COLUMNS

    import yaml
    with open('/Users/anneitrheim/PycharmProjects/Resume-Screening-and-Selection/Resume-Parser-master-new/confs/config.yaml', 'r') as stream:
      yaml_file = yaml.safe_load(stream)

    #certifications
    df.certifications.fillna('', inplace=True)
    for item in yaml_file['case_agnostic_whole_resume']['certifications']:
      if type(item) == list:
        search_term = item[0].replace('\\x20','').replace(' ', '')
        col_name = item[1].replace('\\x20','').replace(' ', '')
      else:
        search_term = item.replace('\\x20','').replace(' ', '')
        col_name = search_term
      hot[col_name] = df.certifications.apply(lambda x: 1 if x.find(search_term) >= 0 else 0)

    #major_minor
    df.major_minor.fillna('', inplace=True)
    for item in yaml_file['case_agnostic_education']['major_minor']:
      if type(item) == list:
        search_term = item[0].replace('\\x20','').replace(' ','')
        col_name = item[0].replace('\\x20','').replace(' ','')
      else:
        search_term = item.replace('\\x20','').replace(' ','')
        col_name = search_term
      hot[col_name] = df.major_minor.apply(lambda x: 1 if x.find(search_term) >= 0 else 0)

    #languages
    df.languages.fillna('', inplace=True)
    for item in yaml_file['case_agnostic_languages']['languages']:
      if type(item) == list:
        search_term = item[0].replace('\\x20','').replace(' ','')
        col_name = item[0].replace('\\x20','').replace(' ','')
      else:
        search_term = item.replace('\\x20','').replace(' ','')
        col_name = search_term
      hot[col_name] = df.languages.apply(lambda x: 1 if x.find(search_term) >= 0 else 0)

    #technical_skills
    df.technical_skills.fillna('', inplace=True)
    for item in yaml_file['case_agnostic_skill']['technical_skills']:
      if type(item) == list:
        search_term = item[0].replace('\\x20','').replace(' ','')
        col_name = item[0].replace('\\x20','').replace(' ','')
      else:
        search_term = item.replace('\\x20','').replace(' ','')
        col_name = search_term
      hot[col_name] = df.technical_skills.apply(lambda x: 1 if x.find(search_term) >= 0 else 0)

    #stem major column
    stem_majors = ['Accounting', 'Aeronautics', 'AirwayScience',
     'Analytics', 'AnimalScience', 'ArtificialIntelligence',
     'AutomotiveEngineering', 'Aviation', 'BakeryScience', 'Biology',
     'BiomedicalEngineering', 'Biophysics', 'BioresourceScience',
     'BusinessAnalytics', 'ChemicalEngineering', 'CivilEngineering',
     'CommunicationTechnology', 'ComputerEngineering', 'ComputerNetworks&Systems',
     'ComputerScience', 'ConstructionEngineering', 'CriminalScience', 'DataManagementTechnology',
     'DataScience', 'DieselTechnology', 'DraftingTechnology', 'Ecology',
     'Economics', 'ElectricalEngineering', 'EnvironmentalDesign', 'EnvironmentalEngineering',
     'EnvironmentalStudies', 'Finance', 'Food&Nutrition', 'Forensics',
     'Genetics', 'Genomics', 'GlobalHealth', 'Horticulture',
     'Immunology', 'IndustrialProductionTechnologies', 'Informatics', 'InformationScience',
     'InformationTechnology', 'Kinesiology', 'Logistics', 'MarineScience',
     'MaterialsScience', 'MathematicsEducation', 'Mathematics',
     'MechanicalEngineering', 'MedicalImagingScience', 'MedicalTechnology',
     'Meteorology', 'MilitaryTechnologies', 'Neuroscience', 'Nursing',
     'Oceanography', 'OsteopathicMedicine', 'PackagingEngineering',
     'Pathology', 'Pharmacy', 'PhysicalScience', 'Physics',
     'QuantumComputing', 'QuantumMechanics', 'Radiology',
     'SystemsEngineering', 'VeterinaryScience', 'Viticulture',
     'WebDevelopment', 'Zoology', 'Anatomy', 'Biochemistry',
     'Bioengineering', 'BiologicalEngineering', 'Biotechnology', 'CeramicEngineering',
     'Chemistry', 'Chiropractic', 'CognitiveScience', 'CyberSecurity',
     'Dentistry', 'EducationalTechnology', 'Endocrinology',
     'EngineeringScience', 'Epidemiology', 'Floriculture',
     'IndustrialEngineering', 'Medicine', 'MetallurgicalEngineering',
     'MineralEngineering', 'MolecularEngineering', 'NuclearEngineering',
     'Optometry', 'Paleontology', 'PetroleumEngineering',
     'Pharmacogenomics', 'Phlebotomy',
     'PhysicianAssistant', 'RespiratoryTherapy', 'SafetyTechnologies', 'ScienceEducation',
     'SoftwareEngineering', 'SoundEngineering', 'Statistics',
     'TechnologyEducation', 'Thermodynamics', 'Toxicology', 'Transportation']

    hot['StemMajor'] = hot[stem_majors].sum(axis=1)
    # hot['StemMajor'].sum()

    hot.to_csv("/Users/anneitrheim/PycharmProjects/Resume-Screening-and-Selection/Resume-Parser-master-new/data/output/resume_summary_one_hot.csv", index=False)
