import pandas as pd
import numpy as np

def onehot(root_file_path):
    #parsed data
    df = pd.read_csv(root_file_path + 'Resume-Parser-JOBS/data/output/job_description_summary_FULL.csv')

    ################################################################
    #ONE HOT ENCODING
    hot = df

    #honor_societies
    hot.honor_societies.fillna('', inplace=True)
    hot['HonorSociety'] = hot.honor_societies.apply(lambda x: 1 if len(x) > 2 else 0)

    #latin_honors
    hot.latin_honors.fillna('', inplace=True)
    hot['LatinHonors'] = hot.latin_honors.apply(lambda x: 1 if len(x) > 2 else 0)

    #scholarships_awards
    hot.scholarships_awards.fillna('', inplace=True)
    hot['ScholarshipsAward'] = hot.scholarships_awards.apply(lambda x: 1 if len(x) > 2 else 0)

    #schools
    hot.community_college.fillna('', inplace=True)
    hot['CommCollege'] = hot.community_college.apply(lambda x: 1 if len(x) > 2 else 0)
    hot.other_universities.fillna('', inplace=True)
    hot['OtherUni'] = hot.other_universities.apply(lambda x: 1 if len(x) > 2 else 0)
    hot.top_100_universities.fillna('', inplace=True)
    hot['Top100Uni'] = hot.top_100_universities.apply(lambda x: 1 if len(x) > 2 else 0)
    hot.top_10_universities.fillna('', inplace=True)
    hot['Top10Uni'] = hot.top_10_universities.apply(lambda x: 1 if len(x) > 2 else 0)

    #degrees
    hot.associate_education_level.fillna('', inplace=True)
    hot['Associates'] = hot.associate_education_level.apply(lambda x: 1 if len(x) > 2 else 0)
    hot.bachelor_education_level.fillna('', inplace=True)
    hot['Bachelors'] = hot.bachelor_education_level.apply(lambda x: 1 if len(x) > 2 else 0)
    hot.master_education_level.fillna('', inplace=True)
    hot['Masters'] = hot.master_education_level.apply(lambda x: 1 if len(x) > 2 else 0)
    hot.doctor_education_level.fillna('', inplace=True)
    hot['Doctors'] = hot.doctor_education_level.apply(lambda x: 1 if len(x) > 2 else 0)

    #companies
    hot.company_foodbev.fillna('', inplace=True)
    hot['FoodBev'] = hot.company_foodbev.apply(lambda x: 1 if len(x) > 2 else 0)
    hot.company_consumer.fillna('', inplace=True)
    hot['Consumer'] = hot.company_consumer.apply(lambda x: 1 if len(x) > 2 else 0)
    hot.company_energychem.fillna('', inplace=True)
    hot['EnergyChem'] = hot.company_energychem.apply(lambda x: 1 if len(x) > 2 else 0)
    hot.company_fin.fillna('', inplace=True)
    hot['Fin'] = hot.company_fin.apply(lambda x: 1 if len(x) > 2 else 0)
    hot.company_health.fillna('', inplace=True)
    hot['HealthMed'] = hot.company_health.apply(lambda x: 1 if len(x) > 2 else 0)
    hot.company_industrial.fillna('', inplace=True)
    hot['Industrial'] = hot.company_industrial.apply(lambda x: 1 if len(x) > 2 else 0)
    hot.company_tech.fillna('', inplace=True)
    hot['Tech'] = hot.company_tech.apply(lambda x: 1 if len(x) > 2 else 0)
    hot.company_services.fillna('', inplace=True)
    hot['Services'] = hot.company_services.apply(lambda x: 1 if len(x) > 2 else 0)
    hot.company_other.fillna('', inplace=True)
    hot['OtherCo'] = hot.company_other.apply(lambda x: 1 if len(x) > 2 else 0)

    ################################################################
    #ONE HOT ENCODING - EXPLODING COLUMNS

    import yaml
    with open(root_file_path + 'Resume-Parser-JOBS/confs/config.yaml', 'r') as stream:
      yaml_file = yaml.safe_load(stream)

    #certifications
    hot.certifications.fillna('', inplace=True)
    for item in yaml_file['case_agnostic_whole_resume']['certifications']:
      if type(item) == list:
        search_term = item[0].replace('\\x20', '').replace(' ', '')
        col_name = item[1].replace('\\x20', '').replace(' ', '')
      else:
        search_term = item.replace('\\x20', '').replace(' ', '')
        col_name = search_term
      hot[col_name] = hot.certifications.apply(lambda x: 1 if x.find(search_term) >= 0 else 0)

    #major_minor
    hot.major_minor.fillna('', inplace=True)
    for item in yaml_file['case_agnostic_education']['major_minor']:
      if type(item) == list:
        search_term = item[0].replace('\\x20','').replace(' ','')
        col_name = item[0].replace('\\x20','').replace(' ','')
      else:
        search_term = item.replace('\\x20','').replace(' ','')
        col_name = search_term
      hot[col_name] = hot.major_minor.apply(lambda x: 1 if x.find(search_term) >= 0 else 0)

    #languages
    hot.languages.fillna('', inplace=True)
    for item in yaml_file['case_agnostic_languages']['languages']:
      if type(item) == list:
        search_term = item[0].replace('\\x20','').replace(' ','')
        col_name = item[0].replace('\\x20','').replace(' ','')
      else:
        search_term = item.replace('\\x20','').replace(' ','')
        col_name = search_term
      hot[col_name] = hot.languages.apply(lambda x: 1 if x.find(search_term) >= 0 else 0)

    #technical_skills
    hot.technical_skills.fillna('', inplace=True)
    for item in yaml_file['case_agnostic_skill']['technical_skills']:
      if type(item) == list:
        search_term = item[0].replace('\\x20','').replace(' ','')
        col_name = item[0].replace('\\x20','').replace(' ','')
      else:
        search_term = item.replace('\\x20','').replace(' ','')
        col_name = search_term
      hot[col_name] = hot.technical_skills.apply(lambda x: 1 if x.find(search_term) >= 0 else 0)

    #stem major column
    def needs_stem(text_string):
      try:
        x = text_string.find('STEM background')
        if x > 0:
          return 1
        else:
          return 0
      except:
        return 0
    hot['StemMajor'] = hot.text.apply(lambda x: needs_stem(x))

    stem_majors = ['StemMajor', 'Accounting', 'Aeronautics', 'AirwayScience',
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
    hot['StemMajor'] = hot['StemMajor'].apply(lambda x: 1 if x > 0 else 0)

    ################################################################
    #ADDING IN ONES FOR IDEAL CANDIDATE

    num_rows = len(hot)

    hot['HonorSociety'] = np.repeat(1,num_rows)
    hot['LatinHonors'] = np.repeat(1,num_rows)
    hot['ScholarshipsAward'] = np.repeat(1,num_rows)
    hot['CommCollege'] = np.repeat(1,num_rows)       # or 0?
    hot['OtherUni'] = np.repeat(1,num_rows)          # or 0?
    hot['Top100Uni'] = np.repeat(1,num_rows)
    hot['Top10Uni'] = np.repeat(1,num_rows)
    hot['GPAmax'] = np.repeat(4.0,num_rows)

    hot.to_csv(root_file_path + 'Resume-Parser-JOBS/data/job_description_one_hot_ideal_FULL.csv', index=False)
