import pandas as pd
import numpy as np

#parsed data
df = pd.read_csv('~/Resume-Parser-JOBS/data/output/job_description_summary_FULL.csv')
#df.to_csv('~/data/job_description_summary_v4.csv')

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
with open('/home/cdsw/Resume-Parser-JOBS/confs/config.yaml', 'r') as stream:
  yaml_file = yaml.safe_load(stream)

#certifications
hot.certifications.fillna('', inplace=True)
for item in yaml_file['case_agnostic_whole_resume']['certifications']:
  if type(item) == list:
    search_term = item[0].replace('\\x20','').replace(' ','')
    col_name = item[1].replace('\\x20','').replace(' ','')
  else:
    search_term = item.replace('\\x20','').replace(' ','')
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

path = '~/data/job_description_one_hot_FULL.csv'
hot.to_csv(path, index=False)
print('saved one hot encoding to: {}'.format(path))

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

#specific for types of roles
#looking at type of role to then one hot encode
jobs = pd.read_csv("~/data/full_requisition_data.csv")
jobs.drop(['Unnamed: 0','Job Description','Division','Job Category','Band', 'Candidate ID','Location','offer completed/accepted date', 'Date Requisition Opened','Job Requisition Status', 'Country'], axis=1, inplace=True)
jobs['Managerial Role'].fillna('No', inplace=True)
jobs['Managerial Role'] = jobs['Managerial Role'].apply(lambda x: 0 if x == 'No' else 1)
jobs['Function'] = jobs['Req Title'].apply(lambda x: x.split(' ')[-1])
jobs.columns = ['ReqID', 'ReqTitle', 'Func', 'MngrRole']

jobs = jobs.merge(hot, on='ReqID', how='inner')
jobs.Func.value_counts()

#for i in jobs.index:
#  if jobs.Func[i] == 'Sales':
#    rel_cols = hot.filter(like='sales').columns
#    for cols_to_one_hot in rel_cols:
#      jobs[cols_to_one_hot] = np.repeat(1,num_rows)
#    rel_cols = hot.filter(like='sell').columns
#    for cols_to_one_hot in rel_cols:
#      jobs[cols_to_one_hot] = np.repeat(1,num_rows)
#    rel_cols = hot.filter(like='relationship').columns
#    for cols_to_one_hot in rel_cols:
#      jobs[cols_to_one_hot] = np.repeat(1,num_rows)
#
#  elif jobs.Func[i] == 'Manufacturing':
#    rel_cols = hot.filter(like='manufactur').columns
#    for cols_to_one_hot in rel_cols:
#      jobs[cols_to_one_hot] = np.repeat(1,num_rows)
#    jobs['packaging'] = np.repeat(1,num_rows)
#
#  elif jobs.Func[i] == 'Finance':
#    rel_cols = hot.filter(like='financ').columns
#    for cols_to_one_hot in rel_cols:
#      jobs[cols_to_one_hot] = np.repeat(1,num_rows)
#    rel_cols = hot.filter(like='forecast').columns
#    for cols_to_one_hot in rel_cols:
#      jobs[cols_to_one_hot] = np.repeat(1,num_rows)
#    rel_cols = hot.filter(like='modeling').columns
#    for cols_to_one_hot in rel_cols:
#      jobs[cols_to_one_hot] = np.repeat(1,num_rows)
#    rel_cols = hot.filter(like='budget').columns
#    for cols_to_one_hot in rel_cols:
#      jobs[cols_to_one_hot] = np.repeat(1,num_rows)
#    rel_cols = hot.filter(like='cost').columns
#    for cols_to_one_hot in rel_cols:
#      jobs[cols_to_one_hot] = np.repeat(1,num_rows)
#    jobs['analytical'] = np.repeat(1,num_rows)
#
#  elif jobs.Func[i] == 'Logistics':
#    rel_cols = hot.filter(like='logistic').columns
#    for cols_to_one_hot in rel_cols:
#      jobs[cols_to_one_hot] = np.repeat(1,num_rows)
#    rel_cols = hot.filter(like='inventory').columns
#    for cols_to_one_hot in rel_cols:
#      jobs[cols_to_one_hot] = np.repeat(1,num_rows)
#    rel_cols = hot.filter(like='chaing').columns
#    for cols_to_one_hot in rel_cols:
#      jobs[cols_to_one_hot] = np.repeat(1,num_rows)
#    rel_cols = hot.filter(like='suppl').columns
#    for cols_to_one_hot in rel_cols:
#      jobs[cols_to_one_hot] = np.repeat(1,num_rows)
#
#  elif jobs.Func[i] == 'R&D':
#    jobs['r&d'] = np.repeat(1,num_rows)
#    jobs['productdevelopment'] = np.repeat(1,num_rows)
#    rel_cols = hot.filter(like='research').columns
#    for cols_to_one_hot in rel_cols:
#      jobs[cols_to_one_hot] = np.repeat(1,num_rows)
#
#  elif jobs.Func[i] == 'Marketing':
#    rel_cols = hot.filter(like='marketing').columns
#    for cols_to_one_hot in rel_cols:
#      jobs[cols_to_one_hot] = np.repeat(1,num_rows)
#    rel_cols = hot.filter(like='advertising').columns
#    for cols_to_one_hot in rel_cols:
#      jobs[cols_to_one_hot] = np.repeat(1,num_rows)
#    rel_cols = hot.filter(like='brand').columns
#    for cols_to_one_hot in rel_cols:
#      jobs[cols_to_one_hot] = np.repeat(1,num_rows)
#    rel_cols = hot.filter(like='trend').columns
#    for cols_to_one_hot in rel_cols:
#      jobs[cols_to_one_hot] = np.repeat(1,num_rows)
#    rel_cols = hot.filter(like='innovation').columns
#    for cols_to_one_hot in rel_cols:
#      jobs[cols_to_one_hot] = np.repeat(1,num_rows)
#
#  elif jobs.Func[i] == 'HR':
#    rel_cols = hot.filter(like='human').columns
#    for cols_to_one_hot in rel_cols:
#      jobs[cols_to_one_hot] = np.repeat(1,num_rows)
#    rel_cols = hot.filter(like='hiring').columns
#    for cols_to_one_hot in rel_cols:
#      jobs[cols_to_one_hot] = np.repeat(1,num_rows)
#    rel_cols = hot.filter(like='recruit').columns
#    for cols_to_one_hot in rel_cols:
#      jobs[cols_to_one_hot] = np.repeat(1,num_rows)
#    rel_cols = hot.filter(like='staff').columns
#    for cols_to_one_hot in rel_cols:
#      jobs[cols_to_one_hot] = np.repeat(1,num_rows)
#    rel_cols = hot.filter(like='talent').columns
#    for cols_to_one_hot in rel_cols:
#      jobs[cols_to_one_hot] = np.repeat(1,num_rows)
#    rel_cols = hot.filter(like='workforce').columns
#    for cols_to_one_hot in rel_cols:
#      jobs[cols_to_one_hot] = np.repeat(1,num_rows)
#
#  elif jobs.Func[i] == 'Quality':
#    rel_cols = hot.filter(like='quality').columns
#    for cols_to_one_hot in rel_cols:
#      jobs[cols_to_one_hot] = np.repeat(1,num_rows)
#    rel_cols = hot.filter(like='usda').columns
#    for cols_to_one_hot in rel_cols:
#      jobs[cols_to_one_hot] = np.repeat(1,num_rows)
#    rel_cols = hot.filter(like='sanitation').columns
#    for cols_to_one_hot in rel_cols:
#      jobs[cols_to_one_hot] = np.repeat(1,num_rows)
#    rel_cols = hot.filter(like='monitor').columns
#    for cols_to_one_hot in rel_cols:
#      jobs[cols_to_one_hot] = np.repeat(1,num_rows)
#    rel_cols = hot.filter(like='safe').columns
#    for cols_to_one_hot in rel_cols:
#      jobs[cols_to_one_hot] = np.repeat(1,num_rows)
#
#  elif jobs.Func[i] == 'Procurement':
#    rel_cols = hot.filter(like='procur').columns
#    for cols_to_one_hot in rel_cols:
#      jobs[cols_to_one_hot] = np.repeat(1,num_rows)
#    rel_cols = hot.filter(like='sourcing').columns
#    for cols_to_one_hot in rel_cols:
#      jobs[cols_to_one_hot] = np.repeat(1,num_rows)
#    rel_cols = hot.filter(like='vendor').columns
#    for cols_to_one_hot in rel_cols:
#      jobs[cols_to_one_hot] = np.repeat(1,num_rows)
#    rel_cols = hot.filter(like='contract').columns
#    for cols_to_one_hot in rel_cols:
#      jobs[cols_to_one_hot] = np.repeat(1,num_rows)
#    rel_cols = hot.filter(like='influencing').columns
#    for cols_to_one_hot in rel_cols:
#      jobs[cols_to_one_hot] = np.repeat(1,num_rows)
#    rel_cols = hot.filter(like='negot').columns
#    for cols_to_one_hot in rel_cols:
#      jobs[cols_to_one_hot] = np.repeat(1,num_rows)
#
#  elif jobs.Func[i] == 'IT':
#    rel_cols = hot.filter(like='technology').filter(regex=r'^((?!Bio).)*$').columns # does not include bio
#    for cols_to_one_hot in rel_cols:
#      jobs[cols_to_one_hot] = np.repeat(1,num_rows)
#    rel_cols = hot.filter(like='implement').columns
#    for cols_to_one_hot in rel_cols:
#      jobs[cols_to_one_hot] = np.repeat(1,num_rows)
#    rel_cols = hot.filter(like='deploy').columns
#    for cols_to_one_hot in rel_cols:
#      jobs[cols_to_one_hot] = np.repeat(1,num_rows)
#    rel_cols = hot.filter(like='technic').columns
#    for cols_to_one_hot in rel_cols:
#      jobs[cols_to_one_hot] = np.repeat(1,num_rows)
#
#  elif jobs.Func[i] == 'GBS':
#    pass  
#
#  elif jobs.Func[i] == 'Administration':
#    rel_cols = hot.filter(like='admin').columns
#    for cols_to_one_hot in rel_cols:
#      jobs[cols_to_one_hot] = np.repeat(1,num_rows)
#    rel_cols = hot.filter(like='organize').columns
#    for cols_to_one_hot in rel_cols:
#      jobs[cols_to_one_hot] = np.repeat(1,num_rows)
#    rel_cols = hot.filter(like='schedul').columns
#    for cols_to_one_hot in rel_cols:
#      jobs[cols_to_one_hot] = np.repeat(1,num_rows)
#    rel_cols = hot.filter(like='fastpaced').columns
#    for cols_to_one_hot in rel_cols:
#      jobs[cols_to_one_hot] = np.repeat(1,num_rows)
#    rel_cols = hot.filter(like='detail').columns
#    for cols_to_one_hot in rel_cols:
#      jobs[cols_to_one_hot] = np.repeat(1,num_rows)
#    jobs['officeadministration'] = np.repeat(1,num_rows)
#    jobs['administrative'] = np.repeat(1,num_rows)
#    
#  elif jobs.Func[i] == 'Regulatory':
#    rel_cols = hot.filter(like='regulatory').columns
#    for cols_to_one_hot in rel_cols:
#      jobs[cols_to_one_hot] = np.repeat(1,num_rows)
#    jobs['writing'] = np.repeat(1,num_rows)
#    jobs['writtencommunication'] = np.repeat(1,num_rows)
#    jobs['socialresponsibility'] = np.repeat(1,num_rows)
#    jobs['fundraising'] = np.repeat(1,num_rows)
#    rel_cols = hot.filter(like='interpersonal').columns
#    for cols_to_one_hot in rel_cols:
#      jobs[cols_to_one_hot] = np.repeat(1,num_rows)
#    rel_cols = hot.filter(like='present').columns
#    for cols_to_one_hot in rel_cols:
#      jobs[cols_to_one_hot] = np.repeat(1,num_rows)
#    rel_cols = hot.filter(like='support').columns
#    for cols_to_one_hot in rel_cols:
#      jobs[cols_to_one_hot] = np.repeat(1,num_rows)
#
#  elif jobs.Func[i] == 'Legal':
#    rel_cols = hot.filter(like='law').columns
#    for cols_to_one_hot in rel_cols:
#      jobs[cols_to_one_hot] = np.repeat(1,num_rows)
#    rel_cols = hot.filter(like='governance').columns
#    for cols_to_one_hot in rel_cols:
#      jobs[cols_to_one_hot] = np.repeat(1,num_rows)
#    rel_cols = hot.filter(like='rules').columns
#    for cols_to_one_hot in rel_cols:
#      jobs[cols_to_one_hot] = np.repeat(1,num_rows)
#    rel_cols = hot.filter(like='regulat').columns
#    for cols_to_one_hot in rel_cols:
#      jobs[cols_to_one_hot] = np.repeat(1,num_rows)
#    rel_cols = hot.filter(like='complianc').columns
#    for cols_to_one_hot in rel_cols:
#      jobs[cols_to_one_hot] = np.repeat(1,num_rows)
#
#  else:
#    pass
#      
#for i in jobs.index:
#  if jobs.MngrRole[i] == 1: # is a manager role
#    rel_cols = ['effectivelymanage','managing','managingdifficultconversations','managingdifficultpeople','managingteam','managingvirtualteams']
#    for cols_to_one_hot in rel_cols:
#      jobs[cols_to_one_hot] = np.repeat(1,num_rows)

jobs.drop(['ReqTitle', 'Func', 'MngrRole'],axis=1,inplace=True)

path = '~/data/job_description_one_hot_ideal_FULL.csv'
jobs.to_csv(path, index=False)
print('saved ideal one hot encoding to: {}'.format(path))

