import pandas as pd

#parsed data
x = pd.read_csv("~/Resume-Parser-master-new/data/output/resume_summary.csv", error_bad_lines=False)
len(x)
x = x[x.CanID == x.CanID]
len(x)
x = x[x.ReqID == x.ReqID]
len(x)
x.reset_index(drop=True, inplace=True)

drop_rows = []
for i in x.index:
  if len(x.ReqID.iloc[i]) > 8:
    drop_rows.append(i)
for i in x.index:
  if len(x.ReqID.iloc[i]) < 6:
    drop_rows.append(i)
x.drop(drop_rows, inplace=True, axis=0)
x.reset_index(drop=True, inplace=True)
len(x)

drop_rows = []
for i in x.index:
  if len(x.CanID.iloc[i]) > 8:
    drop_rows.append(i)
for i in x.index:
  if len(x.CanID.iloc[i]) < 6:
    drop_rows.append(i)   
#len(drop_rows)
#drop_rows = list(set(drop_rows))
#len(drop_rows)
x.drop(drop_rows, inplace=True, axis=0)
x.reset_index(drop=True, inplace=True)
len(x)
len(x) - 118582, "more to go, or (# lost)"

#OG data
observations = pd.read_csv('~/data/Candidate Report.csv', usecols=[3,0,12], encoding = 'latin-1')
observations.columns = ['ReqID','CanID','text']
observations.drop_duplicates(inplace=True)
observations.reset_index(drop=True, inplace=True)
observations.text.fillna('', inplace=True)
observations.dropna(how='all', inplace=True)
len(observations) #118582

df = observations.merge(x, how="inner", on=['ReqID','CanID','text'])
len(df)
len(df) - 118582, "more to go, or (# lost)"

del x, observations

#convert GPA to a single number
import numpy as np
import re
GPA_REGEX = r"[01234]{1}\.[0-9]{1,3}"
df.GPA.fillna('[]', inplace=True)
df['GPAnum'] = df.GPA.apply(lambda x: re.findall(re.compile(GPA_REGEX), x))
def getmax(x):
  try:
    y = max(x)
  except:
    y = 0
  return(y)
df['GPAmax'] = df['GPAnum'].apply(lambda x: getmax(x))
df['GPAmax'] = df['GPAmax'].apply(lambda x: np.nan if x == 0 else x)
df.filter(like='GPA')
np.mean(df['GPAmax'].astype('float'))
df.drop('GPAnum', axis=1, inplace=True)

df.to_csv("~/data/resume_summary_v8.csv", index=False)

df = pd.read_csv("~/data/resume_summary_v8.csv", error_bad_lines=False)

##look how much is missing
#total = df.isnull().sum().sort_values(ascending=False)
#percent_1 = df.isnull().sum()/df.isnull().count()*100
#percent_2 = (round(percent_1,2)).sort_values(ascending=False)
#missing_data=pd.concat([total, percent_2], axis = 1, keys=["Total", "%"], sort=False)
#missing_data
#
#columns not useful right now
#drop_cols = ['GPA', 'GPAnum', 'GPAmax', 'courses', 'hobbies', 'email', 'phone','Education', 'Extracurriculars','Language', 'Work', 'Summaries', 'Skill', 'Member', 'Writing', 'Researching', 'Honor', 'Activity']
#df.drop(drop_cols,inplace=True,axis=1)
#
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
with open('Resume-Parser-master-new/confs/config.yaml', 'r') as stream:
  yaml_file = yaml.safe_load(stream)

#certifications
df.certifications.fillna('', inplace=True)
for item in yaml_file['case_agnostic_whole_resume']['certifications']:
  if type(item) == list:
    search_term = item[0].replace('\\x20','').replace(' ','')
    col_name = item[1].replace('\\x20','').replace(' ','')
  else:
    search_term = item.replace('\\x20','').replace(' ','')
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
hot['StemMajor'].sum()                           

hot.to_csv("~/data/resume_summary_one_hot.csv", index=False)

  
#FOR REFERENCE
len(hot.columns)
                            
empty_cols = []
for i in hot.columns[47:]:
  if sum(hot[i]) == 0:
    empty_cols.append(i)
empty_cols               
               