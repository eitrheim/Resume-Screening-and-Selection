import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from spacy.lang.en.stop_words import STOP_WORDS
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from gensim.models import doc2vec
from collections import namedtuple
# hide settingwithcopywarning
pd.options.mode.chained_assignment = None

def rank(jobID, topX, root_file_path, all_resumes):

    # read structure + one hot encoded dfs
    job_dummies_ideal = pd.read_csv(root_file_path + "Resume-Parser-JOBS/data/job_description_one_hot_ideal_FULL.csv")
    resume_dummies = pd.read_csv(root_file_path + "Resume-Parser-master-new/data/output/resume_summary_one_hot.csv")

    # cols = ['ReqID', 'text', 'Language', 'Work', 'Summaries', 'Skill', 'Member',
    #    'Writing', 'Researching', 'Honor', 'Activity', 'Education',
    #    'Extracurriculars', 'email', 'phone', 'GPA', 'years_experience',
    #    'mos_experience', 'certifications', 'honor_societies', 'latin_honors',
    #    'scholarships_awards', 'community_college', 'major_minor',
    #    'other_universities', 'top_100_universities', 'top_10_universities',
    #    'associate_education_level', 'bachelor_education_level',
    #    'doctor_education_level', 'master_education_level', 'courses',
    #    'languages', 'hobbies', 'technical_skills', 'company_consumer',
    #    'company_energychem', 'company_fin', 'company_foodbev',
    #    'company_health', 'company_industrial', 'company_other',
    #    'company_services', 'company_tech', 'HonorSociety', 'LatinHonors',
    #    'ScholarshipsAward', 'CommCollege', 'OtherUni', 'Top100Uni', 'Top10Uni',
    #    'Associates', 'Bachelors', 'Masters', 'Doctors', 'FoodBev', 'Consumer',
    #    'EnergyChem', 'Fin', 'HealthMed', 'Industrial', 'Tech', 'Services',
    #    'OtherCo', 'CFA', 'CQE', 'HACCP', 'CPA', 'ACCA', 'CMA', 'CIMA', 'CFCS',
    #    'CPIM', 'PHR', 'PMP', 'FEexam', 'LeanSixSigmaBlackBelt', 'CHCM',
    #    'OSHACertified', 'CSP', 'ASP', 'CertifiedSanitarian', 'CBA', 'CCT',
    #    'CFSQA', 'CMQOE', 'CMBB', 'CQA', 'CQIA', 'CQI', 'CQPA', 'CQT', 'CSSBB',
    #    'CSSGB', 'CSSYB', 'CSQE', 'CSQP', 'Accounting', 'AdultDevelopment',
    #    'Aeronautics', 'AfricanStudies', 'AgriculturalBusiness', 'Agriculture',
    #    'AirwayScience', 'AmericanStudies', 'Analytics', 'AnimalScience',
    #    'Archaeology', 'Architecture', 'ArtStudies', 'ArtificialIntelligence',
    #    'AsianStudies', 'AutomotiveEngineering', 'Aviation', 'BakeryScience',
    #    'Biology', 'BiomedicalEngineering', 'Biophysics', 'BioresourceScience',
    #    'BusinessAdministration', 'BusinessAnalytics', 'ChemicalEngineering',
    #    'ChildCare', 'CivilEngineering', 'CommunicationTechnology',
    #    'Communication', 'ComputerEngineering', 'ComputerNetworks&Systems',
    #    'ComputerScience', 'ConstructionEngineering', 'Counseling',
    #    'CriminalScience', 'DataManagementTechnology', 'DataScience',
    #    'DentalHygiene', 'DieselTechnology', 'DraftingTechnology',
    #    'EarlyChildhoodEducation', 'Ecology', 'Economics',
    #    'EducationAdministration', 'ElectricalEngineering',
    #    'EnglishAsASecondLanguage', 'EnglishEducation', 'EnvironmentalDesign',
    #    'EnvironmentalEngineering', 'EnvironmentalStudies', 'EquestrianStudies',
    #    'EthnicStudies', 'EuropeanStudies', 'FacilitiesAdministration',
    #    'Fashion', 'Film', 'Finance', 'FitnessManagement', 'Food&Nutrition',
    #    'Forensics', 'Forestry', 'FuneralServices', 'GenderStudies', 'Genetics',
    #    'Genomics', 'Geography', 'Geology', 'GlobalHealth', 'GraphicDesign',
    #    'HealthServicesAdministration', 'HispanicStudies', 'HistoryofArt',
    #    'History', 'HomeEconomics', 'Horticulture', 'Hospitality',
    #    'HumanDevelopment', 'HumanResources', 'HumanRightsStudies',
    #    'Immunology', 'IndustrialProductionTechnologies', 'Informatics',
    #    'InformationScience', 'InformationTechnology', 'InteriorDesign',
    #    'InternationalBusiness', 'InternationalRelations', 'Journalism',
    #    'Kinesiology', 'LaborRelations', 'LandscapeArchitecture',
    #    'LawEnforcement', 'LawPolitics&Society', 'Law', 'LegalAssistant',
    #    'LiberalStudies', 'LiteraryStudies', 'Literature', 'Logistics',
    #    'Management', 'MarineScience', 'Marketing', 'MaterialManagement',
    #    'MaterialsScience', 'MathematicsEducation', 'Mathematics',
    #    'MechanicalEngineering', 'MedicalAssistant', 'MedicalBilling',
    #    'MedicalImagingScience', 'MedicalTechnology', 'MentalHealthServices',
    #    'Merchandising', 'Meteorology', 'MiddleEasternStudies',
    #    'MiddleSchoolEducation', 'MilitaryTechnologies', 'Music',
    #    'Neuroscience', 'Nursing', 'Oceanography', 'OfficeManagement',
    #    'OrganizationalBehavior', 'OsteopathicMedicine', 'PackagingEngineering',
    #    'Parks&Recreation', 'Pathology', 'Pharmacy', 'Philosophy',
    #    'PhysicalScience', 'Physics', 'PoliticalScience', 'Psychology',
    #    'PublicHealth', 'PublicPolicy', 'PublicRelations',
    #    'PurchasingManagement', 'QuantumComputing', 'QuantumMechanics',
    #    'Radiology', 'RehabilitationTherapy', 'ResourceManagement',
    #    'RetailManagement', 'RiskManagement', 'SeniorHighEducation',
    #    'SocialScienceEducation', 'SocialScience', 'SocialWelfare', 'Sociology',
    #    'SpeechStudies', 'SportManagement', 'SubstanceAbuseCounseling',
    #    'Surveying', 'SystemsEngineering', 'TheaterStudies', 'Theology',
    #    'Tourism', 'UrbanPlanning', 'UrbanStudies', 'VeterinarianAssisting',
    #    'VeterinaryScience', 'VisualCommunication', 'Viticulture', 'WarStudies',
    #    'WebDesign', 'WebDevelopment', 'WeldingTechnology',
    #    'WildlifeManagement', 'WorldLanguages', 'Zoology', 'Advertising',
    #    'AgriculturalEducation', 'Anatomy', 'ArtEducation', 'Biochemistry',
    #    'Bioengineering', 'Bioethics', 'BiologicalEngineering', 'Biotechnology',
    #    'Business', 'BusinessEducation', 'CeramicEngineering', 'Chemistry',
    #    'Chiropractic', 'ClinicalTrialsManagement', 'CognitiveScience',
    #    'CommunityAdvocacy', 'CommunityOrganization', 'ConflictResolution',
    #    'ContractsManagement', 'CyberSecurity', 'Dentistry', 'Design',
    #    'EducationalLeadership', 'EducationalTechnology', 'Endocrinology',
    #    'EngineeringScience', 'Entrepreneurship', 'Epidemiology',
    #    'Floriculture', 'FoodServicesManagement', 'Genealogy', 'GlobalStudies',
    #    'HealthEducation', 'HistoricPreservation', 'HomelandSecurity',
    #    'HumanServices', 'IndustrialDesign', 'IndustrialEngineering',
    #    'IndustrialManagement', 'JuridicalScience', 'LibraryScience',
    #    'ManagerialEconomics', 'Medicine', 'MetallurgicalEngineering',
    #    'MineralEngineering', 'MinorityStudies', 'MolecularEngineering',
    #    'NuclearEngineering', 'OccupationalSafety', 'Optometry', 'Paleontology',
    #    'PetroleumEngineering', 'Pharmacogenomics', 'Phlebotomy',
    #    'PhysicalEducation', 'PhysicianAssistant', 'PostsecondaryEducation',
    #    'ProjectManagement', 'PublicAdministration', 'QualityControl',
    #    'RealEstate', 'RespiratoryTherapy', 'SafetyTechnologies',
    #    'ScienceEducation', 'SecretarialStudies', 'SoftwareEngineering',
    #    'SoundEngineering', 'SpecialEducation', 'Statistics',
    #    'TechnologyEducation', 'Thermodynamics', 'Toxicology', 'Transportation',
    #    'UrbanTeacherEducation', 'Akkadian', 'Anatolian', 'Arabic', 'Aramaic',
    #    'Armenian', 'Bengali', 'Basque', 'Bosnian', 'Cantonese', 'Catalan',
    #    'Chinese', 'Croatian', 'Czech', 'Danish', 'Dutch', 'Egyptian',
    #    'English', 'French', 'Geez', 'German', 'Greek', 'Hebrew', 'Hindi',
    #    'Hokkien', 'Italian', 'Japanese', 'Kazakh', 'Korean', 'Latin', 'Malay',
    #    'Mandarin', 'Marathi', 'Norwegian', 'Pali', 'Persian', 'Polish',
    #    'Portuguese', 'Punjabi', 'Russian', 'Sanskrit', 'Serbian',
    #    'SignLanguange', 'Slavic', 'Spanish', 'Sumerian', 'Swahili', 'Tamil',
    #    'Telugu', 'Tibetan', 'Turkish', 'Welsh', 'Urdu', 'Uzbek', 'Yiddish',
    #    'APstyle', 'Abletonlive', 'Abaqus', 'AcrobatDistiller',
    #    'AdobeFireworks', 'AdobeFlash', 'Airflow', 'Algorithms',
    #    'AppDevelopment', 'Ariba', 'AutoCAD', 'Axioma', 'Azure', 'Azkaban',
    #    'Adobecreativesuite', 'Agilemethodology', 'AmazonWebServices',
    #    'ApplicationProgrammingInterface', 'Bigdata', 'BirdEye', 'Bloomberg',
    #    'C#', 'C[+][+]', 'Ceridian', 'CREST', 'CRD', 'CSS', 'Clustering',
    #    'CognosAnalytics', 'ContentManagementSystems', 'DBMS', 'DHTML',
    #    'Databasedesign', 'Datapipeline', 'Dataarchitecture', 'Dataengineering',
    #    'Datamining', 'Decisiontrees', 'Deltek', 'Docker', 'Dreamweaver',
    #    'Datavisualization', 'Elixir', 'Excel', 'EPARegulation', 'FactSet',
    #    'Facebookforbusiness', 'FinalCut', 'FSSC22000', 'Googleadwords',
    #    'Googleanalytics', 'GraphicUserInterface', 'H20', 'HRIS',
    #    'Hubspotsales', 'Hyperion', 'Hadoop', 'Hbase', 'IBMCloud',
    #    'IECstandards', 'ISOstandards', 'JQuery', 'JSON', 'Jenkins', 'Jsp',
    #    'Jspf', 'JavaScript', 'Java', 'Kanban', 'Keras', 'Keywordoptimization',
    #    'Kronos', 'LaTeX', 'LexisNexis', 'Linux', 'Luigi', 'MATLAB', 'MYOB',
    #    'Matlab', 'Machinelearning', 'Mathematica', 'Maven', 'Mendeley',
    #    'MongoDB', 'MicrosoftAccess', 'MicrosoftPowerPoint',
    #    'MicrosoftPublisher', 'MicrosoftWord', 'Microsoftoffice', 'NVivo',
    #    'Neo4j', 'Neuralnetworks', 'NodeXL', 'ObjectiveC', 'Oracle',
    #    'ORMregulation', 'PCF', 'PHP', 'Peoplesoft', 'PowerBI', 'Precima',
    #    'PRINCE2', 'Python', 'Pylearn2', 'PostgreSQL', 'Qualtrics',
    #    'Quickbooks', 'QuarkXpress', 'RAPTOR', 'RoboticProcessAutomation',
    #    'Rust', 'Ruby', 'sarbanesoxley', 'SDLC', 'ServiceNow', 'SPSS', 'SQL',
    #    'SQLite', 'STATA', 'Scala', 'Sendgrid', 'SmartCo', 'Smalltalk',
    #    'SolidWorks', 'Softwaredevelopment', 'SoftwareDevelopment',
    #    'SparkStreaming', 'Storm', 'SupplyTrack', 'Surveyxact', 'Swift',
    #    'Syteline', 'SQLServer', 'Searchengineoptimization', 'Sklearn', 'Spark',
    #    'Tableau', 'Typeform', 'Tensorflow', 'UIdesign', 'UWP', 'Ultipro',
    #    'Unittesting', 'Unix', 'VHDL', 'VBA', 'Visio', 'WML', 'WPF',
    #    'Waterfall', 'Weblogic', 'WestLaw', 'Webscraping', 'WindowsServer',
    #    'Windows', 'Wordpress', 'XAML', 'XHTML', 'XML', 'AWK', 'SEB', 'Sed',
    #    'C', 'SAP', 'R', 'SAS', 'Hive', 'Html', 'SSIS', 'GCP', 'Git', 'Mac',
    #    'operatingsystems', 'StemMajor', 'GPAmax']
    #
    # job_dummies_ideal = job_dummies_ideal[cols]


    job_features_ideal = job_dummies_ideal[job_dummies_ideal.columns[:44]]
    resume_features = resume_dummies[resume_dummies.columns[:47]]
    job_dummies_ideal.drop(job_dummies_ideal.columns[1:44], axis=1, inplace=True)
    resume_dummies.drop(resume_dummies.columns[2:47], axis=1, inplace=True)

    # read raw df text data and vectorize for embeddings
    EMAIL_REGEX = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,4}"


    def remove_stop_words(x):
        try:
            x = re.sub('(\w)(/)(\w)',r'\1 \3', x)  # for responsibilities/accountabilities
            x = re.sub(EMAIL_REGEX, "", x)  # for email
            x = re.sub('[^a-z- ]', '', x)  # for punctuation
            x = re.sub(' - ', ' ', x)  # for dashes
            x = re.sub('-', ' ', x)  # for dashes
            x = re.sub(' \w ', ' ', x)  # for single letters
            x = re.sub('\s\s+', ' ', x)  # for multiple spaces

            word_tokens = word_tokenize(x)
            filtered_sentence = [w for w in word_tokens if not w in STOP_WORDS]
            return filtered_sentence
        except Exception as e:
            print("error:", e)
            print('string:', x)
            sys.exit(1)


    job_text = job_features_ideal[['ReqID', 'text']]
    job_text["text"].replace(r'[\d]', '', regex=True, inplace=True)
    job_text["text"] = job_text["text"].astype(str).apply(lambda x: x.lower().replace('\r', ' ').replace('\n', ' ').replace('\t', ' '))
    job_text["text"] = job_text["text"].apply(remove_stop_words)

    resume_text = resume_features[['ReqID', 'CanID', 'text']]
    resume_text["text"].replace(r'[\d]', '', regex=True, inplace=True) # remove numbers
    resume_text["text"] = resume_text["text"].astype(str).apply(lambda x: x.lower().replace('\r', ' ').replace('\n', ' ').replace('\t', ' '))
    # removing 1 and 2 letter words
    resume_text["text"] = resume_text["text"].apply(lambda x: re.sub('\s\w{1,2}\s', ' ', x))
    resume_text["text"] = resume_text["text"].apply(remove_stop_words)

    resume_dummies.rename(columns={'CanID': 'ID'}, inplace=True)
    job_dummies_ideal['ID'] = job_dummies_ideal['ReqID']
    job_dummies_ideal = job_dummies_ideal[resume_dummies.columns]
    all_dummies_ideal = pd.concat([resume_dummies, job_dummies_ideal])

    def GenerateCountEmbedding(req_id, job_text_df, resume_text_df, all_resumes):
        pos_jd_text = job_text_df[job_text_df["ReqID"] == req_id]
        if all_resumes:
            pos_resume_text = resume_text_df
            pos_resume_text.drop_duplicates(keep='first', inplace=True, subset='CanID')
        else:
            pos_resume_text = resume_text_df[resume_text_df["ReqID"] == req_id]
        pos_jd_text.rename(columns={'ReqID': 'ID', 'Job Description': 'text'}, inplace=True)
        pos_jd_text.ID = req_id
        pos_jd_text = pos_jd_text[['ID', 'text']]
        pos_resume_text.rename(columns={'CanID': 'ID', 'Resume Text': 'text'}, inplace=True)
        pos_resume_text = pos_resume_text[['ID', 'text']]

        df = pos_jd_text.append(pos_resume_text)
        df.set_index('ID', inplace=True)

        df['text'] = df['text'].apply(lambda x: [''] if x[0] == 'nan' else x)
        df['text'] = df['text'].apply(lambda x: ' '.join(x))
        count = CountVectorizer()
        pos_embedding = count.fit_transform(df['text'])
        pos_embedding = pd.DataFrame(pos_embedding.toarray())
        pos_embedding.insert(loc=0, column="ID", value=df.index)

        return pos_embedding


    def RecommendTop(jobID, full_df):  # Cos Sim and rank candidates
        # returns x recommended resume ID's based on Job Description
        recommended_candidates = []
        full_df.fillna(0, inplace=True)
        indices = pd.Series(full_df["ID"])
        cos_sim = cosine_similarity(full_df.drop("ID", axis=1))  # pairwise similarities for all samples in the df
        try:
            idx = indices[indices == jobID].index[0]
            score_series = pd.Series(cos_sim[idx]).sort_values(ascending=False)
            for i in score_series.index:
                recommended_candidates.append(list(indices)[i])
        except IndexError:
            print(jobID, 'had and index error')
        output = pd.DataFrame({'Candidate ID': recommended_candidates, 'cosine': score_series})
        output = output[output['Candidate ID'] != jobID]
        return output

    count_embeddings = GenerateCountEmbedding(jobID, job_text, resume_text, all_resumes)
    count_embeddings['ReqID'] = np.repeat(jobID, len(count_embeddings))
    all_features = pd.DataFrame(count_embeddings).merge(all_dummies_ideal, how="left", on=["ID", 'ReqID'])
    rankings = RecommendTop(jobID=jobID, full_df=all_features.drop('ReqID', axis=1))

    rankings = rankings[:topX]
    rankings.reset_index(drop=True, inplace=True)

    return rankings, job_features_ideal.text[job_features_ideal.ReqID == jobID].values, all_features
