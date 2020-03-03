"""
Sectioning the resume into education, work experience, summary, technical skills, research, and activities.
"""
import numpy as np
import pandas as pd


def section_into_columns(observations):
    df = observations
    df = df.reset_index(drop=True)
    # EDUCATION SECTION ##############################
    df['EducationLocation'] = np.repeat(-1, len(df))
    df['AcademicLocation'] = np.repeat(-1, len(df))
    df['CourseWorkLocation'] = np.repeat(-1, len(df))
    df['CoursesLocation'] = np.repeat(-1, len(df))
    df['RelatedCourseLocation'] = np.repeat(-1, len(df))
    # TODO school? university? college?
    # WORK SECTION ##############################
    df['ProfessionalHistoryLocation'] = np.repeat(-1, len(df))
    df['ProfessionalBackgroundLocation'] = np.repeat(-1, len(df))
    df['EmploymentHistoryLocation'] = np.repeat(-1, len(df))
    df['ProfessionalTrainingLocation'] = np.repeat(-1, len(df))
    df['CareerHistoryLocation'] = np.repeat(-1, len(df))
    df['WorkHistoryLocation'] = np.repeat(-1, len(df))
    df['ExperienceLocation'] = np.repeat(-1, len(df))
    df['ApprenticeshipsLocation'] = np.repeat(-1, len(df))
    df['InternshipsLocation'] = np.repeat(-1, len(df))
    df['PreviousRolesLocation'] = np.repeat(-1, len(df))
    df['CurrentRoleLocation'] = np.repeat(-1, len(df))
    df['PositionsLocation'] = np.repeat(-1, len(df))
    # SUMMARY SECTION ##############################
    df['ObjectiveLocation'] = np.repeat(-1, len(df))
    df['SummaryLocation'] = np.repeat(-1, len(df))
    df['CareerGoalLocation'] = np.repeat(-1, len(df))
    df['AboutMeLocation'] = np.repeat(-1, len(df))
    df['ProfileLocation'] = np.repeat(-1, len(df))
    df['PersonalStatementLocation'] = np.repeat(-1, len(df))
    # TECHNICAL SECTION ##############################  knowledge related to the job
    df['TechnicalSkillsLocation'] = np.repeat(-1, len(df))
    df['TechnologiesLocation'] = np.repeat(-1, len(df))
    df['SoftwareLocation'] = np.repeat(-1, len(df))
    df['ComputerSkillsLocation'] = np.repeat(-1, len(df))
    df['SkillsLocation'] = np.repeat(-1, len(df))
    df['CompetenciesLocation'] = np.repeat(-1, len(df))
    df['CoreCompetenciesLocation'] = np.repeat(-1, len(df))
    df['CertificationsLocation'] = np.repeat(-1, len(df))
    df['LicensesLocation'] = np.repeat(-1, len(df))
    df['CredentialsLocation'] = np.repeat(-1, len(df))
    df['ComputerKnowledgeLocation'] = np.repeat(-1, len(df))
    df['QualificationsLocation'] = np.repeat(-1, len(df))
    df['CareerRelatedSkillsLocation'] = np.repeat(-1, len(df))
    df['LanguageLocation'] = np.repeat(-1, len(df))
    df['ProgrammingLanguageLocation'] = np.repeat(-1, len(df))
    df['SpecializedSkillsLocation'] = np.repeat(-1, len(df))
    df['SpecialTrainingLocation'] = np.repeat(-1, len(df))
    df['TrainingLocation'] = np.repeat(-1, len(df))
    df['ProficienciesLocation'] = np.repeat(-1, len(df))
    df['AreasOfLocation'] = np.repeat(-1, len(df))
    df['ProfessionalSkillsLocation'] = np.repeat(-1, len(df))
    df['ProfessionalActivitiesLocation'] = np.repeat(-1, len(df))
    df['ProfessionalAffiliationsLocation'] = np.repeat(-1, len(df))
    df['ProfessionalAssociationsLocation'] = np.repeat(-1, len(df))
    df['ProfessionalMembershipsLocation'] = np.repeat(-1, len(df))
    df['ProfessionalInvolvementLocation'] = np.repeat(-1, len(df))
    df['ProfessionalOrganizationsLocation'] = np.repeat(-1, len(df))
    df['AssociationsLocation'] = np.repeat(-1, len(df))
    df['DistinctionsLocation'] = np.repeat(-1, len(df))
    df['EndorsementsLocation'] = np.repeat(-1, len(df))
    df['MembershipsLocation'] = np.repeat(-1, len(df))
    # RESEARCH SECTION ############################## work and school mix
    df['FellowshipsLocation'] = np.repeat(-1, len(df))
    df['AcademicHonorsLocation'] = np.repeat(-1, len(df))
    df['DissertationsLocation'] = np.repeat(-1, len(df))
    df['PapersLocation'] = np.repeat(-1, len(df))
    df['HonorsLocation'] = np.repeat(-1, len(df))
    df['PresentationsLocation'] = np.repeat(-1, len(df))
    df['PublicationsLocation'] = np.repeat(-1, len(df))
    df['ResearchLocation'] = np.repeat(-1, len(df))
    df['ScholarshipsLocation'] = np.repeat(-1, len(df))
    df['CurrentResearchLocation'] = np.repeat(-1, len(df))
    df['AcademicServiceLocation'] = np.repeat(-1, len(df))
    df['ConferenceLocation'] = np.repeat(-1, len(df))
    df['AwardsLocation'] = np.repeat(-1, len(df))
    df['ConventionsLocation'] = np.repeat(-1, len(df))
    df['ProjectsLocation'] = np.repeat(-1, len(df))
    df['ExhibitsLocation'] = np.repeat(-1, len(df))
    df['AccoladesLocation'] = np.repeat(-1, len(df))
    df['ProgramsLocation'] = np.repeat(-1, len(df))
    # TODO add REFERENCES, PROFESSIONAL RECOGNITION, FUNDING, PROFESSIONAL SERVICE section?
    # ACTIVITY SECTION ############################## what you do outside of work and school
    df['VolunteerLocation'] = np.repeat(-1, len(df))
    df['Co_CurricularLocation'] = np.repeat(-1, len(df))
    df['ExtracurricularLocation'] = np.repeat(-1, len(df))
    df['Extra_CurricularLocation'] = np.repeat(-1, len(df))
    df['CommunityLocation'] = np.repeat(-1, len(df))
    df['AffiliationsLocation'] = np.repeat(-1, len(df))
    df['InterestsLocation'] = np.repeat(-1, len(df))
    df['HobbiesLocation'] = np.repeat(-1, len(df))
    df['ActivitiesLocation'] = np.repeat(-1, len(df))
    df['AthleticInvolvementLocation'] = np.repeat(-1, len(df))
    df['CivicActivitiesLocation'] = np.repeat(-1, len(df))
    df['CollegeActivitiesLocation'] = np.repeat(-1, len(df))
    df['LeadershipLocation'] = np.repeat(-1, len(df))
    df['InvolvementLocation'] = np.repeat(-1, len(df))
    df['CampusInvolvementLocation'] = np.repeat(-1, len(df))
    df['AccomplishmentsLocation'] = np.repeat(-1, len(df))
    df['AchievementsLocation'] = np.repeat(-1, len(df))
    df['AdditionalLocation'] = np.repeat(-1, len(df))

    resume_total = len(df)
    print(resume_total, "total resumes.")
    for i in df.index:
        if i % 1000 == 0:
          print('Parsing', i, 'of', resume_total)

        # EDUCATION SECTION ##############################
        if df.text.loc[i].lower().find('education') != -1:  # if it can find the word education
            if df.text.loc[i].find('EDUCATION') != -1:
                df['EducationLocation'].loc[i] = df.text.loc[i].find('EDUCATION')
            else:
                df['EducationLocation'].loc[i] = df.text.loc[i].find('\nEducation')
            # education
            # educational background
            # educational qualifications
            # educational training
            # education and training
            # education and certifications
        if df.text.loc[i].lower().find('academic') != -1:
            if df.text.loc[i].lower().find('academic detail') != -1:
                df['AcademicLocation'].loc[i] = df.text.loc[i].lower().find('\nacademic detail')
            elif df.text.loc[i].lower().find('academic profile') != -1:
                df['AcademicLocation'].loc[i] = df.text.loc[i].lower().find('\nacademic profile')
            elif df.text.loc[i].lower().find('academic background') != -1:
                df['AcademicLocation'].loc[i] = df.text.loc[i].lower().find('\nacademic background')
            elif df.text.loc[i].lower().find('academic qualification') != -1:
                df['AcademicLocation'].loc[i] = df.text.loc[i].lower().find('\nacademic qualification')
            elif df.text.loc[i].lower().find('academic experience') != -1:
                df['AcademicLocation'].loc[i] = df.text.loc[i].lower().find('\nacademic experience')
            else:
                df['AcademicLocation'].loc[i] = df.text.loc[i].lower().find('\nacademic training')
        if df.text.loc[i].lower().find('related course') != -1:
            df['RelatedCourseLocation'].loc[i] = df.text.loc[i].lower().find('\nrelated course')
            # related course projects
            # related course work
        if df.text.loc[i].lower().find('course work') != -1:
            if df.text.loc[i].lower().find('\ncourse work') != -1:
                df['CourseWorkLocation'].loc[i] = df.text.loc[i].lower().find('\ncourse work')
            elif df.text.loc[i].find('\nCoursework') != -1:
                df['CourseWorkLocation'].loc[i] = df.text.loc[i].find('\nCoursework')
            else:
                df['CourseWorkLocation'].loc[i] = df.text.loc[i].find('\nCOURSEWORK')
        if df.text.loc[i].lower().find('courses') != -1:
            if df.text.loc[i].find('COURSES') != -1:
                df['CoursesLocation'].loc[i] = df.text.loc[i].find('COURSES')
            else:
                df['CoursesLocation'].loc[i] = df.text.loc[i].find('\nCourses')

        # WORK SECTION ##############################

        if df.text.loc[i].lower().find('professional history') != -1:
            df['ProfessionalHistoryLocation'].loc[i] = df.text.loc[i].lower().find('\nprofessional history')
        if df.text.loc[i].lower().find('professional background') != -1:
            df['ProfessionalBackgroundLocation'].loc[i] = df.text.loc[i].lower().find('\nprofessional background')
        if df.text.loc[i].lower().find('employment') != -1:
            if df.text.loc[i].lower().find('\nemployment history') != -1:
                df['EmploymentHistoryLocation'].loc[i] = df.text.loc[i].lower().find('\nemployment history')
            elif df.text.loc[i].find('EMPLOYMENT HISTORY') != -1:
                df['EmploymentHistoryLocation'].loc[i] = df.text.loc[i].find('EMPLOYMENT HISTORY')
            else:
                df['EmploymentHistoryLocation'].loc[i] = df.text.loc[i].find('EMPLOYMENT')
        if df.text.loc[i].lower().find('professional training') != -1:
            df['ProfessionalTrainingLocation'].loc[i] = df.text.loc[i].lower().find('\nprofessional training')
        if df.text.loc[i].lower().find('career history') != -1:
            df['CareerHistoryLocation'].loc[i] = df.text.loc[i].lower().find('\ncareer history')
        if df.text.loc[i].lower().find('work history') != -1:
            df['WorkHistoryLocation'].loc[i] = df.text.loc[i].lower().find('\nwork history')
        if df.text.loc[i].lower().find('experience') != -1:
            if df.text.loc[i].lower().find('\nprofessional experience') != -1:
                df['ExperienceLocation'].loc[i] = df.text.loc[i].lower().find('\nprofessional experience')
            elif df.text.loc[i].lower().find('\nadditional experience') != -1:
                df['ExperienceLocation'].loc[i] = df.text.loc[i].lower().find('\nadditional experience')
            elif df.text.loc[i].lower().find('\nwork experience') != -1:
                df['ExperienceLocation'].loc[i] = df.text.loc[i].lower().find('\nwork experience')
            elif df.text.loc[i].lower().find('\nrelevant experience') != -1:
                df['ExperienceLocation'].loc[i] = df.text.loc[i].lower().find('\nrelevant experience')
            elif df.text.loc[i].find('EXPERIENCE') != -1:
                df['ExperienceLocation'].loc[i] = df.text.loc[i].find('EXPERIENCE')
            elif df.text.loc[i].lower().find('\nexperience') != -1:
                df['ExperienceLocation'].loc[i] = df.text.loc[i].lower().find('\nexperience')
            elif df.text.loc[i].lower().find('experience\n') != -1:
                if df.text.loc[i].lower().find('academic experience\n') != -1:
                    pass
                elif df.text.loc[i].lower().find('abroad experience\n') != -1:
                    pass
                else:
                    df['ExperienceLocation'].loc[i] = df.text.loc[i].lower().find('experience\n')
            elif df.text.loc[i].lower().replace(':', ' ').find('experience \n'):
                if df.text.loc[i].lower().find('academic experience \n') != -1:
                    pass
                elif df.text.loc[i].lower().find('abroad experience \n') != -1:
                    pass
                else:
                    df['ExperienceLocation'].loc[i] = df.text.loc[i].lower().replace(':', ' ').find('experience \n')
            elif df.text.loc[i].lower().replace(':', ' ').replace('-', ' ').find(' experience \n'):
                if df.text.loc[i].lower().find('academic experience  \n') != -1:
                    pass
                elif df.text.loc[i].lower().find('abroad experience  \n') != -1:
                    pass
                else:
                    df['ExperienceLocation'].loc[i] = df.text.loc[i].lower().replace(':', ' ').find(' experience \n')
            # experience
            # work experience
            # professional experience
            # additional experience
            # relevant experience
            # legal experience
            # other experience
            # teaching experience
            # xyz experience
        if df.text.loc[i].lower().find('apprenticeships') != -1:
            df['ApprenticeshipsLocation'].loc[i] = df.text.loc[i].lower().find('\napprenticeships')
        if df.text.loc[i].lower().find('internships') != -1:
            df['InternshipsLocation'].loc[i] = df.text.loc[i].lower().find('\ninternships')
        if df.text.loc[i].lower().find('previous roles') != -1:
            df['PreviousRolesLocation'].loc[i] = df.text.loc[i].lower().find('\nprevious roles')
        if df.text.loc[i].lower().find('current role') != -1:
            df['CurrentRoleLocation'].loc[i] = df.text.loc[i].lower().find('\ncurrent role')
        if df.text.loc[i].lower().find('positions held') != -1:
            if df.text.loc[i].lower().find('\npositions held') != -1:
                df['PositionsLocation'].loc[i] = df.text.loc[i].lower().find('\npositions held')
            else:
                df['PositionsLocation'].loc[i] = df.text.loc[i].lower().replace('-', ' ').find('part time positions')

        # SUMMARY SECTION ##############################

        if df.text.loc[i].lower().find('objective') != -1:
            if df.text.loc[i].find('OBJECTIVE') != -1:
                df['ObjectiveLocation'].loc[i] = df.text.loc[i].find('OBJECTIVE')
            elif df.text.loc[i].find('\nObjective') != -1:
                df['ObjectiveLocation'].loc[i] = df.text.loc[i].find('\nObjective')
            elif df.text.loc[i].find(' objective\n') != -1:
                df['ObjectiveLocation'].loc[i] = df.text.loc[i].find(' Objective\n')
            else:
                df['ObjectiveLocation'].loc[i] = df.text.loc[i].replace(':', ' ').find(' Objective \n')
                # covers titles:
            # objective
            # career objective
            # employment objective
            # professional objective
        if df.text.loc[i].lower().find('summary') != -1:
            if df.text.loc[i].lower().find('\nsummary') != -1:
                df['SummaryLocation'].loc[i] = df.text.loc[i].lower().find('\nsummary')
            elif df.text.loc[i].lower().find(' summary\n') != -1:
                df['SummaryLocation'].loc[i] = df.text.loc[i].lower().find(' summary\n')
            else:
                df['SummaryLocation'].loc[i] = df.text.loc[i].lower().find(' summary \n')
                # summary
            # career summary
            # professional summary
            # summary of qualifications
        if df.text.loc[i].lower().find('career goal') != -1:
            df['CareerGoalLocation'].loc[i] = df.text.loc[i].lower().find('\ncareer goal')
        if df.text.loc[i].lower().find('about me') != -1:
            df['AboutMeLocation'].loc[i] = df.text.loc[i].lower().replace(':', ' ').replace('\n',
                                                                                            '  ').find('about me ')
        if df.text.loc[i].lower().find('profile') != -1:
            if df.text.loc[i].lower().find('\nprofile of skills') != -1:
                pass
            elif df.text.loc[i].lower().find('\nqualificiation profile:') != -1:
                pass
            elif df.text.loc[i].lower().find('\nacademic profile:') != -1:
                pass
            elif df.text.loc[i].lower().find('\nprofile:') != -1:
                df['ProfileLocation'] = df.text.loc[i].lower().find('\nprofile:')
            elif df.text.loc[i].lower().find('\npersonal profile') != -1:
                df['ProfileLocation'] = df.text.loc[i].lower().find('\npersonal profile')
            elif df.text.loc[i].lower().find('\ncareer profile') != -1:
                df['ProfileLocation'] = df.text.loc[i].lower().find('\ncareer profile')
            elif df.text.loc[i].lower().find('\nprofessional profile') != -1:
                df['ProfileLocation'] = df.text.loc[i].lower().find('\nprofessional profile')
            elif df.text.loc[i].lower().find('\nbusiness profile') != -1:
                df['ProfileLocation'] = df.text.loc[i].lower().find('\nbusiness profile')
            elif df.text.loc[i].lower().find('executive profile') != -1:
                if df.text.loc[i].lower().find('\nexecutive profile') != -1:
                    df['ProfileLocation'] = df.text.loc[i].lower().find('\nexecutive profile')
                elif df.text.loc[i].lower().find('executive profile') != -1:
                    df['ProfileLocation'] = df.text.loc[i].lower().find('executive profile\n')
                else:
                    df['ProfileLocation'] = df.text.loc[i].lower().replace(':', ' ').find('executive profile \n')
                # senior executive profile
            else:
                df['ProfileLocation'] = df.text.loc[i].replace(':', ' ').replace('\n', '  ').find('PROFILE ')
            if df.text.loc[i].lower().find('\npersonal statement') != -1:
                df['PersonalStatementLocation'].loc[i] = df.text.loc[i].lower().find('\npersonal statement')

        # TECHNICAL SECTION ##############################

        if df.text.loc[i].lower().find('technical skills') != -1:
            df['TechnicalSkillsLocation'].loc[i] = df.text.loc[i].lower().find('\ntechnical skills')
        if df.text.loc[i].lower().find('technologies') != -1:
            if df.text.loc[i].find('TECHNOLOGIES') != -1:
                df['TechnologiesLocation'].loc[i] = df.text.loc[i].find('TECHNOLOGIES')
            else:
                df['TechnologiesLocation'].loc[i] = df.text.loc[i].find('\nTechnologies')
        if df.text.loc[i].lower().find('software') != -1:
            if df.text.loc[i].find('\nSOFTWARE') != -1:
                df['SoftwareLocation'].loc[i] = df.text.loc[i].find('\nSOFTWARE')
            else:
                df['SoftwareLocation'].loc[i] = df.text.loc[i].find('\nSoftware')
        if df.text.loc[i].lower().find('computer skills') != -1:
            df['ComputerSkillsLocation'].loc[i] = df.text.loc[i].lower().find('\ncomputer skills')
        if df.text.loc[i].lower().find('skill') != -1:
            if df.text.loc[i].lower().find('language skills') != -1:
                pass
            elif df.text.loc[i].find('SKILLS') != -1:
                df['SkillsLocation'].loc[i] = df.text.loc[i].find('SKILLS')
            elif df.text.loc[i].lower().find('skill set') != -1:
                df['SkillsLocation'].loc[i] = df.text.loc[i].lower().find('skill set')
            else:
                df['SkillsLocation'].loc[i] = df.text.loc[i].find('\nSkills')
        if df.text.loc[i].lower().find('competencies') != -1:
            if df.text.loc[i].find('COMPETENCIES') != -1:
                df['CompetenciesLocation'].loc[i] = df.text.loc[i].find('COMPETENCIES')
            else:
                df['CompetenciesLocation'].loc[i] = df.text.loc[i].find('\nCompetencies')
        if df.text.loc[i].lower().find('core competencies') != -1:
            df['CoreCompetenciesLocation'].loc[i] = df.text.loc[i].lower().find('\ncore competencies')
        if df.text.loc[i].lower().find('certificat') != -1:
            if df.text.loc[i].find('CERTIFICATIONS') != -1:
                df['CertificationsLocation'].loc[i] = df.text.loc[i].find('CERTIFICATIONS')
            elif df.text.loc[i].find('CERTIFICATES') != -1:
                df['CertificationsLocation'].loc[i] = df.text.loc[i].find('CERTIFICATES')
            else:
                df['CertificationsLocation'].loc[i] = df.text.loc[i].find('\nCertifications')
        if df.text.loc[i].lower().find('licenses') != -1:
            if df.text.loc[i].find('LICENSES') != -1:
                df['LicensesLocation'].loc[i] = df.text.loc[i].find('LICENSES')
            else:
                df['LicensesLocation'].loc[i] = df.text.loc[i].find('\nLicenses')
        if df.text.loc[i].lower().find('credentials') != -1:
            if df.text.loc[i].find('CREDENTIALS') != -1:
                df['CredentialsLocation'].loc[i] = df.text.loc[i].find('CREDENTIALS')
            else:
                df['CredentialsLocation'].loc[i] = df.text.loc[i].find('\nCredentials')
        if df.text.loc[i].lower().find('computer knowledge') != -1:
            df['ComputerKnowledgeLocation'].loc[i] = df.text.loc[i].lower().find('\ncomputer knowledge')
        if df.text.loc[i].lower().find('qualifications') != -1:
            if df.text.loc[i].find('\nQUALIFICATION') != -1:
                df['QualificationsLocation'].loc[i] = df.text.loc[i].find('\nQUALIFICATION')
            elif df.text.loc[i].find('\nQualifications') != -1:
                df['QualificationsLocation'].loc[i] = df.text.loc[i].find('\nQualifications')
            else:
                df['QualificationsLocation'].loc[i] = df.text.loc[i].lower().find('\nprofessional qualification')
        if df.text.loc[i].lower().find('career related skills') != -1:
            df['CareerRelatedSkillsLocation'].loc[i] = df.text.loc[i].lower().find('\ncareer related skills')
        if df.text.loc[i].lower().find('language') != -1:
            if df.text.loc[i].rfind('LANGUAGE') != -1:
                df['LanguageLocation'].loc[i] = df.text.loc[i].rfind('LANGUAGE')
            elif df.text.loc[i].rfind('Language Proficienc') != -1:
                df['LanguageLocation'].loc[i] = df.text.loc[i].rfind('Language Proficienc')
            else:
                df['LanguageLocation'].loc[i] = df.text.loc[i].rfind('\nLanguage')
            # language skills
            # languages
            # language competencies
        if df.text.loc[i].lower().find('programming language') != -1:
            df['ProgrammingLanguageLocation'].loc[i] = df.text.loc[i].lower().find('\nprogramming language')
        if df.text.loc[i].lower().find('specialized skills') != -1:
            df['SpecializedSkillsLocation'].loc[i] = df.text.loc[i].lower().find('\nspecialized skills')
        if df.text.loc[i].lower().find('special training') != -1:
            df['SpecialTrainingLocation'].loc[i] = df.text.loc[i].lower().find('\nspecial training')
        if df.text.loc[i].lower().find('training') != -1:
            if df.text.loc[i].find('TRAINING') != -1:
                df['TrainingLocation'].loc[i] = df.text.loc[i].find('TRAINING')
            else:
                df['TrainingLocation'].loc[i] = df.text.loc[i].find('\nTraining')
        if df.text.loc[i].lower().find('proficiencies') != -1:
            if df.text.loc[i].find('PROFICIENCIES') != -1:
                df['ProficienciesLocation'].loc[i] = df.text.loc[i].find('PROFICIENCIES')
            else:
                df['ProficienciesLocation'].loc[i] = df.text.loc[i].find('\nProficiencies')
        if df.text.loc[i].lower().find('areas of ') != -1:
            df['AreasOfLocation'].loc[i] = df.text.loc[i].lower().find('\nareas of ')
            # areas of experience
            # areas of expertise
            # areas of knowledge
        if df.text.loc[i].lower().find('professional skills') != -1:
            df['ProfessionalSkillsLocation'].loc[i] = df.text.loc[i].lower().find('\nprofessional skills')
        if df.text.loc[i].lower().find('professional activities') != -1:
            df['ProfessionalActivitiesLocation'].loc[i] = df.text.loc[i].lower().find('\nprofessional activities')
        if df.text.loc[i].lower().find('professional affiliation') != -1:
            df['ProfessionalAffiliationsLocation'].loc[i] = df.text.loc[i].lower().find('\nprofessional affiliation')
        if df.text.loc[i].lower().find('professional association') != -1:
            df['ProfessionalAssociationsLocation'].loc[i] = df.text.loc[i].lower().find('\nprofessional association')
        if df.text.loc[i].lower().find('professional membership') != -1:
            df['ProfessionalMembershipsLocation'].loc[i] = df.text.loc[i].lower().find('\nprofessional membership')
        if df.text.loc[i].lower().find('professional involvement') != -1:
            df['ProfessionalInvolvementLocation'].loc[i] = df.text.loc[i].lower().find('\nprofessional involvement')
        if df.text.loc[i].lower().find('professional organization') != -1:
            df['ProfessionalOrganizationsLocation'].loc[i] = df.text.loc[i].lower().find('\nprofessional organization')
        if df.text.loc[i].lower().find('associations') != -1:
            if df.text.loc[i].find('ASSOCIATIONS') != -1:
                df['AssociationsLocation'].loc[i] = df.text.loc[i].find('ASSOCIATIONS')
            else:
                df['AssociationsLocation'].loc[i] = df.text.loc[i].find('\nAssociations')
        if df.text.loc[i].lower().find('distinctions') != -1:
            if df.text.loc[i].find('DISTINCTIONS') != -1:
                df['DistinctionsLocation'].loc[i] = df.text.loc[i].find('DISTINCTIONS')
            else:
                df['DistinctionsLocation'].loc[i] = df.text.loc[i].find('\nDistinctions')
        if df.text.loc[i].lower().find('endorsements') != -1:
            if df.text.loc[i].find('ENDORSEMENTS') != -1:
                df['EndorsementsLocation'].loc[i] = df.text.loc[i].find('ENDORSEMENTS')
            else:
                df['EndorsementsLocation'].loc[i] = df.text.loc[i].find('\nEndorsements')
        if df.text.loc[i].lower().find('memberships') != -1:
            if df['ProfessionalMembershipsLocation'].loc[i] > -1:
                pass
            elif df.text.loc[i].find('MEMBERSHIPS') != -1:
                df['MembershipsLocation'].loc[i] = df.text.loc[i].find('MEMBERSHIPS')
            else:
                df['MembershipsLocation'].loc[i] = df.text.loc[i].find('\nmemberships')

        # RESEARCH SECTION ##############################

        if df.text.loc[i].lower().find('fellowships') != -1:
            if df.text.loc[i].find('FELLOWSHIPS') != -1:
                df['FellowshipsLocation'].loc[i] = df.text.loc[i].find('FELLOWSHIPS')
            else:
                df['FellowshipsLocation'].loc[i] = df.text.loc[i].find('\nFellowships')
        if df.text.loc[i].lower().find('academic honors') != -1:
            df['AcademicHonorsLocation'].loc[i] = df.text.loc[i].lower().find('\nacademic honors')
        if df.text.loc[i].lower().find('dissertations') != -1:
            if df.text.loc[i].find('DISSERTATIONS') != -1:
                df['DissertationsLocation'].loc[i] = df.text.loc[i].find('DISSERTATIONS')
            else:
                df['DissertationsLocation'].loc[i] = df.text.loc[i].find('\nDissertations')
        if df.text.loc[i].lower().find('papers') != -1:
            if df.text.loc[i].find('PAPERS') != -1:
                df['PapersLocation'].loc[i] = df.text.loc[i].find('PAPERS')
            else:
                df['PapersLocation'].loc[i] = df.text.loc[i].find('\npapers')
        if df.text.loc[i].lower().find('honors') != -1:
            if df.text.loc[i].find('HONORS') != -1:
                df['HonorsLocation'].loc[i] = df.text.loc[i].find('HONORS')
            else:
                df['HonorsLocation'].loc[i] = df.text.loc[i].find('\nHonors')
        if df.text.loc[i].lower().find('presentations') != -1:
            if df.text.loc[i].find('PRESENTATIONS') != -1:
                df['PresentationsLocation'].loc[i] = df.text.loc[i].find('PRESENTATIONS')
            else:
                df['PresentationsLocation'].loc[i] = df.text.loc[i].find('\nPresentations')
        if df.text.loc[i].lower().find('publication') != -1:
            if df.text.loc[i].find('PUBLICATION') != -1:
                df['PublicationsLocation'].loc[i] = df.text.loc[i].find('PUBLICATION')
            else:
                df['PublicationsLocation'].loc[i] = df.text.loc[i].find('\nPublications')
        if df.text.loc[i].lower().find('research') != -1:
            if df.text.loc[i].find('\nRESEARCH') != -1:
                df['ResearchLocation'].loc[i] = df.text.loc[i].find('\nRESEARCH')
            elif df.text.loc[i].lower().find('\nresearch grants') != -1:
                df['ResearchLocation'].loc[i] = df.text.loc[i].lower().find('\nresearch grants')
            else:
                df['ResearchLocation'].loc[i] = df.text.loc[i].lower().find('\nresearch projects')
        if df.text.loc[i].lower().find('scholarships') != -1:
            if df.text.loc[i].find('SCHOLARSHIPS') != -1:
                df['ScholarshipsLocation'].loc[i] = df.text.loc[i].find('SCHOLARSHIPS')
            else:
                df['ScholarshipsLocation'].loc[i] = df.text.loc[i].find('\nScholarships')
        if df.text.loc[i].lower().find('current research') != -1:
            if df.text.loc[i].lower().find('\ncurrent research') != -1:
                df['CurrentResearchLocation'].loc[i] = df.text.loc[i].lower().find('\ncurrent research')
            else:
                df['CurrentResearchLocation'].loc[i] = df.text.loc[i].find('CURRENT RESEARCH')
        if df.text.loc[i].lower().find('academic service') != -1:
            df['AcademicServiceLocation'].loc[i] = df.text.loc[i].lower().find('\nacademic service')
        if df.text.loc[i].lower().find('conference') != -1:
            if df.text.loc[i].find('\nCONFERENCE'):
                df['ConferenceLocation'].loc[i] = df.text.loc[i].find('\nCONFERENCE')
            else:
                df['ConferenceLocation'].loc[i] = df.text.loc[i].lower().find('\nconferences')
        if df.text.loc[i].lower().find('awards') != -1:
            if df.text.loc[i].find('AWARDS') != -1:
                df['AwardsLocation'].loc[i] = df.text.loc[i].find('AWARDS')
            else:
                df['AwardsLocation'].loc[i] = df.text.loc[i].find('\nAwards')
        if df.text.loc[i].lower().find('conventions') != -1:
            if df.text.loc[i].find('CONVENTIONS') != -1:
                df['ConventionsLocation'].loc[i] = df.text.loc[i].find('CONVENTIONS')
            else:
                df['ConventionsLocation'].loc[i] = df.text.loc[i].find('\nconventions')
        if df.text.loc[i].lower().find('projects') != -1:
            if df.text.loc[i].lower().find('course project') != -1:
                df['ProjectsLocation'].loc[i] = df.text.loc[i].lower().find('\ncourse project')
            elif df.text.loc[i].find('PROJECTS') != -1:
                df['ProjectsLocation'].loc[i] = df.text.loc[i].find('PROJECTS')
            elif df.text.loc[i].find('\nProjects') != -1:
                df['ProjectsLocation'].loc[i] = df.text.loc[i].lower().find('\nprojects')
            elif df.text.loc[i].find(' Projects\n') != -1:
                df['ProjectsLocation'].loc[i] = df.text.loc[i].replace(':', ' ').find(' Projects\n')
            else:
                df['ProjectsLocation'].loc[i] = df.text.loc[i].replace(':', ' ').find(' Projects \n')
        if df.text.loc[i].lower().find('exhibits') != -1:
            if df.text.loc[i].find('EXHIBITS') != -1:
                df['ExhibitsLocation'].loc[i] = df.text.loc[i].find('EXHIBITS')
            else:
                df['ExhibitsLocation'].loc[i] = df.text.loc[i].find('\nExhibits')
        if df.text.loc[i].lower().find('accolades') != -1:
            if df.text.loc[i].find('ACCOLADES') != -1:
                df['AccoladesLocation'].loc[i] = df.text.loc[i].find('ACCOLADES')
            else:
                df['AccoladesLocation'].loc[i] = df.text.loc[i].find('\nAccolades')
        if df.text.loc[i].lower().find('programs') != -1:
            if df.text.loc[i].find('PROGRAMS') != -1:
                df['ProgramsLocation'].loc[i] = df.text.loc[i].find('PROGRAMS')
            else:
                df['ProgramsLocation'].loc[i] = df.text.loc[i].find('\nPrograms')

            # ACTIVITY SECTION ##############################

        if df.text.loc[i].lower().find('volunteer') != -1:
            if df.text.loc[i].lower().find('\nvolunteer') != -1:
                df['VolunteerLocation'].loc[i] = df.text.loc[i].lower().find('\nvolunteer')
                # volunteer work
            elif df.text.loc[i].lower().replace(':', ' ').find(' volunteer roles\n') != -1:
                df['VolunteerLocation'].loc[i] = df.text.loc[i].lower().replace(':', ' ').find(' volunteer roles\n')
            else:
                df['VolunteerLocation'].loc[i] = df.text.loc[i].lower().replace(':', ' ').find(' volunteer roles \n')
        if df.text.loc[i].lower().find('co-curricular') != -1:
            df['Co_CurricularLocation'].loc[i] = df.text.loc[i].lower().find('\nco-curricular')
        if df.text.loc[i].lower().find('extracurricular') != -1:
            df['ExtracurricularLocation'].loc[i] = df.text.loc[i].lower().find('\nextracurricular')
            # extracurricular activities
            # extracurriculars
        if df.text.loc[i].lower().find('extra-curricular') != -1:
            df['Extra_CurricularLocation'].loc[i] = df.text.loc[i].lower().find('\nextra-curricular')
        if df.text.loc[i].lower().find('community') != -1:
            if df.text.loc[i].find('COMMUNITY') != -1:
                df['CommunityLocation'].loc[i] = df.text.loc[i].find('COMMUNITY')
            else:
                df['CommunityLocation'].loc[i] = df.text.loc[i].find('\nCommunity')
            # community engagement
            # community involvement
        if df.text.loc[i].lower().find('affiliations') != -1:
            if df.text.loc[i].find('AFFILIATIONS') != -1:
                df['AffiliationsLocation'].loc[i] = df.text.loc[i].find('AFFILIATIONS')
            else:
                df['AffiliationsLocation'].loc[i] = df.text.loc[i].find('\nAffiliations')
        if df.text.loc[i].lower().find('interests') != -1:
            if df.text.loc[i].find('INTERESTS') != -1:
                df['InterestsLocation'].loc[i] = df.text.loc[i].find('interests')
            else:
                df['InterestsLocation'].loc[i] = df.text.loc[i].find('\nInterests')
        if df.text.loc[i].lower().find('hobbies') != -1:
            if df.text.loc[i].find('HOBBIES') != -1:
                df['HobbiesLocation'].loc[i] = df.text.loc[i].find('HOBBIES')
            else:
                df['HobbiesLocation'].loc[i] = df.text.loc[i].find('\nHobbies')
        if df.text.loc[i].lower().find('activities') != -1:
            if df.text.loc[i].find('\nActivities') != -1:
                df['ActivitiesLocation'].loc[i] = df.text.loc[i].find('\nActivities')
            elif df['ProfessionalActivitiesLocation'].loc[i] > -1:
                pass
            elif df.text.loc[i].lower().find('\nresearch activities') != -1:
                pass
            # TODO other xyz activities to pass?
            elif df.text.loc[i].find('ACTIVITIES') != -1:
                df['ActivitiesLocation'].loc[i] = df.text.loc[i].find('ACTIVITIES')
            elif df.text.loc[i].replace(':', ' ').find(' Activities\n') != -1:
                df['ActivitiesLocation'].loc[i] = df.text.loc[i].replace(':', ' ').find(' Activities\n')
            else:
                df['ActivitiesLocation'].loc[i] = df.text.loc[i].replace(':', ' ').find(' Activities \n')
            # activities
            # creative activities
        if df.text.loc[i].lower().find('athletic') != -1:
            if df.text.loc[i].lower().find('\nathletic involvement') != -1:
                df['AthleticInvolvementLocation'].loc[i] = df.text.loc[i].lower().find('\nathletic involvement')
            elif df.text.loc[i].lower().find('athletics\n') != -1:
                df['AthleticInvolvementLocation'].loc[i] = df.text.loc[i].lower().find('athletics\n')
            else:
                df['AthleticInvolvementLocation'].loc[i] = df.text.loc[i].lower().replace(':', ' ').find('athletics \n')
        if df.text.loc[i].lower().find('civic activities') != -1:
            df['CivicActivitiesLocation'].loc[i] = df.text.loc[i].lower().find('\ncivic activities')
        if df.text.loc[i].lower().find('college activities') != -1:
            df['CollegeActivitiesLocation'].loc[i] = df.text.loc[i].lower().find('\ncollege activities')
        if df.text.loc[i].lower().find('leadership') != -1:
            if df.text.loc[i].find('LEADERSHIP') != -1:
                df['LeadershipLocation'].loc[i] = df.text.loc[i].find('LEADERSHIP')
            else:
                df['LeadershipLocation'].loc[i] = df.text.loc[i].find('\nleadership')
        if df.text.loc[i].lower().find('involvement') != -1:
            if df.text.loc[i].find('\ninvolvement') != -1:
                df['InvolvementLocation'].loc[i] = df.text.loc[i].find('\ninvolvement')
            elif df['ProfessionalInvolvementLocation'].loc[i] != -1:
                pass
            else:
                df['InvolvementLocation'].loc[i] = df.text.loc[i].find('INVOLVEMENT')
        if df.text.loc[i].lower().find('campus involvement') != -1:
            df['CampusInvolvementLocation'].loc[i] = df.text.loc[i].lower().find('\ncampus involvement')
        if df.text.loc[i].lower().find('accomplishments') != -1:
            if df.text.loc[i].find('ACCOMPLISHMENTS') != -1:
                df['AccomplishmentsLocation'].loc[i] = df.text.loc[i].find('ACCOMPLISHMENTS')
            else:
                df['AccomplishmentsLocation'].loc[i] = df.text.loc[i].find('\nAccomplishments')
        if df.text.loc[i].lower().find('achievements') != -1:
            if df.text.loc[i].find('ACHIEVEMENTS') != -1:
                df['AchievementsLocation'].loc[i] = df.text.loc[i].find('ACHIEVEMENTS')
            else:
                df['AchievementsLocation'].loc[i] = df.text.loc[i].find('\nAchievements')
        if df.text.loc[i].find('\nADDITIONAL') != -1:
            df['AdditionalLocation'].loc[i] = df.text.loc[i].find('\nADDITIONAL')

    # to prevent education from being cut off and put into different sections (e.g. Education and Certifications)
    for num in df.index:
        x1 = df.EducationLocation.loc[num]
        x2 = x1 + 16
        for col in df.columns[3:]:
            if (df[col].loc[num] > x1) & (df[col].loc[num] < x2):
                df[col].loc[num] = -1
    # TODO do we need to do this for other sections?

    return df


def word_put_in_sections(observations):
    df = observations

    col_list = df.columns[3:]

    # EDUCATION SECTION ##############################
    df['Education'] = np.repeat("", len(df))
    df['Academic'] = np.repeat("", len(df))
    df['RelatedCourse'] = np.repeat("", len(df))
    df['CourseWork'] = np.repeat("", len(df))
    df['Courses'] = np.repeat("", len(df))
    # WORK SECTION ##############################
    df['ProfessionalHistory'] = np.repeat("", len(df))
    df['ProfessionalBackground'] = np.repeat("", len(df))
    df['EmploymentHistory'] = np.repeat("", len(df))
    df['ProfessionalTraining'] = np.repeat("", len(df))
    df['CareerHistory'] = np.repeat("", len(df))
    df['WorkHistory'] = np.repeat("", len(df))
    df['Experience'] = np.repeat("", len(df))
    df['Apprenticeships'] = np.repeat("", len(df))
    df['Internships'] = np.repeat("", len(df))
    df['PreviousRoles'] = np.repeat("", len(df))
    df['CurrentRole'] = np.repeat("", len(df))
    df['Positions'] = np.repeat("", len(df))
    # SUMMARY SECTION ##############################
    df['Objective'] = np.repeat("", len(df))
    df['Summary'] = np.repeat("", len(df))
    df['CareerGoal'] = np.repeat("", len(df))
    df['AboutMe'] = np.repeat("", len(df))
    df['Profile'] = np.repeat("", len(df))
    df['PersonalStatement'] = np.repeat("", len(df))
    # TECHNICAL SECTION ##############################
    df['TechnicalSkills'] = np.repeat("", len(df))
    df['Technologies'] = np.repeat("", len(df))
    df['Software'] = np.repeat("", len(df))
    df['ComputerSkills'] = np.repeat("", len(df))
    df['Skills'] = np.repeat("", len(df))
    df['Competencies'] = np.repeat("", len(df))
    df['CoreCompetencies'] = np.repeat("", len(df))
    df['Certifications'] = np.repeat("", len(df))
    df['Licenses'] = np.repeat("", len(df))
    df['Credentials'] = np.repeat("", len(df))
    df['ComputerKnowledge'] = np.repeat("", len(df))
    df['Qualifications'] = np.repeat("", len(df))
    df['CareerRelatedSkills'] = np.repeat("", len(df))
    df['Language'] = np.repeat("", len(df))
    df['ProgrammingLanguage'] = np.repeat("", len(df))
    df['SpecializedSkills'] = np.repeat("", len(df))
    df['SpecialTraining'] = np.repeat("", len(df))
    df['Training'] = np.repeat("", len(df))
    df['Proficiencies'] = np.repeat("", len(df))
    df['AreasOf'] = np.repeat("", len(df))
    df['ProfessionalSkills'] = np.repeat("", len(df))
    df['ProfessionalActivities'] = np.repeat("", len(df))
    df['ProfessionalAffiliations'] = np.repeat("", len(df))
    df['ProfessionalAssociations'] = np.repeat("", len(df))
    df['ProfessionalMemberships'] = np.repeat("", len(df))
    df['ProfessionalInvolvement'] = np.repeat("", len(df))
    df['ProfessionalOrganizations'] = np.repeat("", len(df))
    df['Associations'] = np.repeat("", len(df))
    df['Distinctions'] = np.repeat("", len(df))
    df['Endorsements'] = np.repeat("", len(df))
    df['Memberships'] = np.repeat("", len(df))
    # RESEARCH SECTION ##############################
    df['Fellowships'] = np.repeat("", len(df))
    df['AcademicHonors'] = np.repeat("", len(df))
    df['Dissertations'] = np.repeat("", len(df))
    df['Papers'] = np.repeat("", len(df))
    df['Honors'] = np.repeat("", len(df))
    df['Presentations'] = np.repeat("", len(df))
    df['Publications'] = np.repeat("", len(df))
    df['Research'] = np.repeat("", len(df))
    df['Scholarships'] = np.repeat("", len(df))
    df['CurrentResearch'] = np.repeat("", len(df))
    df['AcademicService'] = np.repeat('', len(df))
    df['Conference'] = np.repeat("", len(df))
    df['Awards'] = np.repeat("", len(df))
    df['Conventions'] = np.repeat("", len(df))
    df['Projects'] = np.repeat("", len(df))
    df['Exhibits'] = np.repeat("", len(df))
    df['Accolades'] = np.repeat("", len(df))
    df['Programs'] = np.repeat("", len(df))
    # ACTIVITY SECTION ##############################
    df['Volunteer'] = np.repeat("", len(df))
    df['Co_Curricular'] = np.repeat("", len(df))
    df['Extracurricular'] = np.repeat("", len(df))
    df['Extra_Curricular'] = np.repeat("", len(df))
    df['Community'] = np.repeat("", len(df))
    df['Affiliations'] = np.repeat("", len(df))
    df['Interests'] = np.repeat("", len(df))
    df['Hobbies'] = np.repeat("", len(df))
    df['Activities'] = np.repeat("", len(df))
    df['AthleticInvolvement'] = np.repeat("", len(df))
    df['CivicActivities'] = np.repeat("", len(df))
    df['CollegeActivities'] = np.repeat("", len(df))
    df['Leadership'] = np.repeat("", len(df))
    df['Involvement'] = np.repeat("", len(df))
    df['CampusInvolvement'] = np.repeat("", len(df))
    df['Accomplishments'] = np.repeat("", len(df))
    df['Achievements'] = np.repeat("", len(df))
    df['Additional'] = np.repeat("", len(df))

    resume_total = len(df)
    print(resume_total,"resumes total")
    # putting the words into the new columns
    for num in df.index:
        if num % 1000 == 0:
            print('Sectioning', num, "...")
        x = df[col_list].loc[num].sort_values() 
        for i in range(0, len(x)):
            try:
                df[x.index[i][:-8]][num] = df.text[num][x[i]:x[i + 1]]
            except IndexError:
                df[x.index[i][:-8]][num] = df.text[num][x[i]:]

    # drop columns with the integer locations
    df.drop(col_list, axis=1, inplace=True)
    df.fillna('', inplace=True)
    
    # drop na text colums
    df = df[df.text == df.text]
    print("{} total resumes/rows".format(len(df)))

    return df


def combine_sections_preparse(observations):
    df = observations

    print('\nCombining sub-sections.')

    # EDUCATION SECTION ##############################
    # make sure that courses is after related course so it concatenates in a natural flow
    df['Edu'] = df['Education'] + df['Academic']
    df.drop(['Education', 'Academic'], axis=1, inplace=True)
    df['Course'] = df['RelatedCourse'] + df['CourseWork'] + df['Courses']
    df.drop(['RelatedCourse', 'CourseWork', 'Courses'], axis=1, inplace=True)

    # WORK SECTION ##############################
    df['Work'] = df['CurrentRole'] + df['Experience'] + df['PreviousRoles'] + df['Positions'] + \
                 df['Apprenticeships'] + df['Internships'] + df['ProfessionalHistory'] + \
                 df['ProfessionalBackground'] + df['EmploymentHistory'] + df['ProfessionalTraining'] + \
                 df['CareerHistory'] + df['WorkHistory']
    df.drop(['CurrentRole', 'Experience', 'PreviousRoles', 'Positions', 'Apprenticeships', 'Internships',
             'ProfessionalHistory', 'ProfessionalBackground', 'EmploymentHistory', 'ProfessionalTraining',
             'CareerHistory', 'WorkHistory'], axis=1, inplace=True)

    # SUMMARY SECTION ##############################
    df['Summaries'] = df['Objective'] + df['Summary'] + df['CareerGoal'] + df['AboutMe'] + df['Profile'] + \
                      df['PersonalStatement']
    df.drop(['Objective', 'Summary', 'CareerGoal', 'AboutMe', 'Profile', 'PersonalStatement'], axis=1, inplace=True)

    # TECHNICAL SECTION ##############################
    df['Skill'] = df['ProfessionalSkills'] + df['SpecializedSkills'] + df['CareerRelatedSkills'] + \
                  df['TechnicalSkills'] + df['ComputerSkills'] + df['Skills'] + df['Technologies'] + \
                  df['Software'] + df['ProgrammingLanguage'] + df['ComputerKnowledge'] + df['Certifications'] + \
                  df['Credentials'] + df['AreasOf'] + df['CoreCompetencies'] + df['Competencies'] + \
                  df['Proficiencies'] + df['Qualifications'] + df['SpecialTraining'] + df['Training'] + \
                  df['Licenses'] + df['Distinctions'] + df['Endorsements']
    df.drop(['ProfessionalSkills', 'SpecializedSkills', 'CareerRelatedSkills', 'TechnicalSkills', 'ComputerSkills',
             'Skills', 'Technologies', 'Software', 'ProgrammingLanguage', 'ComputerKnowledge', 'Certifications',
             'Licenses', 'Credentials', 'AreasOf', 'CoreCompetencies', 'Competencies', 'Proficiencies',
             'Qualifications', 'SpecialTraining', 'Training', 'Distinctions', 'Endorsements'], axis=1, inplace=True)
    df['Member'] = df['ProfessionalMemberships'] + df['Memberships'] + df['ProfessionalAssociations'] + \
                   df['Associations'] + df['ProfessionalOrganizations'] + df['ProfessionalAffiliations'] + \
                   df['ProfessionalInvolvement'] + df['ProfessionalActivities']
    df.drop(['ProfessionalMemberships', 'Memberships', 'ProfessionalAssociations', 'Associations',
             'ProfessionalOrganizations', 'ProfessionalAffiliations', 'ProfessionalInvolvement',
             'ProfessionalActivities'], axis=1, inplace=True)

    # RESEARCH SECTION ##############################
    df['Writing'] = df['Dissertations'] + df['Papers'] + df['Presentations'] + df['Publications'] + df['Exhibits']
    df.drop(['Dissertations', 'Papers', 'Presentations', 'Publications', 'Exhibits'], axis=1, inplace=True)
    df['Researching'] = df['CurrentResearch'] + df['Research'] + df['Projects'] + df['Programs'] + \
                        df['Conference'] + df['Conventions']
    df.drop(['CurrentResearch', 'Research', 'Projects', 'Programs', 'Conference', 'Conventions'], axis=1, inplace=True)
    df['Honor'] = df['Scholarships'] + df['Awards'] + df['Fellowships'] + df['AcademicHonors'] + df['Honors'] + \
                  df['Accolades'] + df['AcademicService']
    df.drop(['Scholarships', 'Awards', 'Fellowships', 'AcademicHonors', 'Honors', 'Accolades', 'AcademicService'],
            axis=1, inplace=True)

    # ACTIVITY SECTION ##############################
    df['Activity'] = df['Volunteer'] + df['Community'] + df['CivicActivities'] + df['CollegeActivities'] + \
                     df['Activities'] + df['Leadership'] + df['Additional']
    df.drop(['Volunteer', 'Community', 'CivicActivities', 'CollegeActivities', 'Activities', 'Leadership',
             'Additional'], axis=1, inplace=True)

    df['Curriculars'] = df['Co_Curricular'] + df['Extracurricular'] + df['Extra_Curricular'] + \
                        df['CampusInvolvement'] + df['AthleticInvolvement'] + df['Involvement'] +\
                        df['Affiliations'] + df['Accomplishments'] + df['Achievements']
    df.drop(['Co_Curricular', 'Extracurricular', 'Extra_Curricular', 'CampusInvolvement', 'AthleticInvolvement',
             'Involvement', 'Affiliations', 'Accomplishments', 'Achievements'],
            axis=1, inplace=True)

    df['Hobby'] = df['Interests'] + df['Hobbies']
    df.drop(['Interests', 'Hobbies'], axis=1, inplace=True)

    return df


def combine_sections_postparse(observations):
    df = observations

    print('\nCombining sub-sections.')

    # EDUCATION SECTION ##############################
    df['Education'] = df['Edu'] + df['Course']
    df.drop(['Edu', 'Course'], axis=1, inplace=True)

    # ACTIVITY SECTION ##############################

    df['Extracurriculars'] = df['Curriculars'] + df['Hobby']
    df.drop(['Curriculars', 'Hobby'], axis=1, inplace=True)

    return df

