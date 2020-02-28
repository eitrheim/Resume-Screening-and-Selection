# Resume Screening and Selection
An Intelligent System to Automate Candidate Selection for Interview  
*By Ann Eitrheim, Michael Mairuo Liu, and Matthew Echols - Supervisor Dr. Utku Pamuksuz*      
University of Chicago, Master of Science in Analytics Capstone Project   



# How To Use The Candidate Selection Tool

### Optional pre-step: Update yaml file, which determines what items to parse from the job descriptions and resumes.
-	File 1: Resume-Parser-master-new/confs/config.yaml
-	File 2: Resume-Parser-JOBS/confs/config.yaml
-	Add, subtract, or change items in these files. Sections that need to be identical are courses, major_minor, languages, techincal_skills, and certifications. Some items are in a hierarchical format, meaning they are equivalents of each other. For example in the  Resume-Parser-JOBS/confs/config.yaml file, 4 year college degree, degree, Bachelor, and 4 year degree are equivalent to each other.
### Step 1: Section and parse resumes
-	File: Resume-Parser-master-new/bin/main.py
-	Change line 39 to point to the new candidate resume data, currently it is point to “data/Candidate Report.csv”
### Step 2: To Section and parse job descriptions to create the ideal candidate
-	File: Resume-Parser-JOBS/bin/main.py
-	Change line 33 to point to the new job descriptions, currently it is point to “data/full_requisition_data.csv”
### Step 3: One hot encode the resume data
-	File: Resume-Parser-master-new/OneHotRESUMES.py
### Step 4: One hot encode the job description data
-	File: Resume-Parser-JOBS/OneHotJOBS.py
### Step 5: Rank candidates
-	File: models/content_based/final_model.py
-	Change line 119 to the requisition ID you want to see the top candidates
