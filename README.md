# Resume Screening and Selection
An Intelligent System to Automate Candidate Selection for Interview

    Ann Eitrheim, Michael Mairuo Liu, and Matthew Echols  
    University of Chicago, Master of Science in Analytics Capstone Project  
    Supervisor Dr. Utku Pamuksuz  



# How To Use The Candidate Selection Tool

### Optional pre-step: Update yaml file, which determines what items to parse from the job descriptions and resumes.
-	File 1: Resume-Parser-master-new/confs/config.yaml
-	File 2: Resume-Parser-JOBS/confs/config.yaml
-	Add, subtract, or change items in these files. Sections that need to be identical are courses, major_minor, languages, techincal_skills, and certifications. Some items are in a hierarchical format, meaning they are equivalents of each other. For example in the  Resume-Parser-JOBS/confs/config.yaml file, 4 year college degree, degree, Bachelor, and 4 year degree are equivalent to each other.
### Step 1: Add Resumes
-	Folder: Resume-Parser-master-new/data/input/resumes
### Step 2: Add job descriptions to create the ideal candidate
-	File: Resume-Parser-JOBS/data/job_descriptions.csv
-	The first column is a unique code (ReqID) and the second is the job description (text)
### Step 3: Rank candidates
-	File: pipeline.py
-	Update line 6 `pipeline(pdf_to_text=0, jobID='abcd123', topX=100)`
        -   Choose if you want to parse the resumes (do if you have new resumes) `pdf_to_text=1`   
        -   Select jobID to rank candidates for `jobID='abcd123'`   
        -   Select number of top candidates to show `topX=100`   
