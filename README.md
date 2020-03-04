# Resume Screening and Selection
An Intelligent System to Automate Candidate Selection for Interview

    Ann Eitrheim, Michael Mairuo Liu, and Matthew Echols  
    University of Chicago, Master of Science in Analytics Capstone Project  
    Supervisor Dr. Utku Pamuksuz  



# How To Use The Candidate Selection Tool

### Optional pre-step: Update yaml file, which determines what items to parse from the job descriptions and resumes.
-	File 1: `Resume-Parser-master-new/confs/config.yaml`
-	File 2: `Resume-Parser-JOBS/confs/config.yaml`
-	Add, subtract, or change items in these files. Sections that need to be identical are courses, major_minor, languages, techincal_skills, and certifications. Some items are in a hierarchical format, meaning they are equivalents of each other. For example in the  `Resume-Parser-JOBS/confs/config.yaml` file, 4 year college degree, degree, Bachelor, and 4 year degree are equivalent to each other.
### Step 1: Add Resumes
-	Folder: `Resume-Parser-master-new/data/input/resumes`
-   Change the job id that new resumes are tied to:
    -   In line 34 of `Resume-Parser-master-new/bin/main.py `   
    *----------or----------*    
    -   In the csv file directly: `Resume-Parser-master-new/data/output/resume_summary.csv`
### Step 2: Add job descriptions to create the ideal candidate
-	File: `Resume-Parser-JOBS/data/job_descriptions.csv`
-	The first column is a unique code (ReqID) and the second is the job description (text)
### Step 3: Rank candidates
-	File: `pipeline.py`
-	Update line 34 `pipeline(jobID='abcd123', topX=10, root_file_path='/Users/anneitrheim/PycharmProjects/Resume-Screening-and-Selection/')`:
    -   Select jobID to rank candidates: `jobID`   
    -   Select number of top candidates to show: `topX`   
    -   Enter file path of the repo if on your local machine: `root_file_path`
