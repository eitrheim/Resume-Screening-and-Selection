# Resume Screening and Selection
An Intelligent System to Automate Candidate Selection for Interview

    University of Chicago, Master of Science in Analytics Capstone Project  
    Supervisor Dr. Utku Pamuksuz  



# How To Use The Candidate Selection Tool

### Optional pre-step: Update yaml file, which determines what items to parse from the job descriptions and resumes.
-	File 1: `Resume-Parser-master-new/confs/config.yaml`
-	File 2: `Resume-Parser-JOBS/confs/config.yaml`
-	Add, subtract, or change items in these files. Sections that need to be identical are courses, major_minor, languages, techincal_skills, and certifications. Some items are in a hierarchical format, meaning they are equivalents of each other. For example in the  `Resume-Parser-JOBS/confs/config.yaml` file, 4 year college degree, degree, Bachelor, and 4 year degree are equivalent to each other.

## Not using UI; just using modules
### Step 1: Add Resumes
-	Folder: `Resume-Parser-master-new/data/input/resumes`
-   Change the job id that new resumes are tied to, it currently uses the job id and uses all resumes for demo purposes:
    -   In line 34 of `Resume-Parser-master-new/bin/main.py `   
    *----------or----------*    
    -   In the csv file directly: `Resume-Parser-master-new/data/output/resume_summary.csv`
-   *Note: we currently assume all resumes apply to the job id currently ranking candidates, this is due to our small amount of resumes we have. This would need to be updated if there is a large amount of resumes.*
### Step 2: Add job descriptions to create the ideal candidate
-	File: `Resume-Parser-JOBS/data/job_descriptions.csv`
-	The first column is a unique code (ReqID) and the second is the job description (text)
### Step 3: Rank candidates
-	File: `pipeline.py`
-	Update lines 109-112
    -   Select jobID (*str*) to rank candidates: `jobID`   
    -   Select number (*int*) of top candidates to show: `Num`   
    -   Enter file path (*str*) of the repo if on your local machine: `file_path`
    -   Choose True or False (*bool*) to look all candidates: `all_resumes`

## Using UI
### Steps:
-   Add new resumes, per steps above
-   Update line 7 of `Resume-Screening-and-Selection/jobAPP.py` to reflect location of repo
-   Run `Resume-Screening-and-Selection/khc_home_edited/app.py`
-   Go to http://127.0.0.1:5000/ to interact with UI
-   *Note: it will take a while to run a new job since it is parsing all the new resumes that apply to it*


