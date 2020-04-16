from flask import Flask, request, jsonify, render_template
import jobAPP
import pipeline


root_path = jobAPP.root_path
app = Flask(__name__, template_folder='templates')
@app.route('/')
def index():
    ids = jobAPP.get_Index()
    return render_template('index.html', ids=ids)


@app.route('/jobs/<string:jobId>/description')
def getJobDescription(jobId):
    jobDescription = jobAPP.get_Job(jobId)
    return jobDescription


@app.route('/jobs/<string:jobID>/candidates')
def predict(jobID):
    numApp = int(request.args["numApp"])
    allCandidates = bool(request.args["numApp"])

    results, jd = pipeline.pipeline(jobID, numApp, root_file_path=root_path, all_resumes=allCandidates)
    jd = jd[0].encode('ascii', errors='ignore').decode('ascii')
    # jd = 'Job Description: ' + jd[:300] + '...'
    responseData = []
    for i in range(numApp):
        result = {
            "Rank": i + 1,
            "Candidate ID": results.iloc[i][0],
            "Similarity": round(results.iloc[i][1], 4)
        }
        responseData.append(result)
    return render_template('candidateResults.html', candidates=responseData, job_description=jd)

@app.route('/jobs/', methods=["POST"])
def createNewJob():
    if (request.is_json):
        jobData = request.get_json(silent=True)
        print("jobData: ")
        print(jobData)
        jobID = jobData["jobID"]
        jobDescription = jobData["jobDescription"]
        jobAPP.set_Job(jobID, jobDescription)

    return "200"


if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
