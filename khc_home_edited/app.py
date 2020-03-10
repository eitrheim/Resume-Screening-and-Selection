import pandas as pd
import numpy as np
import sys
import csv
from flask import Flask, request, jsonify, render_template
root_path = '/Users/matthewechols/PycharmProjects/Resume-Screening-and-Selection/'
sys.path.append(root_path)
import jobAPP
import pipeline
app = Flask(__name__, template_folder='templates')
# Jobs_path = "TestDescriptions.csv"
# data = pd.read_csv(Jobs_path, delimiter=',')
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
    results = pipeline.pipeline(jobID, numApp, allCandidates)
    responseData = []
    for i in range(numApp):
        result = {
            "Rank": i + 1,
            "Candidate ID": results.iloc[i][0],
            "Similarity": results.iloc[i][1]
        }
        responseData.append(result)
    return render_template('candidateResults.html', candidates=responseData)

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