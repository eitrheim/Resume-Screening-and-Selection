import pandas as pd
import numpy as np
import csv
from flask import Flask, request, jsonify, render_template
import pickle



app = Flask(__name__, template_folder='templates')
# reg_model = pickle.load(open('reg_model.pkl','rb'))


@app.route('/')
def index():

    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    Jobs_path = "TestDescriptions.csv"
    data = pd.read_csv(Jobs_path, delimiter=',')
    features = [x for x in request.form.values()]
    new_row = dict(zip(data.columns, features))
    data = data.append(new_row, ignore_index=True)

    #save job description to csv for Smart Reader
    data.to_csv("TestDescriptions.csv", index=False)
    # array = np.asarray(features[3:])
    # prediction = reg_model.predict(array.reshape(1,-1))

    # output = int(prediction)
    return render_template('index.html')
    #return render_template('index.html', prediction_text=)


@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5001)