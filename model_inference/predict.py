import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from flask import Flask, request, render_template

app=Flask(__name__)

def load_model(model_path):
    return mlflow.sklearn.load_model(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict(model, data):
    if request.method == "POST":
        MALE = float(request.form['male'])
        AGE = float(request.form['age'])
        EDUCATION = float(request.form['education'])
        CURRENTSMOKER = float(request.form['currentSmoker'])
        CIGSPERDAY = float(request.form['cigsPerDay'])
        BPMEDS = float(request.form['BPMeds'])
        PREVALENTSTROKE = float(request.form['prevalentStroke'])
        PREVALENTHYP = float(request.form['prevalentHyp'])
        DIABETES = float(request.form['diabetes'])
        TOTCHOL = float(request.form['totChol'])
        SYSBP = float(request.form['sysBP'])
        DIABP = float(request.form['diaBP'])
        BMI = float(request.form['BMI'])
        HEARTRATE = float(request.form['heartRate'])
        GLUCOSE = float(request.form['glucose'])

        data = [[MALE, AGE, EDUCATION, CURRENTSMOKER, CIGSPERDAY,
                BPMEDS, PREVALENTSTROKE, PREVALENTHYP, DIABETES,
                TOTCHOL, SYSBP, DIABP, BMI, HEARTRATE, GLUCOSE]]
        
        result = model.predict(data)[0]

        if result==1:
            predictions = "You are prone to Heart Disease"
        else:
            predictions = "You are not prone to Heart Disease"

        return predictions
    
def parse_args():
    parser = argparse.ArgumentParser(description='Make predictions using a trained MLflow model.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the MLflow model')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    model_path = "runs:/<RUN_ID>/random_forest_model" 
    loaded_model = load_model(model_path)

    predictions = predict(loaded_model, predict.data)

    print(predictions)
