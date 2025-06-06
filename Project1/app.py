from flask import Flask,render_template,request
import numpy as np
import pandas as pd
from src.Pipeline.predict_pipeline import CustomData,PredictPipeline
from sklearn.preprocessing import StandardScaler

application=Flask(__name__)

app=application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=='GET':
        return render_template('Predict.html')
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
        )
        pred_df=data.get_data_as_Dataframe()
        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        return render_template('predict.html',results[0])
        
if __name__=='__main__':
    app.run(host='0.0.0.0',port=80)