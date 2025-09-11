from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData
from src.pipeline.predict_pipeline import PredictPipeline

application=Flask(__name__)

app=application

#Route for Home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=="GET":
        return render_template('home.html')
    
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race'),
            parental_level_of_education=request.form.get('education'),
            lunch=request.form.get('lunch'),
            test_preparation_score=request.form.get('prep'),
            reading_score=request.form.get('reading'),
            writing_score=request.form.get('writing')
        )

        # Convert to DataFrame
        final_new_data = data.get_data_as_df()
        print(final_new_data)

        # Call prediction pipeline (if you have it)
        predict_pipeline = PredictPipeline()
        results= predict_pipeline.predict(final_new_data)

        return render_template('home.html',results=results[0])
    


if __name__=="__main__":
    app.run("0.0.0.0",debug=True)
   
