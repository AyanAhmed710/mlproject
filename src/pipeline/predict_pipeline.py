import pandas as pd
import numpy as np
import sys 
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
       try: 
        model_path='artifact/model_trainer.pkl'
        preprocessor_path='artifact/preprocessor.pkl'
        model=load_object(file_path=model_path)
        preprocessor=load_object(file_path=preprocessor_path)
        data_scaled=preprocessor.transform(features)
        preds=model.predict(data_scaled)
        return preds
       
       except Exception as e:
           raise CustomException(e,sys)
    



class CustomData:
    def __init__(self,gender,race_ethnicity ,parental_level_of_education,lunch,test_preparation_score,reading_score,writing_score):
        self.gender=gender
        self.race_ethnicity=race_ethnicity
        self.parental_level_of_education=parental_level_of_education
        self.lunch=lunch
        self.test_preparation_score=test_preparation_score
        self.reading_score=reading_score
        self.writing_score=writing_score

    
    def get_data_as_df(self):
        try:
            custom_input_data_input = {
                "gender": [self.gender],
                "race/ethnicity": [self.race_ethnicity],
                "parental level of education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test preparation course": [self.test_preparation_score],
                "reading score": [self.reading_score],
                "writing score": [self.writing_score]
            }
            return pd.DataFrame(custom_input_data_input)
        except Exception as e:
            raise CustomException(e,sys)