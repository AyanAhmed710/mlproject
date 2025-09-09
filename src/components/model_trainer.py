from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object,evaluate_model
import os
import sys
@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artificat','model_trainer.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Spliting training and test input data")
            X_train, Y_train, X_test, Y_test = (
            train_array[:, :-1],   # all rows, all columns except last
            train_array[:, -1],    # all rows, last column
            test_array[:, :-1],
            test_array[:, -1]
             )


            models={
                    'LinearRegression':LinearRegression(),
                    'Lasso':Lasso(),
                    'Ridge':Ridge(),
                    'KNeighborsRegressor':KNeighborsRegressor(),
                    'Decison_tree':DecisionTreeRegressor(),
                    'Random_Forest':RandomForestRegressor(),
                    'SVR':SVR(),
                    'XGBoost':XGBRegressor(),
                    'CatBoost':CatBoostRegressor(),
                    'AdaBoostRegressor':AdaBoostRegressor()
                      }
            
            model_report:dict=evaluate_model(X_train=X_train,Y_train=Y_train,X_test=X_test,Y_test=Y_test,models=models)

            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            
            logging.info("Best model found for training and testing")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            return r2_score(Y_test,predicted)
        except Exception as e:
            raise CustomException(e,sys)

            


