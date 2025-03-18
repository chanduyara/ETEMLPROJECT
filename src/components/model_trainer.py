import numpy
import pandas as pd
import os
import sys
from dataclasses import dataclass
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("split training and test data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
               "Decision Tree": DecisionTreeRegressor(),
               "Random Forest": RandomForestRegressor(),
               "Gradient Boosting": GradientBoostingRegressor(),
               'Linear Regression':LinearRegression(),
               "CatBoosting Regressor": CatBoostRegressor(),
               "AdaBoost Regressor": AdaBoostRegressor(),
               "KNeighbors Regressor":KNeighborsRegressor(),
               "xgboost": XGBRegressor(),
               "SVM": SVR(),
               
            }

            params={
                "Decision Tree": {
                    "max_depth": [3, 5, 10],
                    "min_samples_split": [2, 5, 10]
                    
                },
                "Random Forest":{
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "KNeighbors Regressor":{
                    'n_neighbors': [5,7,9,11]
                },

                "xgboost":{
                    "n_estimators": [100, 500, 1000],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 6, 9],
                    "min_child_weight": [1, 3, 5],
                    "gamma": [0, 0.1, 0.2],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0]
                },

                "SVM":{
                    "C": [0.1, 1, 10, 100],
                    "kernel": ["linear", "rbf", "poly", "sigmoid"],
                    "gamma": ["scale", "auto"],
                    "degree": [2, 3, 4],
                }
                
            }

            model_report=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                        models=models,param=params)
            
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )

            predicted=best_model.predict(X_test)
            r2_square=r2_score(y_test,predicted)

            return r2_square

        except Exception as e:
            raise CustomException(e,sys)
            