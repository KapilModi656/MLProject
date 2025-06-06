import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import load_obj,evaluate_model,save_object
from catboost import CatBoostRegressor

@dataclass
class Model_trainer_config:
    model_path=str(os.path.join('artifacts','model.pkl'))
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=Model_trainer_config()
    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Splitting X_train,y_train,X_test,y_test")
            X_train,y_train,X_test,y_test=train_arr[:,:-1],train_arr[:,-1],test_arr[:,:-1],test_arr[:,-1]
            logging.info("Training and testing data completed")
            models={
                'RandomForestRegressor': RandomForestRegressor(),
                'GradientBoostingRegressor': GradientBoostingRegressor(),
                'LinearRegression': LinearRegression(),
                'KNeighborsRegressor': KNeighborsRegressor(),
                'DecisionTreeRegressor': DecisionTreeRegressor(),
                'XGBRegressor': XGBRegressor(),
                'CatBoostRegressor': CatBoostRegressor(verbose=0),
                'AdaBoostRegressor': AdaBoostRegressor()
            }
            
            model_report=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
            best_model_name=max(model_report,key=model_report.get)
            best_model_score=model_report[best_model_name]
            best_model=models[best_model_name]
            if best_model_score<0.6:
                raise CustomException("No best model found with good performance",sys)
            save_object(self.model_trainer_config.model_path,best_model)
            logging.info("Best Model is:",best_model_name,"With score of:",best_model_score)
            return best_model_name,best_model_score
            
            
        except Exception as e:
            raise CustomException(e,sys)
        
        
        