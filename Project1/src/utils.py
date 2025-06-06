import pandas as pd
import pickle
from src.exception import CustomException
from src.logger import logging
import os
import sys
from sklearn.metrics import r2_score
def save_object(file_path,object):
    try:
        directory=os.path.dirname(file_path)
        os.makedirs(directory,exist_ok=True)
        with open(file_path,'wb') as file:
            pickle.dump(object,file)
        logging.info("File Saved Successfully at path: %s", file_path)
        
        
    except Exception as e:
        raise CustomException(e,sys)
def evaluate_model(X_train,y_train,X_test,y_test,models):
    model_report={}
    for model_name,model in models.items():
        model.fit(X_train,y_train)
        y_pred=model.predict(X_test)
        r2_square=r2_score(y_test,y_pred)
        model_report[model_name]=r2_square  # Store the score, not the function
        logging.info(f"{model_name} R2 Score: {r2_square}")
    return model_report
def load_obj(file_path):
    try:
        with open(file_path,'rb') as file:
            obj=pickle.load(file)
        return obj
    except Exception as e:
        raise CustomException(e,sys)