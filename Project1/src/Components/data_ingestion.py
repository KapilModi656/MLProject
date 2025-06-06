from src.exception import CustomException
from src.logger import logging
import sys
import os
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split
from .model_trainer import ModelTrainer
from .data_transformation import DataTransformation
@dataclass
class Data_Ingestion_Config:
    train_data_path=str(os.path.join('artifacts','train.csv'))
    test_data_path=str(os.path.join('artifacts','test.csv'))
    raw_data_path=str(os.path.join('artifacts','raw.csv'))
    
    
class Data_Ingestion(Data_Ingestion_Config):
    def __init__(self):
        self.ingestion_config=Data_Ingestion_Config()
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion start")
        try:
            df=pd.read_csv('src/notebook/data/stud.csv')
            logging.info('read the data from stud.csv')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("Raw Data File Completed")
            logging.info("Train Test Split Initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Train Test Split Completed")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.raw_data_path
            )
            
            
            
            
            
        except Exception as e:
            raise CustomException(e,sys)
        
        
        
if __name__ == "__main__":
    obj = Data_Ingestion()
    
    obj.initiate_data_ingestion()
    print("Data ingestion completed successfully.")
    model_trainer = ModelTrainer()
    train_path, test_path, raw_path = obj.initiate_data_ingestion()
    
    print("Model training completed successfully.")

    logging.info("Data transformation initiated")
    target="math_score"
    dataTransformation= DataTransformation()
    preprocessor = dataTransformation.get_data_transformer_object(raw_path=raw_path, target_column_name='math_score')
    train_array, test_array, _ = dataTransformation.initiate_data_transformation(train_path, test_path, 'math_score', raw_path=raw_path)

    print(model_trainer.initiate_model_trainer(train_array, test_array))
