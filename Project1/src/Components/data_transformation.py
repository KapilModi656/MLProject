import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    def get_data_transformer_object(self, raw_path, target_column_name):
        try:
            df = pd.read_csv(raw_path)
            # Normalize column names to match train/test normalization
            df.columns = [col.strip().replace(' ', '_') for col in df.columns]
            # Drop the target column
            if target_column_name in df.columns:
                df = df.drop([target_column_name], axis=1)
            num_columns = [features for features in df if df[features].dtype != 'object']
            cat_columns = [features for features in df if df[features].dtype == 'object']
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='median')),
                ("scaler", StandardScaler())
            ])
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='most_frequent')),
                ("ohe", OneHotEncoder(handle_unknown='ignore')),
                ("scaler", StandardScaler(with_mean=False))
            ])
            logging.info("Column Transformer initiated")
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', num_pipeline, num_columns),
                    ('cat', cat_pipeline, cat_columns)
                ]
            )
            logging.info("Column Transformer Created Successfully")
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
    def initiate_data_transformation(self,train_path,test_path,target_column_name, raw_path):
        try:
            train=pd.read_csv(train_path)
            test=pd.read_csv(test_path)
            # Normalize column names to avoid issues with spaces/underscores
            train.columns = [col.strip().replace(' ', '_') for col in train.columns]
            test.columns = [col.strip().replace(' ', '_') for col in test.columns]
            target=target_column_name
            # Check if target column exists after normalization
            if target not in train.columns or target not in test.columns:
                raise CustomException(f"Target column '{target}' not found in train or test columns after normalization.", sys)
            logging.info("Train and test data read successfully")
            logging.info("Train Dataframe shape: %s",train.shape)
            logging.info("Test Dataframe shape: %s",test.shape)
            preprocessor = self.get_data_transformer_object(raw_path, target)
            X_train=train.drop([target],axis=1)
            X_test=test.drop([target],axis=1)
            y_train=train[target]
            y_test=test[target]
            X_train_processed=preprocessor.fit_transform(X_train)
            X_test_processed=preprocessor.transform(X_test)
            train_arr=np.c_[X_train_processed,np.array(y_train)]
            test_arr=np.c_[X_test_processed,np.array(y_test)]
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                object=preprocessor
            )
            return (train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path)
            
        except Exception as e:
            logging.error(f"Error in initiate data transformation: {str(e)}")
            raise CustomException(e,sys)