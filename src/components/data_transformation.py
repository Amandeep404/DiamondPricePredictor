# we do all the feature engineering in this file

from sklearn.impute import SimpleImputer # Imputation transformer for completing missing values.
from sklearn.preprocessing import OrdinalEncoder # Encode categorical features as an integer array.
from sklearn.preprocessing import StandardScaler # handle feature scaling

#pipelines
from sklearn.pipeline import Pipeline # Pipeline of transforms with a final estimator.
from sklearn.compose import ColumnTransformer # Applies transformers to columns of an array or pandas DataFrame.
import sys, os
from dataclasses import dataclass
import pandas as pd
import numpy as np
from logger import logging
from exception import CustomException
from utils import save_object

# Data Transformation Config
@dataclass
class DataTransformationConfig:
    preprocessor_obj_path = os.path.join('artifacts', 'preprocessor.pkl')




# Data ingestion config class
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info("Data Transformation initiated")

            # define which columns should be ordinal encoded and which should be scaled
            categorical_columns = ['cut', 'color', 'clarity']
            numerical_columns = ['carat', 'depth', 'table', 'x', 'y', 'z']

            cut_categories  = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

            logging.info('Pipeline Initiated')

            # Numerical pipeline
            numerical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            # Categorical pipeline
            categorical_pipeline = Pipeline(steps = 
            [
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ordinal', OrdinalEncoder(categories=[cut_categories, color_categories, clarity_categories])),
            ('scaler', StandardScaler())
            ] )

            preprocessor = ColumnTransformer([
            ('num_pipeline', numerical_pipeline, numerical_columns),
            ('cat_pipeline', categorical_pipeline, categorical_columns)
            ])

            logging.info('Pipeline Completed')
            #logging.info(preprocessor)
            return preprocessor
            
            
        except Exception as e:
            logging.info('Error occurred in data transformation')
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_data_path,test_data_path ):

        try:
            # read train and test data
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            logging.info('Train and Test data read from csv')

            logging.info('obtaining preprocessing object')
            preprocessing_obj = self.get_data_transformation_object()

            logging.info('Preprocessing transformation object obtained')

            target_col = 'price'
            drop_col = [target_col, 'Unnamed: 0']

            input_feature_train_df = train_df.drop(columns=drop_col, axis=1)
            target_feature_train_df = train_df[target_col]
           

            input_feature_test_df = test_df.drop(columns=drop_col, axis=1)
            target_feature_test_df = test_df[target_col]

            # Apply the transformation
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            logging.info('Applying preprocessing object on train and test data')

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # saving pickle file to artifacts folder
            save_object(
                file_path= self.data_transformation_config.preprocessor_obj_path,
                obj = preprocessing_obj
            )
            logging.info('Preprocessing pickle saved to artifacts folder')

            return(
                train_arr,
                test_arr,
                #self.data_transformation_config.preprocessor_obj_path
            )
            

        except Exception as e:
            logging.info('Error occurred initiating data transformation')
            raise CustomException(e, sys)
