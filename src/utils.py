import sys, os
import  numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import pickle
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)

    except FileNotFoundError as e:
        logging.error(f'File not found: {file_path}')
        raise CustomException(e)
    except Exception as e:
        logging.error('Error occurred in saving pickle file')
        raise CustomException(e)
    
def evaluate_model(x_train, y_train, x_test, y_test, models):
    try :
        report = {}
        for model_name, model in models.items():
            # Train the model
            model.fit(x_train, y_train)

            # Predict the test data
            y_test_pred = model.predict(x_test)

            # Get R2 score
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report
    except ValueError as e:
        logging.error('Value Error occurred in evaluating model')
        raise CustomException(e)
    except Exception as e:
        logging.error('Error occurred in evaluating model')
        raise CustomException(e)
    
def load_object(file_path):
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e :
        logging.error('Error occurred in loading pickle file')
        raise CustomException(e)