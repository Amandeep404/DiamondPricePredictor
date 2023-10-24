# Jo bhi model train hoga and evaluate hoga... wo iss file mein hoga
import sys, os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from exception import CustomException
from logger import logging
from utils import save_object
from dataclasses import dataclass
from utils import evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path =   os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_arr, test_arr):
        try:
            logging.info('Splitting dependent and independent variables from train and test data')
            x_train,y_train, x_test, y_test = (
                train_arr[:, :-1], train_arr[:, -1], test_arr[:, :-1], test_arr[:, -1]
            )

            models = {
                'LinearRegression' : LinearRegression(),
                'Ridge' : Ridge(),
                'Lasso' : Lasso(),
                'ElasticNet' : ElasticNet(),
                'DecisionTreeRegressor' : DecisionTreeRegressor()
            }

            model_report : dict = evaluate_model(x_train, y_train, x_test, y_test, models)
            print(model_report)
            print('\n')
            print('=='*35)
            logging.info(f'Model report : {model_report}')

            # To get the best model score from the dictionary
            best_model_name, best_model_score = max(model_report.items(), key=lambda item: item[1])


            best_model = models[best_model_name]

            print(f'Best Model found, Best model name : {best_model_name}, R2 score : {best_model_score}')
            print('\n', '=='*35)
            logging.info(f'Best Model found, Best model name : {best_model_name}, R2 score : {best_model_score}')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

        except Exception as e:
            logging.info('Error occurred at model training')
            raise CustomException(e, sys)



