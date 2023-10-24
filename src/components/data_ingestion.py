# Data Ingestion is used to ingest data from different sources
# aap kaha se data read kar rhe hai and train test split kar rhe

import os, sys
from logger import logging
from exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# initialize the data ingestion configuration

@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artifacts', 'train.csv')
    test_data_path  = os.path.join('artifacts', 'test.csv')
    raw_data_path  = os.path.join('artifacts', 'raw.csv')


# create a data ingestion class

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion method Started")

        try:
            df = pd.read_csv(os.path.join('notebook/data', 'gemstone.csv'))
            logging.info('Dataset read from pandas dataframe')

            # If the 'artifacts' folder is not present, create it
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            logging.info('Train test split')
            train_set, test_set = train_test_split(df, test_size=0.3, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info('Data Ingestion of data is completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error(f'Error occurred in data ingestion: {str(e)}')
            return None, None

