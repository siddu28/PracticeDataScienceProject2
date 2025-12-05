import os
import sys

from src.MLProject.exception import CustomException
from src.MLProject import logger
import pandas as pd
from sklearn.model_selection import train_test_split
from src.MLProject.entity.config_entity import DataIngestionConfig
from src.MLProject.utils.common import create_directories,save_csv

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logger.info("Entered the Data Ingestion....")
        try:
            df = pd.read_csv('research\data\stud.csv')
            logger.info('Read the dataset as dataframe....')

            create_directories(self.ingestion_config.train_data_path)
            save_csv(self.ingestion_config.raw_data_path,df)

            logger.info("train test split is initiated....")
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)

            save_csv(self.ingestion_config.train_data_path,train_set)
            save_csv(self.ingestion_config.test_data_path,test_set)

            logger.info("Ingestion of data is completed!!!!")

            return DataIngestionConfig



        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()