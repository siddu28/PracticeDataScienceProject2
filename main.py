from src.MLProject.components.data_ingestion import DataIngestion
from src.MLProject.components.data_transformation import DataTransformation

obj = DataIngestion()
train_data,test_data = obj.initiate_data_ingestion()

obj_transformation = DataTransformation()
obj_transformation.initiate_data_tranformation(train_data,test_data)