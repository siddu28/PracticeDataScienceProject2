from src.MLProject.components.data_ingestion import DataIngestion
from src.MLProject.components.data_transformation import DataTransformation
from src.MLProject.components.model_trainer import ModelTrainer

obj = DataIngestion()
train_data,test_data = obj.initiate_data_ingestion()

obj_transformation = DataTransformation()
train_arr,test_arr,_ = obj_transformation.initiate_data_tranformation(train_data,test_data)

modeltrainer = ModelTrainer()
print(modeltrainer.initiate_model_trainer(train_arr,test_arr))