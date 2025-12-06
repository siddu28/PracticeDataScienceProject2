from dataclasses import dataclass
from pathlib import Path
import os


@dataclass
class DataIngestionConfig:
    train_data_path:str=Path(os.path.join('artifact','train.csv'))
    test_data_path:str = Path(os.path.join('artifact','test.csv'))
    raw_data_path:str = Path(os.path.join('artifact','data.csv'))

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifact','preprocessor.pkl')

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifact","model.pkl")