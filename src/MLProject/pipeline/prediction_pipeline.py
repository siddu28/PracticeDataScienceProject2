import sys
import pandas as pd
from src.MLProject.exception import CustomException
from src.MLProject.utils.common import load_object
from src.MLProject import logger

class PredictPipeline:
    def __init__(self):
        self.model_path = 'artifact/model.pkl'
        self.preprocessor_path = 'artifact/preprocessor.pkl'

    def predict(self, features):
        try:
            logger.info("Loading model and preprocessor")
            model = load_object(file_path=self.model_path)
            preprocessor = load_object(file_path=self.preprocessor_path)
            
            logger.info(f"Features before transformation: {features}")

            # Check for missing values in features
            if features.isnull().values.any():
                raise ValueError("Input features contain missing values.")

            # Ensure all columns expected by the preprocessor are present
            required_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course', 'reading_score', 'writing_score']
            missing_columns = set(required_columns) - set(features.columns)
            if missing_columns:
                raise ValueError(f"Missing columns in input data: {missing_columns}")

            data_scaled = preprocessor.transform(features)
            logger.info(f"Features after transformation: {data_scaled}")

            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(
            self,
            gender: str,
            race_ethnicity: str,
            parental_level_of_education: str,
            lunch: str,
            test_preparation_course: str,
            reading_score: int,
            writing_score: int
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score]
            }

            df = pd.DataFrame(custom_data_input_dict)
            
            logger.info(f"Custom data as dataframe: {df}")
            
            return df
        
        except Exception as e:
            raise CustomException(e, sys)