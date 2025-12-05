import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer

from src.MLProject.exception import CustomException
from src.MLProject import logger
from src.MLProject.entity.config_entity import DataTransformationConfig
from src.MLProject.utils.common import save_csv,create_directories,save_object

class DataTransformation:
    def __init__(self):
        self.data_transformation = DataTransformationConfig()

    def get_data_transformer_object(self):
        logger.info("Entered Data transformation.....")
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ("scaler",StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore')),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            logger.info(f"numerical columns: {numerical_columns}")
            logger.info(f"categorical columns: {categorical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipelines", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor


        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_tranformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logger.info("Read train and test data completed")

            logger.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Fill missing values for object columns with the most frequent value
            for col in input_feature_train_df.columns:
                if input_feature_train_df[col].dtype == 'object':
                    input_feature_train_df[col].fillna(input_feature_train_df[col].mode()[0], inplace=True)
                    input_feature_test_df[col].fillna(input_feature_test_df[col].mode()[0], inplace=True)

            logger.info(
                "Applying preprocessing object on training dataframe and testing dataframe"
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logger.info("saving preprocessing object.")

            save_object(
                file_path=self.data_transformation.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)