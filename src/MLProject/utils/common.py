import os
from pathlib import Path
from  ensure import ensure_annotations
from pandas import DataFrame
import numpy as np
import pandas as pd
import dill
from src.MLProject.exception import CustomException
from sklearn.metrics import r2_score
import sys
from sklearn.model_selection import GridSearchCV


@ensure_annotations
def create_directories(path:Path):

    os.makedirs(os.path.dirname(path),exist_ok=True)


@ensure_annotations
def save_csv(path:Path,df:DataFrame):

    create_directories(path)
    df.to_csv(path,index=False,header=True)


@ensure_annotations
def save_object(file_path,obj):
    os.makedirs(os.path.dirname(file_path),exist_ok=True)

    with open(file_path,"wb") as file_obj:
        dill.dump(obj,file_obj)



def evaluate_model(x_train,y_train,x_test,y_test,models,param):
    try:
        report={}

        for i in range(len(list(models))):
            model=list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs=GridSearchCV(model,para,cv=3)
            gs.fit(x_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(x_train,y_train)

            # model.fit(x_train,y_train)

            y_train_pred=model.predict(x_train)

            y_test_pred=model.predict(x_test)

            train_model_score=r2_score(y_train,y_train_pred)
            test_model_score=r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]]=test_model_score

        return report
    
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
        try:
            with open(file_path,'rb') as file_obj:
                return dill.load(file_obj)
            
        except Exception as e:
            raise CustomException(e,sys)