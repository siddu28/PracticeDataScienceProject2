import os
from pathlib import Path
from  ensure import ensure_annotations
from pandas import DataFrame
import numpy as np
import pandas as pd
import dill


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