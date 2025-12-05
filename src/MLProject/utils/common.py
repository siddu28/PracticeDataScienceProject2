import os
from pathlib import Path
from  ensure import ensure_annotations
from pandas import DataFrame


@ensure_annotations
def create_directories(path:Path):

    os.makedirs(os.path.dirname(path),exist_ok=True)


@ensure_annotations
def save_csv(path:Path,df:DataFrame):

    create_directories(path)
    df.to_csv(path,index=False,header=True)


    