import pandas as pd
from giskard.datasets import wrap_dataset

def create_dataset(data, name="dataset"):
    dataset = wrap_dataset(
        data,
        name=name
    )
    return dataset

def prepare_data(text):

    return pd.DataFrame({'text': [text]})