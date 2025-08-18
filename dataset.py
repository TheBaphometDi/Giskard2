import pandas as pd
from giskard.datasets import wrap_dataset

def create_dataset(data, name="dataset", target_column=None, categorical_columns=None):

    dataset = wrap_dataset(
        data,
        name=name,
        target=target_column,
        cat_columns=categorical_columns or []
    )
    
    return dataset
