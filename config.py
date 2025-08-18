import os
from pathlib import Path

class Config:
    def __init__(self):
        self.project_name = "dataset_creator"
        
        self.dataset_config = {
            "name": "my_dataset",
            "target_column": None,
            "categorical_columns": []
        }
