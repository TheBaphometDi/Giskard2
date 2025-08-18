import os
from pathlib import Path

class Config:
    def __init__(self):
        self.project_name = "dataset_creator"
        self.dataset_path = Path("datasets")
        self.data_path = Path("data")
        
        self.dataset_config = {
            "name": "my_dataset",
            "target_column": None,
            "categorical_columns": []
        }
    
    def create_directories(self):
        for path in [self.dataset_path, self.data_path]:
            path.mkdir(exist_ok=True)
    
    def get_dataset_path(self):
        return self.dataset_path
    
    def get_data_path(self):
        return self.data_path
