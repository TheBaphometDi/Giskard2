import os
from pathlib import Path


class Config:
    def __init__(self):
        self.project_name = "dataset_creator"

        self.dataset_config = {
            "name": "my_dataset",
            "file_path": "master_margarita_excerpt.txt"
        }

    def load_text_data(self):
        with open(self.dataset_config["file_path"], "r", encoding="utf-8") as f:
            return f.read().strip()