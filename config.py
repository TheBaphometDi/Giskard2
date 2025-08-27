from pathlib import Path


class Config:
    def __init__(self):
        self.project_name = "dataset_creator"
        self.project_root = Path(__file__).parent

        self.dataset_config = {
            "name": "my_dataset",
            "file_path": self.project_root / "Data"
        }