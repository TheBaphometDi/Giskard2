import pandas as pd
from giskard.datasets import wrap_dataset
from config import Config

config = Config()


def create_dataset(data, name="dataset"):
    dataset = wrap_dataset(
        data,
        name=name
    )
    return dataset


def prepare_data(text):
    return pd.DataFrame({'text': [text]})


def load_text_data():
    data_path = config.dataset_config["file_path"]

    if data_path.is_dir():
        all_text = ""
        for file_path in data_path.glob("*.txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                all_text += f.read().strip() + "\n"
        return all_text.strip()
    else:
        with open(data_path, "r", encoding="utf-8") as f:
            return f.read().strip()