import pandas as pd
from config import Config
from dataset import create_dataset

def main():
    
    config = Config()
    
    print("\nCreating dataset...")
    dataset = create_dataset(
        data=pd.DataFrame({'col1': [1]}),
        name=config.dataset_config["name"],
        target_column=config.dataset_config["target_column"],
        categorical_columns=config.dataset_config["categorical_columns"]
    )
    
    print("✓ Dataset created successfully")
    print(f"Dataset name: {dataset.name}")
    print(f"Dataset shape: {dataset.df.shape}")


if __name__ == "__main__":
    main()
