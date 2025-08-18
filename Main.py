import pandas as pd
from config import Config
from dataset import create_dataset

def main():
    
    config = Config()
    
    print("\nCreating dataset...")
    dataset = create_dataset(
        data=pd.DataFrame({
            'text': ['Пример текста 1', 'Пример текста 2', 'Пример текста 3'],
            'value': [10, 20, 30],
            'category': ['A', 'B', 'A']
        }),
        name=config.dataset_config["name"],
        target_column=config.dataset_config["target_column"],
        categorical_columns=config.dataset_config["categorical_columns"]
    )
    
    print("✓ Dataset created successfully")
    print(f"Dataset name: {dataset.name}")
    print(f"Dataset shape: {dataset.df.shape}")
    print("\nDataset preview:")
    print(dataset.df.to_string(index=False))

if __name__ == "__main__":
    main()
