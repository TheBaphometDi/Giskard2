import pandas as pd
from config import Config
from dataset import create_dataset

def main():
    
    config = Config()
    
    print("\nCreating dataset...")
    with open("master_margarita_excerpt.txt", "r", encoding="utf-8") as f:
        text = f.read().strip()
    
    df = pd.DataFrame({
        'text': [text]
    })
    
    dataset = create_dataset(
        data=df,
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
