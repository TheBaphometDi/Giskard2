from config import Config
from dataset import create_dataset, prepare_data


def main():
    config = Config()

    print("\nCreating dataset...")
    text = config.load_text_data()
    df = prepare_data(text)

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