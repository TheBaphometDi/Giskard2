from config import Config
from dataset import create_dataset, prepare_data, load_text_data


def main():

    print("\nCreating dataset...")
    text = load_text_data()
    df = prepare_data(text)

    dataset = create_dataset(
        data=df
    )

    print("✓ Dataset created successfully")
    print(f"Dataset name: {dataset.name}")
    print(f"Dataset shape: {dataset.df.shape}")
    print("\nDataset preview:")
    print(dataset.df.to_string(index=False))


if __name__ == "__main__":
    main()