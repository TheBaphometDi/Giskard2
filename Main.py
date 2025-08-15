from config import GiskardConfig
from dataset import LiteraryTextDataset

def main():
    print("Starting Literary Text Testing Program for Gemini...")
    print(f"Book: {GiskardConfig().literary_config['book_title']}")
    print(f"Author: {GiskardConfig().literary_config['author']}")
    
    config = GiskardConfig()
    config.create_directories()
    
    print(f"\nCreated directories:")
    print(f"  - Texts: {config.get_texts_path()}")
    print(f"  - Datasets: {config.get_dataset_path()}")
    print(f"  - Models: {config.get_model_path()}")
    print(f"  - Results: {config.get_results_path()}")
    print(f"  - Logs: {config.get_logs_path()}")
    
    literary_dataset = LiteraryTextDataset(config)
    
    print("\nCreating sample literary text...")
    sample_text = """
    В час жаркого весеннего заката на Патриарших прудах появились два гражданина. Первый из них, одетый в летнюю серенькую пару, был маленького роста, упитан, лыс, свою приличную шляпу пирожком нес в руке, а на хорошо выбритом лице его помещались сверхъестественных размеров очки в черной роговой оправе. Второй – плечистый, рыжеватый, вихрастый молодой человек в заломленной на затылок клетчатой кепке – был в ковбойке, жеваных белых брюках и в черных тапочках.
    
    Первый был не кто иной, как Михаил Александрович Берлиоз, председатель правления одной из крупнейших московских литературных ассоциаций, сокращенно именуемой МАССОЛИТ, и редактор толстого художественного журнала, а молодой спутник его – поэт Иван Николаевич Понырев, пишущий под псевдонимом Бездомный.
    
    Попав в тень чуть зеленеющих лип, писатели первым делом бросились к пестро раскрашенной будочке с надписью «Пиво и воды».
    """.strip()
    
    text_path = config.get_texts_path() / "master_margarita_sample.txt"
    with open(text_path, 'w', encoding='utf-8') as f:
        f.write(sample_text)
    print(f"Sample text saved to: {text_path}")
    
    print("\nLoading literary text...")
    loaded_text = literary_dataset.load_literary_text(str(text_path))
    print(f"✓ Literary text loaded successfully ({len(loaded_text)} characters)")
    
    print("\nExtracting excerpts from text...")
    excerpts = literary_dataset.extract_excerpts(loaded_text, excerpt_length=200)
    print(f"✓ Extracted {len(excerpts)} excerpts from text")
    
    for i, excerpt in enumerate(excerpts[:3]):
        print(f"  Excerpt {i+1}: {excerpt['excerpt_length']} chars")
    
    print("\nCreating literary dataset...")
    dataset_df = literary_dataset.create_literary_dataset(excerpts)
    print(f"✓ Literary dataset created with {len(dataset_df)} rows")
    
    dataset_path = config.get_dataset_path() / "literary_dataset.csv"
    dataset_df.to_csv(dataset_path, index=False)
    print(f"✓ Dataset saved to: {dataset_path}")
    
    print("\nLoading data into Giskard...")
    dataset = literary_dataset.create_giskard_dataset(dataset_df)
    print("✓ Giskard dataset created successfully")
    
    dataset_info = literary_dataset.get_literary_dataset_info()
    print(f"Dataset info: {dataset_info['shape'][0]} rows, {dataset_info['shape'][1]} columns")
    
    print("\nValidating literary dataset...")
    validation_results = literary_dataset.validate_literary_dataset(dataset)
    print(f"✓ Dataset validation completed")
    print(f"  - Issues found: {validation_results.get('issues_count', 0)}")
    
    if validation_results.get('issues'):
        print("  - Issues:")
        for issue in validation_results['issues'][:3]:
            print(f"    * {issue}")
        if len(validation_results['issues']) > 3:
            print(f"    ... and {len(validation_results['issues']) - 3} more")
    
    print("\nCreating Gemini model...")
    gemini_model = literary_dataset.create_gemini_model("gemini-pro")
    print("✓ Gemini model created successfully")
    
    print("\nGenerating literary test suite...")
    test_suite = literary_dataset.generate_literary_tests(dataset, gemini_model)
    print("✓ Literary test suite generated successfully")
    
    print("\nRunning literary tests...")
    test_results = literary_dataset.run_literary_tests(test_suite)
    print("✓ Literary tests completed successfully")
    print(f"  - Total tests: {test_results['total_tests']}")
    print(f"  - Passed: {test_results['passed_tests']}")
    print(f"  - Failed: {test_results['failed_tests']}")
    
    results_filename = "literary_test_results.json"
    literary_dataset.save_literary_results(test_results, results_filename)
    print(f"✓ Results saved to: {results_filename}")
    
    print("\nProgram completed successfully!")
    print(f"Check the following directories for outputs:")
    print(f"  - Texts: {config.get_texts_path()}")
    print(f"  - Results: {config.get_results_path()}")
    print(f"  - Logs: {config.get_logs_path()}")

if __name__ == "__main__":
    main()
