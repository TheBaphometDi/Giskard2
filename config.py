import os
import json
from pathlib import Path

class GiskardConfig:
    def __init__(self):
        self.project_name = "gemini_literary_testing"
        self.dataset_path = Path("datasets")
        self.model_path = Path("models")
        self.results_path = Path("results")
        self.logs_path = Path("logs")
        self.texts_path = Path("texts")
        
        self.load_api_keys()
        
        self.dataset_config = {
            "name": "master_margarita_dataset",
            "description": "Dataset for testing Gemini on Master and Margarita excerpts",
            "target_column": "correctness_score",
            "feature_columns": ["text_excerpt", "question", "expected_answer"],
            "categorical_columns": ["excerpt_type", "question_type"],
            "numerical_columns": ["excerpt_length", "question_complexity"]
        }
        
        self.model_config = {
            "name": "gemini_literary_model",
            "description": "Gemini model for literary text analysis",
            "model_type": "text_generation",
            "framework": "gemini"
        }
        
        self.test_config = {
            "test_suite_name": "literary_comprehension_tests",
            "test_suite_description": "Test suite for evaluating Gemini's understanding of literary texts"
        }
        
        self.literary_config = {
            "book_title": "Мастер и Маргарита",
            "author": "Михаил Булгаков",
            "excerpt_length_range": (100, 500),
            "question_types": ["comprehension", "analysis", "interpretation", "detail"],
            "scoring_criteria": ["accuracy", "completeness", "relevance", "insight"]
        }
    
    def load_api_keys(self):
        keys_file = Path("keys.json")
        if keys_file.exists():
            with open(keys_file, 'r') as f:
                keys = json.load(f)
                self.openai_api_key = keys.get("openai_api_key", "")
                self.gemini_api_key = keys.get("gemini_api_key", "")
        else:
            self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
            self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
    
    def create_directories(self):
        for path in [self.dataset_path, self.model_path, self.results_path, self.logs_path, self.texts_path]:
            path.mkdir(exist_ok=True)
    
    def get_dataset_path(self):
        return self.dataset_path
    
    def get_model_path(self):
        return self.model_path
    
    def get_results_path(self):
        return self.results_path
    
    def get_logs_path(self):
        return self.logs_path
    
    def get_texts_path(self):
        return self.texts_path
