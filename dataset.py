import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
import logging
from pathlib import Path
import json
import giskard
from giskard import Dataset, Model, scan
from giskard.datasets import wrap_dataset
from giskard.models.base import BaseModel

class LiteraryTextDataset:
    def __init__(self, config):
        self.config = config
        self.dataset = None
        self.model = None
        self.test_suite = None
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.get_logs_path() / 'literary_testing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_literary_text(self, text_path: str) -> str:
        if text_path.endswith('.txt'):
            with open(text_path, 'r', encoding='utf-8') as f:
                text = f.read()
        elif text_path.endswith('.json'):
            with open(text_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                text = data.get('text', '')
        else:
            raise ValueError("Unsupported text file format")
        
        self.logger.info(f"Loaded literary text with length: {len(text)} characters")
        return text
    
    def extract_excerpts(self, text: str, excerpt_length: int = 300) -> List[Dict[str, Any]]:
        words = text.split()
        excerpts = []
        
        for i in range(0, len(words), excerpt_length // 2):
            excerpt_words = words[i:i + excerpt_length]
            if len(excerpt_words) >= excerpt_length // 2:
                excerpt_text = ' '.join(excerpt_words)
                excerpt_data = {
                    'text_excerpt': excerpt_text,
                    'excerpt_length': len(excerpt_text),
                    'excerpt_type': 'narrative',
                    'start_position': i,
                    'end_position': min(i + excerpt_length, len(words))
                }
                excerpts.append(excerpt_data)
        
        self.logger.info(f"Extracted {len(excerpts)} excerpts from text")
        return excerpts
    
    def generate_questions_for_excerpt(self, excerpt: str, question_type: str) -> List[Dict[str, Any]]:
        questions = []
        
        if question_type == "comprehension":
            questions.append({
                'question': f"О чем говорится в данном отрывке?",
                'question_type': 'comprehension',
                'question_complexity': 1,
                'expected_answer': 'general_summary'
            })
        
        elif question_type == "analysis":
            questions.append({
                'question': f"Какие литературные приемы используются в этом отрывке?",
                'question_type': 'analysis',
                'question_complexity': 2,
                'expected_answer': 'literary_devices'
            })
        
        elif question_type == "interpretation":
            questions.append({
                'question': f"Как вы понимаете смысл этого отрывка?",
                'question_type': 'interpretation',
                'question_complexity': 3,
                'expected_answer': 'personal_interpretation'
            })
        
        elif question_type == "detail":
            questions.append({
                'question': f"Какие конкретные детали упоминаются в отрывке?",
                'question_type': 'detail',
                'question_complexity': 1,
                'expected_answer': 'specific_details'
            })
        
        return questions
    
    def create_literary_dataset(self, excerpts: List[Dict[str, Any]]) -> pd.DataFrame:
        dataset_rows = []
        
        for excerpt in excerpts:
            for question_type in self.config.literary_config["question_types"]:
                questions = self.generate_questions_for_excerpt(excerpt['text_excerpt'], question_type)
                
                for question in questions:
                    row = {
                        'text_excerpt': excerpt['text_excerpt'],
                        'excerpt_length': excerpt['excerpt_length'],
                        'excerpt_type': excerpt['excerpt_type'],
                        'question': question['question'],
                        'question_type': question['question_type'],
                        'question_complexity': question['question_complexity'],
                        'expected_answer': question['expected_answer'],
                        'correctness_score': None
                    }
                    dataset_rows.append(row)
        
        dataset_df = pd.DataFrame(dataset_rows)
        self.logger.info(f"Created literary dataset with {len(dataset_df)} rows")
        return dataset_df
    
    def create_giskard_dataset(self, data: pd.DataFrame) -> Dataset:
        dataset = wrap_dataset(
            data,
            name=self.config.dataset_config["name"],
            target=self.config.dataset_config["target_column"],
            cat_columns=self.config.dataset_config["categorical_columns"]
        )
        
        self.dataset = dataset
        self.logger.info(f"Created Giskard dataset: {dataset.name}")
        return dataset
    
    def validate_literary_dataset(self, dataset: Dataset) -> Dict[str, Any]:
        validation_results = {}
        
        scan_result = scan(dataset)
        validation_results["scan_results"] = scan_result
        
        if hasattr(scan_result, 'issues'):
            validation_results["issues_count"] = len(scan_result.issues)
            validation_results["issues"] = [str(issue) for issue in scan_result.issues]
        else:
            validation_results["issues_count"] = 0
            validation_results["issues"] = []
        
        return validation_results
    
    def create_gemini_model(self, model_name: str = "gemini-pro") -> Model:
        class GeminiModel(BaseModel):
            def __init__(self, model_name: str):
                super().__init__(
                    model_type="text_generation",
                    name=model_name,
                    feature_names=["text_excerpt", "question"]
                )
                self.model_name = model_name
            
            def predict(self, data):
                return [f"Generated answer for: {row['question']}" for _, row in data.iterrows()]
        
        model = GeminiModel(model_name)
        self.model = model
        self.logger.info(f"Created Gemini model: {model.name}")
        return model
    
    def generate_literary_tests(self, dataset: Dataset, model: Model) -> Any:
        test_suite = giskard.Suite(
            name=self.config.test_config["test_suite_name"]
        )
        
        self.test_suite = test_suite
        self.logger.info(f"Generated literary test suite: {test_suite.name}")
        return test_suite
    
    def run_literary_tests(self, test_suite: Any) -> Dict[str, Any]:
        if test_suite is None:
            raise ValueError("Test suite is not initialized")
        
        test_results = test_suite.run()
        
        results_summary = {
            "total_tests": len(test_results),
            "passed_tests": sum(1 for result in test_results if result.passed),
            "failed_tests": sum(1 for result in test_results if not result.passed),
            "test_details": []
        }
        
        for result in test_results:
            test_details = {
                "test_name": result.test_name,
                "passed": result.passed,
                "message": result.message if hasattr(result, 'message') else ""
            }
            results_summary["test_details"].append(test_details)
        
        self.logger.info(f"Literary tests completed: {results_summary['passed_tests']}/{results_summary['total_tests']} passed")
        return results_summary
    
    def save_literary_results(self, results: Dict[str, Any], filename: str):
        results_path = self.config.get_results_path() / filename
        
        if filename.endswith('.json'):
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        elif filename.endswith('.yaml') or filename.endswith('.yml'):
            import yaml
            with open(results_path, 'w', encoding='utf-8') as f:
                yaml.dump(results, f, default_flow_style=False, allow_unicode=True)
        else:
            with open(results_path, 'w', encoding='utf-8') as f:
                for key, value in results.items():
                    f.write(f"{key}: {value}\n")
        
        self.logger.info(f"Literary results saved to: {results_path}")
    
    def get_literary_dataset_info(self) -> Dict[str, Any]:
        if self.dataset is None:
            return {"error": "Dataset not initialized"}
        
        info = {
            "name": self.dataset.name,
            "shape": self.dataset.df.shape,
            "columns": list(self.dataset.df.columns),
            "target_column": self.dataset.target,
            "categorical_columns": self.dataset.cat_columns if hasattr(self.dataset, 'cat_columns') else [],
            "numerical_columns": self.dataset.num_columns if hasattr(self.dataset, 'num_columns') else []
        }
        
        return info
    
    def get_gemini_model_info(self) -> Dict[str, Any]:
        if self.model is None:
            return {"error": "Model not initialized"}
        
        info = {
            "name": self.model.name,
            "model_type": self.model.model_type,
            "feature_names": self.model.feature_names,
            "framework": self.config.model_config["framework"]
        }
        
        return info
