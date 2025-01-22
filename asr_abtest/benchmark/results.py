import json
from datetime import datetime
from typing import Dict, List
import os

class BenchmarkResults:
    def __init__(self, results_dir: str = "benchmark_results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def save_results(self, results: Dict) -> str:
        """Save benchmark results and return file path"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)
        
        return filepath

    def load_results(self, filepath: str) -> Dict:
        """Load benchmark results from file"""
        with open(filepath, "r") as f:
            return json.load(f) 