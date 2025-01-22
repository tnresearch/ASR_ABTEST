from typing import List, Dict
import asyncio
from .processor import BenchmarkProcessor
import json
import os

class BatchRunner:
    def __init__(self):
        self.processor = BenchmarkProcessor()
        self.results_dir = "benchmark_results"
        os.makedirs(self.results_dir, exist_ok=True)

    async def run_batch(self, 
                       files: List[Dict],
                       model_id: str,
                       config: Dict = None) -> Dict:
        """
        Run batch processing on multiple files
        files: List of dicts with audio and reference paths
        """
        results = []
        aggregated_metrics = {
            "wer": 0.0,
            "processing_time": 0.0,
            "total_audio_duration": 0.0
        }

        for file_pair in files:
            result = await self.processor._process_single_file(
                file_pair,
                config or {"model_id": model_id}
            )
            results.append(result)
            
            # Update aggregated metrics
            aggregated_metrics["wer"] += result["wer"]
            aggregated_metrics["processing_time"] += result["duration"]
            # Add audio duration from metadata

        # Calculate averages
        num_files = len(files)
        aggregated_metrics["wer"] /= num_files
        aggregated_metrics["avg_processing_time"] = aggregated_metrics["processing_time"] / num_files

        return {
            "results": results,
            "aggregated_metrics": aggregated_metrics,
            "model_id": model_id,
            "num_files": num_files
        } 