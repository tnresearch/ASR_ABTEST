import asyncio
import json
import os
from datetime import datetime
from typing import List, Dict, Any
from uuid import uuid4
import time
from .evaluator import WERCalculator
import torch
from transformers import pipeline, AutoTokenizer
import shutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BenchmarkProcessor:
    def __init__(self):
        self.active_benchmarks: Dict[str, Dict] = {}
        self.results_dir = "benchmark_results"
        self.temp_dir = "temp_audio_files"
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        self.wer_calculator = WERCalculator()
        self.transcriber = None
        self.current_model_id = None
        logger.info("BenchmarkProcessor initialized")
    
    def __del__(self):
        """Cleanup temporary files on shutdown"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def load_model(self, model_id: str):
        """Load model if needed"""
        # Use default model if none specified
        model_id = model_id or "openai/whisper-small"
        logger.info(f"Using model: {model_id}")
        
        if model_id != self.current_model_id:
            print(f"Loading model: {model_id}")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
            
            self.transcriber = pipeline("automatic-speech-recognition", 
                                      model=model_id,
                                      tokenizer=tokenizer,
                                      chunk_length_s=30,
                                      return_timestamps="word",
                                      device=device)
            self.current_model_id = model_id
            print(f"Model loaded successfully: {model_id}")
        return self.transcriber
    
    async def start_benchmark(self, file_contents: List[Dict], config: Dict) -> str:
        """Start a new benchmark process"""
        benchmark_id = str(uuid4())
        self.active_benchmarks[benchmark_id] = {
            "status": "running",
            "progress": 0,
            "current_file": None,
            "total_files": len(file_contents),
            "results": [],
            "config": config,
            "start_time": datetime.now().isoformat()
        }
        
        # Start processing in background
        asyncio.create_task(self._process_files(benchmark_id, file_contents))
        
        return benchmark_id
    
    def get_status(self, benchmark_id: str) -> Dict:
        """Get current status of a benchmark process"""
        if benchmark_id not in self.active_benchmarks:
            raise KeyError(f"Benchmark {benchmark_id} not found")
        return self.active_benchmarks[benchmark_id]
    
    def stop_benchmark(self, benchmark_id: str) -> None:
        """Stop a running benchmark process"""
        if benchmark_id in self.active_benchmarks:
            self.active_benchmarks[benchmark_id]["status"] = "stopped"
    
    async def _process_files(self, benchmark_id: str, file_contents: List[Dict]) -> None:
        """Process all files in the benchmark"""
        benchmark = self.active_benchmarks[benchmark_id]
        total_files = len(file_contents)
        
        for i, file_pair in enumerate(file_contents):
            if benchmark["status"] == "stopped":
                break
                
            benchmark["current_file"] = file_pair["audio"]["filename"]
            benchmark["progress"] = int((i / total_files) * 100)
            
            try:
                result = await self._process_single_file(file_pair, benchmark["config"])
                benchmark["results"].append(result)
            except Exception as e:
                print(f"Error processing {file_pair['audio']['filename']}: {str(e)}")
                benchmark["results"].append({
                    "file": file_pair["audio"]["filename"],
                    "status": "error",
                    "error": str(e)
                })
        
        # Save final results
        benchmark["status"] = "completed"
        benchmark["progress"] = 100
        benchmark["current_file"] = None
        benchmark["end_time"] = datetime.now().isoformat()
        
        # Save to file
        self._save_results(benchmark_id)
    
    async def _process_single_file(self, file_pair: Dict, config: Dict) -> Dict:
        """Process a single file pair and evaluate results"""
        start_time = time.time()
        temp_audio = None
        logger.info(f"Starting to process file: {file_pair['audio']['filename']}")
        
        try:
            logger.info(f"File object details - audio_file: {type(file_pair['audio'])}, truth_file: {type(file_pair['truth'])}")
            logger.info(f"File attributes - content size: {len(file_pair['audio']['content'])}")
            
            # Create unique temp file path
            temp_audio = os.path.join(self.temp_dir, f"temp_audio_{str(uuid4())}_{file_pair['audio']['filename']}")
            logger.info(f"Created temp path: {temp_audio}")
        
            # Save audio file content
            with open(temp_audio, "wb") as f:
                f.write(file_pair["audio"]["content"])
            logger.info("File copied successfully")
        
            # Process ground truth content
            logger.info("Processing ground truth content...")
            truth_content = file_pair['truth']['content']
            logger.info(f"Ground truth content length: {len(truth_content)}")
        
            if config['format'] == 'json':
                logger.info("Parsing JSON ground truth...")
                truth_data = json.loads(truth_content.decode('utf-8'))
                logger.info(f"Parsed JSON data: {truth_data}")
                # Handle different possible JSON structures
                if isinstance(truth_data, dict):
                    reference_text = (
                        truth_data.get('text') or 
                        truth_data.get('transcript') or 
                        truth_data.get('transcription', '')
                    )
                elif isinstance(truth_data, list) and truth_data:
                    # If it's a list, try to get text from first item
                    reference_text = (
                        truth_data[0].get('text') or
                        truth_data[0].get('transcript') or
                        truth_data[0].get('transcription', '')
                    )
                else:
                    reference_text = ''
                
                if not reference_text:
                    logger.error(f"Could not find text in JSON: {truth_data}")
                    raise ValueError("No text field found in JSON ground truth")
            else:  # txt format
                reference_text = truth_content.decode('utf-8')
            logger.info(f"Reference text length: {len(reference_text)}")
        
            # Load model and transcribe
            logger.info(f"Loading model: {config.get('model_id')}")
            transcriber = self.load_model(config.get('model_id'))
            
            # Prepare generation kwargs
            generate_kwargs = {
                "task": "transcribe",
                "language": config.get('language'),
                "temperature": config.get('temperature', 0.0),
            }
            if config.get('prompt'):
                generate_kwargs["prompt"] = config['prompt']
            
            # Transcribe
            result = transcriber(
                temp_audio,
                return_timestamps="word",
                generate_kwargs=generate_kwargs
            )
            
            # Process words
            words = []
            if isinstance(result, dict) and "chunks" in result:
                for chunk in result["chunks"]:
                    if "text" in chunk and "timestamp" in chunk:
                        words.append({
                            "text": chunk["text"].strip(),
                            "start": chunk["timestamp"][0],
                            "end": chunk["timestamp"][1] if chunk["timestamp"][1] is not None else -1
                        })
            
            # Filter and fix timestamps
            words = [w for w in words if w["text"].strip()]
            for i in range(len(words)-1):
                if words[i]["end"] == -1:
                    words[i]["end"] = words[i+1]["start"]
            if words and words[-1]["end"] == -1:
                words[-1]["end"] = words[-1]["start"] + 0.5
            
            transcription = {
                "text": result["text"],
                "words": words
            }
            
            # Calculate WER
            wer = self.wer_calculator.calculate(reference_text, transcription['text'])
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            result = {
                "file": file_pair["audio"]["filename"],
                "status": "completed",
                "wer": wer,
                "duration": processing_time,
                "transcription": transcription,
                "reference": reference_text,
                "error_analysis": self.wer_calculator.analyze_errors(reference_text, transcription['text'])
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing {file_pair['audio']['filename']}: {str(e)}", exc_info=True)
            raise
        finally:
            # Clean up temporary file
            if temp_audio and os.path.exists(temp_audio):
                logger.info(f"Cleaning up temp file: {temp_audio}")
                os.remove(temp_audio)
    
    def _save_results(self, benchmark_id: str) -> None:
        """Save benchmark results to file"""
        results = self.active_benchmarks[benchmark_id]
        filename = f"benchmark_{benchmark_id}.json"
        path = os.path.join(self.results_dir, filename)
        
        with open(path, 'w') as f:
            json.dump(results, f, indent=2) 