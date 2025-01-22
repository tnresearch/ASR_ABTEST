import json
from fastapi import FastAPI, UploadFile, Form, HTTPException, File
from fastapi.middleware.cors import CORSMiddleware
import torch
from transformers import pipeline
import os
import argparse
import uvicorn
from transformers import AutoTokenizer
import time
from datetime import datetime
from enum import Enum
from typing import Optional, Literal, List, Union
from pydantic import BaseModel, Field
from .benchmark import BenchmarkProcessor
import logging
import shutil

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define enums for standardization
class ResponseFormat(str, Enum):
    json = "json"
    text = "text"
    srt = "srt"
    vtt = "vtt"
    verbose_json = "verbose_json"

# API Models
class TranscriptionResponse(BaseModel):
    text: str
    words: list = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

class ErrorResponse(BaseModel):
    error: str
    code: str
    param: Optional[str] = None

# Global variables to store model state
current_model = None
current_model_id = None
transcriber = None

SUPPORTED_AUDIO_FORMATS = [".wav", ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".webm"]

# Initialize benchmark processor
benchmark_processor = BenchmarkProcessor()

logger = logging.getLogger(__name__)

def validate_audio_format(filename: str) -> bool:
    """Validate if the audio file format is supported"""
    ext = os.path.splitext(filename)[1].lower()
    return ext in SUPPORTED_AUDIO_FORMATS

def load_model(model_id):
    global current_model, current_model_id, transcriber
    if model_id != current_model_id:
        print(f"Loading model: {model_id}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # First load the tokenizer with our specific settings
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        
        transcriber = pipeline("automatic-speech-recognition", 
                             model=model_id,
                             tokenizer=tokenizer,
                             chunk_length_s=30,
                             return_timestamps="word",
                             device=device)
        current_model_id = model_id
        current_model = transcriber
        print(f"Model loaded successfully: {model_id}")
    return transcriber

def get_available_models():
    with open("models.json") as f:
        return json.load(f)["model_id"]

@app.get("/models")
async def list_models():
    """Return list of available models"""
    return {"models": get_available_models()}

@app.get("/current-model")
async def get_current_model():
    """Return currently loaded model"""
    return {"current_model": current_model_id or "No model loaded"}

@app.post("/change-model")
async def change_model(model_id: str = Form(...)):
    """Change the current model"""
    try:
        transcriber = load_model(model_id)
        return {
            "success": True,
            "model": model_id
        }
    except Exception as e:
        print(f"Error changing model: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/audio/transcriptions")
async def create_transcription(
    file: UploadFile,
    model_id: str = Form("openai/whisper-small"),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: ResponseFormat = Form(ResponseFormat.json),
    temperature: float = Form(0.0)
):
    """OpenAI-like transcription endpoint"""
    try:
        if not validate_audio_format(file.filename):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Unsupported file format",
                    "code": "unsupported_file_format",
                    "param": "file"
                }
            )

        start_time = time.time()
        transcriber = load_model(model_id)
        
        # Save uploaded file temporarily
        temp_path = f"temp_audio{os.path.splitext(file.filename)[1]}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        
        file_size = os.path.getsize(temp_path)
        
        # Prepare generation kwargs
        generate_kwargs = {
            "task": "transcribe",
            "language": language if language else None,
            "temperature": temperature,
        }
        if prompt:
            generate_kwargs["prompt"] = prompt

        # Transcribe
        result = transcriber(
            temp_path,
            return_timestamps="word",
            generate_kwargs=generate_kwargs
        )
        
        # Clean up
        os.remove(temp_path)
        
        # Calculate processing time
        processing_time = round(time.time() - start_time, 4)
        
        # Format response based on requested format
        if response_format == ResponseFormat.text:
            return result["text"]
            
        # Process words and create response
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

        response = {
            "text": result["text"],
            "words": words,
            "processed_date": datetime.now().isoformat(),
            "processing_duration_sec": float(format(processing_time, '.4f')),
            "file_size_kb": round(file_size / 1024, 2),
            "model_id": model_id,
            "prompt": prompt if prompt else None,
            "temperature": float(temperature) if temperature else 0.0,
            "language": language if language else None
        }

        if response_format == ResponseFormat.text:
            return result["text"]
        elif response_format == ResponseFormat.verbose_json:
            return response
        else:  # json format and future formats (srt, vtt)
            return response  # Return all data including metadata
        # TODO: Implement SRT and VTT formats
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": str(e),
                "code": "transcription_error",
            }
        )

# Keep the old endpoint for backward compatibility
@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile, model_id: str = Form("openai/whisper-small")):
    """Legacy transcription endpoint"""
    result = await create_transcription(
        file=audio,
        model_id=model_id,
        response_format=ResponseFormat.verbose_json
    )
    return {"success": True, **result}

@app.get("/audio/config")
async def get_config():
    """Get API configuration and capabilities"""
    return {
        "supported_formats": SUPPORTED_AUDIO_FORMATS,
        "available_models": get_available_models(),
        "response_formats": [format.value for format in ResponseFormat],
        "current_model": current_model_id
    }

@app.on_event("startup")
async def startup_event():
    """Print all registered routes on startup"""
    print("\nRegistered routes:")
    for route in app.routes:
        print(f"{route.methods} {route.path}")
    print()

def main():
    parser = argparse.ArgumentParser(description='Start ASR server')
    parser.add_argument('--model', type=str, default="openai/whisper-small",
                        help='Initial model to load')
    args = parser.parse_args()
    
    # Load default model on startup
    load_model(args.model)
    uvicorn.run(app, host="0.0.0.0", port=8000)

class BenchmarkRequest(BaseModel):
    format: str
    pattern: str
    model_id: Optional[str] = None
    language: Optional[str] = None
    prompt: Optional[str] = None
    temperature: float = 0.0
    response_format: str = "json"

@app.post("/benchmark/start")
async def start_benchmark(
    audio_files: List[UploadFile] = File(description="Audio files to benchmark"),
    truth_files: List[UploadFile] = File(description="Ground truth transcript files"),
    config: str = Form(...)
):
    """Start a new benchmark process"""
    try:
        logger.info(f"Received benchmark request with {len(audio_files)} audio files and {len(truth_files)} truth files")
        
        if not audio_files or not truth_files:
            raise ValueError("Both audio files and truth files must be provided")
        
        if len(audio_files) != len(truth_files):
            raise ValueError("Number of audio files must match number of truth files")
        
        # Ensure temp directory exists
        os.makedirs(benchmark_processor.temp_dir, exist_ok=True)
        
        # Clean up any existing files in temp directory
        try:
            for filename in os.listdir(benchmark_processor.temp_dir):
                file_path = os.path.join(benchmark_processor.temp_dir, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    logger.warning(f"Error deleting {file_path}: {e}")
        except Exception as e:
            logger.warning(f"Error cleaning temp directory: {e}")
        
        # Read all file contents immediately
        file_contents = []
        for audio_file, truth_file in zip(audio_files, truth_files):
            logger.info(f"Reading file contents for {audio_file.filename}")
            audio_content = await audio_file.read()
            truth_content = await truth_file.read()
            file_contents.append({
                "audio": {
                    "filename": audio_file.filename,
                    "content": audio_content
                },
                "truth": {
                    "filename": truth_file.filename,
                    "content": truth_content
                }
            })
        
        config_dict = json.loads(config)
        logger.info(f"Received benchmark config: {config_dict}")
        config_model = BenchmarkRequest(**config_dict)
        config_dict = config_model.dict()
        
        logger.info("Starting benchmark process...")
        benchmark_id = await benchmark_processor.start_benchmark(
            file_contents, config_dict
        )
        logger.info(f"Benchmark started with ID: {benchmark_id}")
        return {"success": True, "benchmark_id": benchmark_id}
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Benchmark error: {str(e)}", exc_info=True)
        print(f"Benchmark error: {str(e)}")  # Add logging
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/benchmark/status/{benchmark_id}")
async def get_benchmark_status(benchmark_id: str):
    """Get status of a benchmark process"""
    try:
        return benchmark_processor.get_status(benchmark_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Benchmark not found")

@app.post("/benchmark/stop")
async def stop_benchmark(benchmark_id: str = Form(...)):
    """Stop a running benchmark process"""
    try:
        benchmark_processor.stop_benchmark(benchmark_id)
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/languages")
async def get_languages():
    """Return list of supported languages"""
    try:
        with open("languages.json") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load languages: {str(e)}")

if __name__ == "__main__":
    main() 