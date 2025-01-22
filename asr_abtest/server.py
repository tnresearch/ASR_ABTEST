import json
from fastapi import FastAPI, UploadFile, Form, HTTPException
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
from typing import Optional, Literal
from pydantic import BaseModel, Field

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

if __name__ == "__main__":
    main() 