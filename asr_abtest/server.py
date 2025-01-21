import json
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import torch
from transformers import pipeline
import os
import argparse
import uvicorn
from transformers import AutoTokenizer
import time
from datetime import datetime

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store model state
current_model = None
current_model_id = None
transcriber = None

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

@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile, model_id: str = Form("openai/whisper-small")):
    """Transcribe audio using specified model"""
    try:
        start_time = time.time()
        # Load model if needed
        transcriber = load_model(model_id)
        
        # Save uploaded file temporarily
        temp_path = "temp_audio.wav"
        with open(temp_path, "wb") as f:
            f.write(await audio.read())
        
        # Get file size
        file_size = os.path.getsize(temp_path)
        
        # Transcribe
        result = transcriber(
            temp_path,
            return_timestamps="word",
            generate_kwargs={"task": "transcribe"}
        )
        
        # Clean up
        os.remove(temp_path)
        
        # Calculate processing time
        processing_time = round(time.time() - start_time, 4)
        
        # Format response
        words = []
        # Handle chunk-level timestamps
        if isinstance(result, dict) and "chunks" in result:
            for chunk in result["chunks"]:
                if "text" in chunk and "timestamp" in chunk:
                    words.append({
                        "text": chunk["text"].strip(),
                        "start": chunk["timestamp"][0],
                        "end": chunk["timestamp"][1] if chunk["timestamp"][1] is not None else -1
                    })

        if not words:
            print(f"Debug - Raw result: {result}")
            raise ValueError("No words found in transcription result")
        
        # Filter out empty words and fix None endings
        words = [w for w in words if w["text"].strip()]
        
        # If last word has no end time, use the start time of next word or add 0.5 seconds
        for i in range(len(words)-1):
            if words[i]["end"] == -1:
                words[i]["end"] = words[i+1]["start"]
        if words and words[-1]["end"] == -1:
            words[-1]["end"] = words[-1]["start"] + 0.5
        
        return {
            "success": True,
            "words": words,
            "metadata": {
                "model_id": model_id,
                "processing_time_seconds": processing_time,
                "file_size_bytes": file_size,
                "datetime": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        print(f"Error in transcribe_audio: {str(e)}")  # Add debug logging
        print(f"Full error details: {type(e).__name__}: {str(e)}")
        return {
            "success": False,
            "error": str(e)
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