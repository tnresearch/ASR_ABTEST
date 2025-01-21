from fastapi import FastAPI, UploadFile
from fastapi.staticfiles import StaticFiles
from transformers import pipeline
import uvicorn
import tempfile
import os
import argparse

app = FastAPI()

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Update FastAPI app configuration to use the icon
app = FastAPI(
    title="ASR A/B Testing Framework",
    description="Local transcription service using Whisper models",
    version="0.1.0",
    openapi_url="/api/v1/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    favicon_url="/static/icon.svg"
)

transcriber = None

def parse_args():
    parser = argparse.ArgumentParser(description='Start ASR server with specified model')
    parser.add_argument('--model', 
                       type=str,
                       default="NbAiLab/nb-whisper-tiny",
                       help='Hugging Face model ID for ASR')
    return parser.parse_args()

@app.on_event("startup")
async def startup_event():
    global transcriber
    args = parse_args()
    transcriber = pipeline(
        "automatic-speech-recognition",
        model=args.model,
        # dont use cuda when running locally on mac
        #device="cuda" if os.environ.get("USE_CUDA", "1") == "1" else "cpu",
        # Enable word-level timestamps
        chunk_length_s=30,
        return_timestamps="word"
    )

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_path = temp_file.name

    try:
        # Transcribe with word-level timestamps
        result = transcriber(temp_path)
        
        # Format the response with word-level timing
        words_with_timestamps = []
        
        # Check if we got chunks or word-level timestamps
        if isinstance(result, dict) and "chunks" in result:
            # Handle chunk-level results
            for chunk in result["chunks"]:
                words_with_timestamps.append({
                    "text": chunk["text"].strip(),
                    "start": chunk["timestamp"][0],
                    "end": chunk["timestamp"][1]
                })
        else:
            # Handle word-level results
            for word_data in result:
                if isinstance(word_data, dict) and "text" in word_data:
                    words_with_timestamps.append({
                        "text": word_data["text"].strip(),
                        "start": word_data["timestamp"][0],
                        "end": word_data["timestamp"][1]
                    })

        return {
            "words": words_with_timestamps,
            "full_text": " ".join(w["text"] for w in words_with_timestamps)
        }
    
    finally:
        # Clean up temporary file
        os.unlink(temp_path)

def main():
    args = parse_args()
    print(f"Starting ASR server with model: {args.model}")
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main() 