from fastapi import FastAPI, UploadFile
from transformers import pipeline
import uvicorn
import tempfile
import os

app = FastAPI()
transcriber = None

@app.on_event("startup")
async def startup_event():
    global transcriber
    transcriber = pipeline(
        "automatic-speech-recognition",
        model="NbAiLab/nb-whisper-tiny",
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
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main() 