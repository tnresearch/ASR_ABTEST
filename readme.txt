ASR A/B Testing Framework
========================

This framework provides a local transcription service using the NbAiLab/nb-whisper-tiny model, with a web-based UI for transcription comparison and A/B testing.

Installation
-----------

1. Clone this repository:
   ```
   git clone <repository-url>
   cd <repository-name>
   ```

2. Install the package and its dependencies:
   ```
   pip install -e .
   ```

This will install required dependencies including:
- torch
- transformers
- fastapi
- uvicorn
- python-multipart
- flask
- requests

Usage
-----

1. Start the ASR API server in one terminal:
   ```
   serve-asr
   ```
   This starts the FastAPI server that handles transcription requests.

2. Start the A/B Testing UI in another terminal:
   ```
   asr-abtest-ui
   ```
   The web interface will be available at http://localhost:7860

Features
--------

The web interface provides several key features:

1. Transcribe File
   - Upload WAV audio files for transcription
   - View formatted text and raw JSON output
   - Download transcriptions with word-level timestamps

2. Record & Transcribe
   - Record audio directly through your browser
   - Get instant transcription
   - Download both audio and transcript

3. Compare & Analyze
   - Upload an audio file with two different transcripts
   - Side-by-side comparison
   - Word-level timing synchronization
   - Quality assessment tools
   - Statistical analysis (WER calculation)

File Formats
-----------

- Audio: WAV files only
- Transcripts: JSON format with word-level timestamps
- Example transcript format:
  ```json
  {
    "words": [
      {
        "text": "Hello",
        "start": 0.0,
        "end": 0.5
      },
      ...
    ]
  }
  ```

Notes
-----

- The service uses the NbAiLab/nb-whisper-tiny model by default
- For local use on Mac, CUDA is disabled by default
- The web UI runs on port 7860
- The FastAPI server handles the actual transcription processing

Troubleshooting
--------------

1. If the server won't start, check that:
   - All dependencies are installed
   - No other service is using port 7860
   - Python version is 3.8 or higher

2. If transcription fails:
   - Ensure audio file is in WAV format
   - Check the audio file isn't corrupted
   - Verify the file size isn't too large

For more information or to report issues, please visit the repository page. 