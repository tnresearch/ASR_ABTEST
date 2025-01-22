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

ASR Testing & Analysis UI
========================

A web application for comprehensive testing and analysis of ASR (Automatic Speech Recognition) systems. The application consists of a FastAPI server handling transcriptions and a browser-based UI for interaction.

## Features

### Basic Features

- **🤖 Model Configuration**
  - Select from available ASR models
  - Choose transcription language
  - Set temperature for model output
  - Add optional prompts
  - Select response format (JSON, text, etc.)

- **🎙️ Record & Transcribe**
  - Browser-based audio recording
  - Real-time recording status
  - Instant transcription after recording
  - Download options for audio and transcript

- **📄 Transcribe File**
  - Support for multiple audio formats (WAV, MP3, etc.)
  - Word-level timestamps in transcription
  - View both formatted text and raw JSON
  - Download transcription results

### Analysis Tools

#### Qualitative Analysis
- **🔍 Compare & Annotate**
  - Side-by-side transcript comparison
  - Word Error Rate (WER) calculation
  - Interactive audio playback with word highlighting
  - Click-to-play from any word

#### Quantitative Analysis
- **📊 Benchmark**
  - Process multiple audio files with ground truth
  - Calculate WER and CER metrics
  - Track detailed error analysis (substitutions, deletions, insertions)
  - Export results in JSON and Excel formats
  - Real-time progress tracking

## Setup

1. Create a new conda environment:
```bash
conda create -n asr python=3.10
conda activate asr
```

2. Install dependencies:
```bash
pip install -e .
```

3. Start the server:
```bash
serve-asr
```

4. Open `http://localhost:8000` in your browser

## Development

The project uses:
- FastAPI for the backend server
- Pure HTML/CSS/JS for the frontend
- Whisper models for ASR

## Note

This initial version assumes that the server is running locally on the same machine as the UI. Future versions may support remote server configurations. 