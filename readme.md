🎧 ASR Error-Analysis UI
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

3. Start the whisper server:
```bash
serve-asr
```

4. Start the ui server:
```bash
asr-abtest-ui
```

5. Open `http://localhost:8000` in your browser

## Development

The project uses:
- FastAPI for the backend server
- Pure HTML/CSS/JS for the frontend
- Whisper models for ASR

## Note

This initial version assumes that the server is running locally on the same machine as the UI. Future versions will support remote server configurations. 