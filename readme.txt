ASR Testing & Analysis Framework

This initial version assumes that the server is running locally on the same machine as the UI. Future versions may support remote server configurations.

Installation & Launch Instructions:

1. Using Docker (Recommended)
   
   a) Build the Docker images:
      - Make build script executable: chmod +x build.sh
      - Run: ./build.sh
   
   b) Launch services:
      - Make launch script executable: chmod +x launch.sh
      - Launch with default GPU: ./launch.sh
      - Or specify GPU (e.g. GPU 2): ./launch.sh 2
   
   c) Access the services:
      - ASR Server: http://localhost:8000
      - UI: http://localhost:7860

2. Using Docker Compose (Alternative)
   
   Launch everything with: docker-compose up --build
   Access UI at: http://localhost:7860

System Requirements:
- Docker
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit installed
- Docker with NVIDIA runtime support

Notes:
- The ASR server uses Whisper models which will be downloaded on first use
- Source code is mounted from host machine for easy development
- Changes to Python files are reflected immediately
- Changes to dependencies require rebuilding: ./build.sh

Stopping the Services:
- If using launch.sh: Press Ctrl+C
- If using Docker directly: docker stop <container-id>
- If using Docker Compose: docker-compose down

For development:
- Source code is mounted from the host machine
- Changes to Python files are reflected immediately
- Changes to dependencies require rebuilding the images

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

### Docker Installation

1. Build and start the containers:
```bash
docker-compose up --build
```

2. Access the UI at `http://localhost:7860`

For development:
- Source code is mounted from the host machine
- Changes to Python files are reflected immediately
- Changes to dependencies require rebuilding the containers
```bash
docker-compose down
docker-compose up --build
```

## Development

The project uses:
- FastAPI for the backend server
- Pure HTML/CSS/JS for the frontend
- Whisper models for ASR

## Note

This initial version assumes that the server is running locally on the same machine as the UI. Future versions may support remote server configurations. 