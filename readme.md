# Local Whisper Transcription Service

This package provides a local transcription service using the NbAiLab/nb-whisper-large model.

## Installation

```bash
pip install -e .
```

## Usage
1. Start the Whisper API server in one terminal:
```bash
serve-whisper
```

2. Start the Gradio UI in another terminal:
```bash
whisper-ui
```

The Gradio interface will be available at http://localhost:7860

## Requirements

- Python 3.8+
- torch
- transformers
- fastapi
- gradio
- uvicorn