#!/bin/bash

# Check if GPU ID is provided
GPU_ID=${1:-0}  # Default to GPU 0 if not specified

# Set GPU for ASR server
export CUDA_VISIBLE_DEVICES=$GPU_ID
export PYTHONPATH=$PWD

# Start ASR server in background
echo "Starting ASR server on GPU $GPU_ID..."
python -m asr_abtest.server --host 0.0.0.0 --port 8000 &
ASR_SERVER_PID=$!

# Wait a bit for ASR server to start
sleep 5

# Start UI server
echo "Starting UI server..."
python -m asr_abtest.ui.app --host 0.0.0.0 --port 7860 --debug &
UI_SERVER_PID=$!

# Handle graceful shutdown
trap 'kill $ASR_SERVER_PID $UI_SERVER_PID; exit' SIGINT SIGTERM

# Keep script running
wait 