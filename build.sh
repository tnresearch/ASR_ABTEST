#!/bin/bash

echo "Building ASR Server image..."
docker build -t asr-server \
    --file docker/asr-server.Dockerfile \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    .

echo "Building UI App image..."
docker build -t ui-app \
    --file docker/ui-app.Dockerfile \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    .

echo "Build complete! You can now run the containers with:"
echo "docker run -d --gpus device=2 -p 8000:8000 -v \$(pwd):/app asr-server"
echo "docker run -d -p 7860:7860 -v \$(pwd):/app -e ASR_SERVER_URL=http://localhost:8000 ui-app" 