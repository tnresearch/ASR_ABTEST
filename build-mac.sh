#!/bin/bash

echo "🍎 Building ASR Server image for Mac M1..."
docker build -t asr-server-mac \
    --platform linux/arm64 \
    --file docker/asr-server.Dockerfile \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    .

echo "🖥️  Building UI App image for Mac M1..."
docker build -t ui-app-mac \
    --platform linux/arm64 \
    --file docker/ui-app.Dockerfile \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    .

echo "✅ Build complete! You can now run the containers with:"
echo "docker-compose up"
echo ""
echo "Or access the services at:"
echo "🔗 ASR Server: http://localhost:8000"
echo "🔗 UI App: http://localhost:7860"