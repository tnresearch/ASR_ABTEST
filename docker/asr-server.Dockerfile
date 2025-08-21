FROM --platform=linux/arm64 python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies optimized for Mac M1
COPY requirements-mac.txt .
RUN pip install --no-cache-dir -r requirements-mac.txt

# Don't copy source code - it will be mounted

EXPOSE 8000

CMD ["python", "-m", "asr_abtest.server"] 