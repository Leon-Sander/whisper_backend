FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libcudnn9-cuda-12 \
    libcudnn9-dev-cuda-12 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directory for Whisper models
RUN mkdir -p whisper_models

# Set environment variables
ENV DEVICE=cuda
ENV COMPUTE_TYPE=float16
ENV WHISPER_MODEL_PATH=/app/whisper_models
ENV WEBSOCKET_URL=ws://localhost:8000/listen

# Expose ports
EXPOSE 8000  # FastAPI
EXPOSE 8888  # Jupyter (optional)

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 