FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
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

# Expose ports
EXPOSE 8000  # FastAPI
EXPOSE 8888  # Jupyter (optional)

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 