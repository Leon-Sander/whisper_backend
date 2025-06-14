FROM nvcr.io/nvidia/pytorch:24.01-py3

WORKDIR /app

# Install setuptools first
RUN pip3 install --no-cache-dir setuptools

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV DEVICE=cuda
ENV COMPUTE_TYPE=float16
ENV WHISPER_MODEL_PATH=./whisper_models
ENV WEBSOCKET_URL=ws://localhost:8000/listen

# Expose ports
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 