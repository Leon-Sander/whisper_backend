#!/bin/bash

# We're already in /workspace/whisper_backend
cd /workspace/whisper_backend

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install requirements
pip install --upgrade pip
pip install fastapi==0.104.1
pip install uvicorn==0.24.0
pip install faster-whisper @ git+https://github.com/SYSTRAN/faster-whisper
pip install python-multipart==0.0.6
pip install websockets==12.0

# Create directory for Whisper models if it doesn't exist
mkdir -p whisper_models

# Set up the application to run on startup
echo '#!/bin/bash
cd /workspace/whisper_backend
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000' > /workspace/start.sh

chmod +x /workspace/start.sh

echo "Setup complete! You can now start the application with: ./start.sh" 