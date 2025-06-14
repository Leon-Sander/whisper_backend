#!/bin/bash

# Create virtual environment in a separate directory
mkdir -p /workspace/venv
python -m venv /workspace/venv
source /workspace/venv/bin/activate

# Install requirements
pip install --upgrade pip
pip install fastapi==0.104.1
pip install uvicorn==0.24.0
pip install git+https://github.com/SYSTRAN/faster-whisper
pip install python-multipart==0.0.6
pip install websockets==12.0

# Create directory for Whisper models
mkdir -p /workspace/whisper_models

# Create start script in workspace
cat > /workspace/start-whisper.sh << 'EOL'
#!/bin/bash
cd /workspace/whisper_backend
source /workspace/venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000
EOL

chmod +x /workspace/start-whisper.sh

echo "Setup complete! You can now start the application with: /workspace/start-whisper.sh" 