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
pip install git+https://github.com/SYSTRAN/faster-whisper
pip install python-multipart==0.0.6
pip install websockets==12.0

# Create directory for Whisper models if it doesn't exist
mkdir -p whisper_models

# Create start script in the current directory
cat > start.sh << 'EOL'
#!/bin/bash
cd /workspace/whisper_backend
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000
EOL

chmod +x start.sh

echo "Setup complete! You can now start the application with: ./start.sh" 