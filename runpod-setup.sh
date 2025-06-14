#!/bin/bash

# Install our specific requirements
pip install -r requirements.txt

# Create directory for Whisper models
mkdir -p whisper_models

# Create start script
cat > start-whisper.sh << 'EOL'
#!/bin/bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
EOL

chmod +x start-whisper.sh

echo "Setup complete! You can now start the application with: ./start-whisper.sh" 