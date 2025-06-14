# Live Speech-to-Text API

A real-time speech-to-text transcription service using FastAPI, WebSockets, and faster-whisper.

## Features

- Real-time speech transcription using WebSocket connections
- Powered by faster-whisper for efficient transcription
- Low-latency audio processing
- Simple web interface for testing
- Docker support for easy deployment

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
uvicorn app.main:app --reload
```

3. Open `http://localhost:8000` in your browser to access the test interface.

## API Endpoints

- `GET /`: Web interface for testing
- `WS /listen`: WebSocket endpoint for real-time transcription
- `POST /transcribe`: HTTP endpoint for file transcription

## Docker Deployment

Build and run with Docker:

```bash
docker build -t speech-to-text-api .
docker run -p 8000:8000 speech-to-text-api
```

## RunPod Deployment

1. Build the Docker image
2. Push to Docker Hub
3. Deploy on RunPod using the provided template

## License

MIT 