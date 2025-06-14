import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
import asyncio
import logging
from typing import List, Dict
import os
import json
import torch
import io
import wave
import numpy as np

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Live Speech-to-Text API")

# --- Model Loading ---
MODEL_NAME = "distil-whisper/distil-large-v3.5-ct2"

# Force CPU mode for now
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "float32"

MODEL_PATH = os.getenv("WHISPER_MODEL_PATH", "./whisper_models")

logger.info(f"Loading Distil-Whisper model '{MODEL_NAME}' on {DEVICE}...")
model = WhisperModel(
    MODEL_NAME,
    device=DEVICE,
    compute_type=COMPUTE_TYPE,
    download_root=MODEL_PATH
)
logger.info("Distil-Whisper model loaded successfully.")

# --- WebSocket Endpoint for Live Transcription ---
BUFFER_SIZE_BYTES = 50000  # ~3 seconds of audio at 16kHz

def prepare_audio_buffer(audio_data):
    """Prepare audio buffer for Whisper with proper format detection."""
    try:
        # Create a buffer and set its name to help with format detection
        buffer = io.BytesIO(audio_data)
        buffer.name = "audio.webm"  # This helps Whisper identify the format
        return buffer
    except Exception as e:
        logger.error(f"Error preparing audio buffer: {e}")
        return None

@app.websocket("/listen")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established.")
    
    raw_buffer = bytearray()
    MIN_BUFFER_SIZE = 50000  # Minimum size for a complete WebM frame
    MAX_BUFFER_SIZE = 100000  # Maximum size to prevent memory issues

    try:
        while True:
            data = await websocket.receive_bytes()
            raw_buffer.extend(data)

            # Only process when we have enough data for a complete frame
            if len(raw_buffer) >= MIN_BUFFER_SIZE:
                logger.info(f"Buffer full ({len(raw_buffer)} bytes), transcribing...")
                
                # Take up to MAX_BUFFER_SIZE bytes
                current_chunk = bytes(raw_buffer[:MAX_BUFFER_SIZE])
                # Keep the remainder in the buffer
                raw_buffer = raw_buffer[MAX_BUFFER_SIZE:]
                
                try:
                    # Create a new buffer for each chunk
                    audio_buffer = io.BytesIO(current_chunk)
                    audio_buffer.name = "audio.ogg"  # Set to OGG format

                    # Using advanced features of Distil-Whisper
                    segments, info = await asyncio.to_thread(
                        model.transcribe,
                        audio_buffer,
                        beam_size=5,
                        language="en",
                        condition_on_previous_text=False,
                        vad_filter=True,
                        vad_parameters=dict(
                            min_silence_duration_ms=500,  # Remove silence longer than 500ms
                            speech_pad_ms=100,  # Add padding around speech
                        ),
                        word_timestamps=True  # Enable word-level timestamps
                    )
                    
                    # Process segments with timestamps
                    results = []
                    for segment in segments:
                        segment_data = {
                            "text": segment.text,
                            "start": segment.start,
                            "end": segment.end,
                            "words": [
                                {
                                    "word": word.word,
                                    "start": word.start,
                                    "end": word.end
                                }
                                for word in segment.words
                            ]
                        }
                        results.append(segment_data)
                    
                    if results:
                        logger.info(f"Sending transcription with timestamps")
                        await websocket.send_json({
                            "type": "transcription",
                            "segments": results,
                            "info": {
                                "language": info.language,
                                "language_probability": info.language_probability
                            }
                        })

                except Exception as e:
                    logger.error(f"Transcription error: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "error": str(e)
                    })

    except WebSocketDisconnect:
        logger.info("Client disconnected.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
    finally:
        logger.info("Closing WebSocket connection.")

# Serve static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Serve the HTML interface
@app.get("/", response_class=HTMLResponse)
async def get():
    with open("app/static/index.html") as f:
        return f.read()

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 