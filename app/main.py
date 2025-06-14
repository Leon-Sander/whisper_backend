from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
import asyncio
import logging
import torch
import io
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

# --- Your Model Loading Logic ---
MODEL_NAME = "distil-whisper/distil-large-v3.5-ct2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16"
MODEL_PATH = os.getenv("WHISPER_MODEL_PATH", "./whisper_models")
logger.info(f"Loading model '{MODEL_NAME}'...")
model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE, download_root=MODEL_PATH)
logger.info("Model loaded successfully.")

@app.websocket("/listen")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established.")
    
    try:
        while True:
            # This logic expects the client to send one single, complete,
            # and perfectly-formed audio file per message.
            audio_file_chunk = await websocket.receive_bytes()
            logger.info(f"Received a complete audio file of {len(audio_file_chunk)} bytes.")

            try:
                # Transcribe this single, valid file directly.
                segments, info = await asyncio.to_thread(
                    model.transcribe,
                    io.BytesIO(audio_file_chunk),
                    beam_size=5,
                    language="en",
                    vad_filter=True,
                    word_timestamps=True # You can keep this
                )
                
                # Use your original, more detailed results structure
                results = []
                for segment in segments:
                    if segment.text.strip():
                        segment_data = {
                            "text": segment.text.strip(),
                            "words": [
                                {"word": word.word.strip(), "start": word.start, "end": word.end}
                                for word in segment.words
                            ]
                        }
                        results.append(segment_data)
                
                if results:
                    logger.info(f"SUCCESS: Transcribed {info.duration:.2f}s of audio and sending results.")
                    await websocket.send_json({"type": "transcription", "segments": results})
                else:
                    logger.info("VAD filtered all audio, no speech detected in this chunk.")

            except Exception as e:
                logger.error(f"Transcription error on a chunk: {e}")

    except WebSocketDisconnect:
        logger.info("Client disconnected.")
    except Exception as e:
        logger.error(f"An unexpected websocket error occurred: {e}")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Serve the HTML interface
@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("app/static/index.html", "r") as f:
        html_content = f.read()
    
    # Replace the hardcoded WebSocket URL with the one from environment variables
    websocket_url = os.getenv("WEBSOCKET_URL", "ws://localhost:8000/listen")
    html_content = html_content.replace(
        'const WEBSOCKET_URL = \'ws://localhost:8000/listen\';',
        f'const WEBSOCKET_URL = \'{websocket_url}\';'
    )
    
    return html_content

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