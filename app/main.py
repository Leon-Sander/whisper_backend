from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
import asyncio
import logging
import torch
import io
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# --- Your Model Loading Logic (Unchanged and Respected) ---
MODEL_NAME = "distil-whisper/distil-large-v3.5-ct2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "int8"
MODEL_PATH = os.getenv("WHISPER_MODEL_PATH", "./whisper_models")
logger.info(f"Loading model '{MODEL_NAME}' on {DEVICE}...")
model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE, download_root=MODEL_PATH)
logger.info("Model loaded successfully.")

@app.websocket("/listen")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established.")
    
    try:
        while True:
            # 1. Receive a single, complete audio file from the client.
            #    Thanks to the client-side change, this will be a ~3-second chunk.
            audio_chunk = await websocket.receive_bytes()

            try:
                # 2. Transcribe this single chunk directly. No buffering needed.
                segments, info = await asyncio.to_thread(
                    model.transcribe,
                    io.BytesIO(audio_chunk), # Pass the bytes directly
                    beam_size=5,
                    language="en",
                    vad_filter=True,
                    word_timestamps=True
                )
                
                # 3. Process and send the results for this chunk.
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
                    logger.info(f"Successfully transcribed a {info.duration:.2f}s chunk.")
                    await websocket.send_json({"type": "transcription", "segments": results})
                else:
                    logger.info("Received a chunk with no speech detected.")

            except Exception as e:
                logger.error(f"Transcription error on a chunk: {e}")
                # Don't crash the connection, just log and continue.
                await websocket.send_json({"type": "error", "message": str(e)})

    except WebSocketDisconnect:
        logger.info("Client disconnected.")
    except Exception as e:
        logger.error(f"An unexpected websocket error occurred: {e}")

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