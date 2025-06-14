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

# This buffer size is a balance. Too small = VAD errors. Too large = high latency.
# Let's aim for ~3-5 seconds of audio. A typical WebM chunk is ~4-5KB per second.
MIN_BUFFER_SIZE_FOR_TRANSCRIPTION = 3 * 5000  # ~3 seconds

@app.websocket("/listen")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established.")
    
    # Each connection gets its own buffer.
    audio_buffer = bytearray()

    try:
        while True:
            data = await websocket.receive_bytes()
            audio_buffer.extend(data)

            # Check if the buffer has reached a processable size
            if len(audio_buffer) >= MIN_BUFFER_SIZE_FOR_TRANSCRIPTION:
                logger.info(f"Buffer has reached {len(audio_buffer)} bytes. Processing.")
                
                # Make a copy of the buffer to process, and clear the original
                chunk_to_process = audio_buffer
                audio_buffer = bytearray()

                try:
                    # Transcribe the accumulated audio chunk
                    segments, info = await asyncio.to_thread(
                        model.transcribe,
                        io.BytesIO(chunk_to_process),
                        beam_size=5,
                        language="en",
                        vad_filter=True, # VAD is useful here on the larger chunk
                        word_timestamps=True
                    )
                    
                    # Your results processing logic...
                    results = []
                    for segment in segments:
                        if segment.text.strip():
                            # Re-using your original detailed structure
                            segment_data = {
                                "text": segment.text.strip(),
                                "words": [
                                    {"word": word.word.strip(), "start": word.start, "end": word.end}
                                    for word in segment.words
                                ]
                            }
                            results.append(segment_data)
                    
                    if results:
                        logger.info(f"Sending transcription for {info.duration:.2f}s chunk.")
                        await websocket.send_json({"type": "transcription", "segments": results})

                except Exception as e:
                    logger.error(f"Transcription error: {e}")
                    # Don't crash the connection, just log the error
                    await websocket.send_json({"type": "error", "message": str(e)})

    except WebSocketDisconnect:
        logger.info("Client disconnected.")

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