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
import av # The PyAV library
import numpy as np

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

def decode_and_resample(audio_chunk: bytes) -> np.ndarray:
    """
    Decodes an in-memory audio chunk (like WebM/OGG) and resamples it to
    the 16kHz mono float32 format required by Whisper.
    """
    try:
        with av.open(io.BytesIO(audio_chunk), 'r') as container:
            # Get the first audio stream
            stream = container.streams.audio[0]
            
            # Setup a resampler to convert to 16kHz mono
            resampler = av.AudioResampler(
                format='s16',     # Signed 16-bit PCM
                layout='mono',      # Mono channel
                rate=16000        # 16kHz sample rate
            )
            
            frames = []
            for frame in container.decode(stream):
                # Resample the frame and append to our list
                resampled_frames = resampler.resample(frame)
                for resampled_frame in resampled_frames:
                    # Convert the frame to a numpy array of int16
                    frames.append(np.frombuffer(resampled_frame.planes[0], np.int16))
            
            if not frames:
                return np.array([], dtype=np.int16)

            # Concatenate all frames into a single numpy array
            return np.concatenate(frames)
    except Exception as e:
        logger.error(f"PyAV decoding error: {e}")
        return np.array([], dtype=np.int16)

def create_wav_from_pcm(pcm_data: np.ndarray, sample_rate: int = 16000) -> io.BytesIO:
    """Creates an in-memory WAV file from a numpy array of PCM data."""
    wav_buffer = io.BytesIO()
    with io.BytesIO() as pcm_io:
        # Write the numpy array to a bytes buffer
        pcm_io.write(pcm_data.tobytes())
        pcm_io.seek(0)
        
        # Use PyAV to create the WAV container
        with av.open(wav_buffer, 'w', format='wav') as container:
            stream = container.add_stream('pcm_s16le', rate=sample_rate, layout='mono')
            # Copy the raw PCM data into the new WAV stream
            while True:
                data = pcm_io.read(1024)
                if not data:
                    break
                # Create a packet and mux it
                packet = av.Packet(data)
                container.mux(packet)
    wav_buffer.seek(0)
    return wav_buffer


@app.websocket("/listen")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established.")
    
    pcm_buffer = bytearray()
    # Buffer ~3 seconds of raw PCM data (16000 samples/sec * 2 bytes/sample * 3 sec)
    MIN_PCM_BUFFER_SIZE = 16000 * 2 * 3

    try:
        while True:
            # 1. Receive a WebM/OGG chunk from the client
            webm_chunk = await websocket.receive_bytes()
            
            # 2. Decode it to raw PCM data
            pcm_chunk = decode_and_resample(webm_chunk)
            
            if pcm_chunk.size > 0:
                pcm_buffer.extend(pcm_chunk.tobytes())

            # 3. Check if the PCM buffer is large enough to transcribe
            if len(pcm_buffer) >= MIN_PCM_BUFFER_SIZE:
                logger.info(f"PCM buffer full ({len(pcm_buffer)} bytes). Transcribing.")
                
                # Create a copy to process and clear the main buffer
                buffer_to_process = pcm_buffer
                pcm_buffer = bytearray()

                # 4. Wrap the PCM data in a WAV header and transcribe
                try:
                    wav_file = create_wav_from_pcm(np.frombuffer(buffer_to_process, dtype=np.int16))
                    
                    segments, info = await asyncio.to_thread(
                        model.transcribe,
                        wav_file,
                        beam_size=5, language="en", vad_filter=True, word_timestamps=True
                    )
                    
                    # Your results processing logic...
                    results = [seg.text.strip() for seg in segments if seg.text.strip()]
                    if results:
                        logger.info(f"Sending transcription for {info.duration:.2f}s chunk.")
                        await websocket.send_json({"type": "transcription", "segments": results})

                except Exception as e:
                    logger.error(f"Transcription failed: {e}")

    except WebSocketDisconnect:
        logger.info("Client disconnected.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")

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