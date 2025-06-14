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
import av
import numpy as np

# Use a more detailed logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()

# --- Your Model Loading Logic ---
MODEL_NAME = "distil-whisper/distil-large-v3.5-ct2"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"
MODEL_PATH = os.getenv("WHISPER_MODEL_PATH", "./whisper_models")
logger.info(f"Loading model '{MODEL_NAME}'...")
model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE, download_root=MODEL_PATH)
logger.info("Model loaded successfully.")

class StatefulDecoder:
    """Manages the decoding state for a single WebSocket audio stream."""
    def __init__(self):
        self.codec_context = None
        self.resampler = None
        self.pcm_buffer = bytearray()

    def _initialize_decoder(self, first_chunk: bytes):
        """Initializes the decoder using the header from the first chunk."""
        try:
            with av.open(io.BytesIO(first_chunk)) as container:
                stream = container.streams.audio[0]
                logger.info("---- Initializing Decoder from First Chunk ----")
                logger.info(f"Codec: {stream.codec_context.codec.name}")
                logger.info(f"Layout: {stream.codec_context.layout.name}")
                logger.info(f"Sample Rate: {stream.codec_context.sample_rate}")
                logger.info("-------------------------------------------------")
                self.codec_context = stream.codec_context
                self.resampler = av.AudioResampler(
                    format='s16', layout='mono', rate=16000
                )
            return True
        except Exception as e:
            logger.error(f"Failed to initialize decoder from first chunk: {e}")
            return False

    def decode_chunk(self, chunk: bytes) -> np.ndarray | None:
        """Decodes a subsequent chunk of the stream by wrapping it in a Packet."""
        if not self.codec_context:
            return None
        
        try:
            # THE FIX: Don't use parse(). Treat the entire chunk as a single packet.
            packet = av.Packet(chunk)
            
            # Decode the single packet
            decoded_frames = self.codec_context.decode(packet)

            # If the decoder buffers frames, it might not return anything on the first pass
            if not decoded_frames:
                return None
                
            resampled_data = []
            for frame in decoded_frames:
                # Resample the frame to the format Whisper needs (16kHz mono s16)
                resampled_frames = self.resampler.resample(frame)
                for resampled_frame in resampled_frames:
                    resampled_data.append(resampled_frame.to_ndarray())

            if not resampled_data:
                return None

            # Concatenate all numpy arrays from the resampled frames
            return np.concatenate(resampled_data)

        except Exception as e:
            # Errors like "Resource temporarily unavailable" can happen if a packet is needed
            # to complete a frame. We can often safely ignore these.
            if "Resource temporarily unavailable" not in str(e):
                 logger.error(f"Error decoding subsequent chunk: {e}")
            return None


@app.websocket("/listen")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established.")
    
    decoder = StatefulDecoder()
    initialized = False
    
    MIN_PCM_BUFFER_SIZE = 16000 * 2 * 3 # Buffer ~3 seconds of raw PCM audio

    try:
        while True:
            chunk = await websocket.receive_bytes()
            
            if not initialized:
                #_ NEW LOGGING: Announce first chunk
                logger.info(f"Received first chunk ({len(chunk)} bytes) for initialization.")
                if decoder._initialize_decoder(chunk):
                    initialized = True
                else:
                    await websocket.close(code=1003, reason="Invalid initial audio chunk")
                    break
                continue

            #_ NEW LOGGING: Announce subsequent chunk
            logger.info(f"Received subsequent chunk ({len(chunk)} bytes).")
            pcm_chunk = decoder.decode_chunk(chunk)
            
            if pcm_chunk is not None:
                decoder.pcm_buffer.extend(pcm_chunk.tobytes())
                #_ NEW LOGGING: Show buffer growth
                logger.info(f"PCM buffer size: {len(decoder.pcm_buffer)} / {MIN_PCM_BUFFER_SIZE}")

            if len(decoder.pcm_buffer) >= MIN_PCM_BUFFER_SIZE:
                #_ NEW LOGGING: Announce transcription trigger
                logger.info("--- Buffer full. Triggering transcription. ---")
                buffer_to_process = decoder.pcm_buffer
                decoder.pcm_buffer = bytearray()
                
                try:
                    # Convert raw PCM to a WAV file in memory
                    wav_data = np.frombuffer(buffer_to_process, dtype=np.int16)
                    wav_file = io.BytesIO()
                    channels=1; sample_rate=16000; sampwidth=2
                    wav_header = b'RIFF' + (36 + wav_data.nbytes).to_bytes(4, 'little') + b'WAVEfmt ' + \
                                (16).to_bytes(4, 'little') + (1).to_bytes(2, 'little') + \
                                (channels).to_bytes(2, 'little') + (sample_rate).to_bytes(4, 'little') + \
                                (sample_rate * channels * sampwidth).to_bytes(4, 'little') + \
                                (channels * sampwidth).to_bytes(2, 'little') + \
                                (sampwidth * 8).to_bytes(2, 'little') + b'data' + \
                                wav_data.nbytes.to_bytes(4, 'little')
                    wav_file.write(wav_header + wav_data.tobytes())
                    wav_file.seek(0)
                    
                    #_ NEW LOGGING: Announce what's being sent to Whisper
                    logger.info(f"Transcribing {len(buffer_to_process)} bytes of PCM data (as WAV).")
                    
                    segments, info = await asyncio.to_thread(
                        model.transcribe,
                        wav_file, beam_size=5, language="en", vad_filter=True, word_timestamps=True
                    )
                    
                    results = [seg.text.strip() for seg in segments if seg.text.strip()]
                    if results:
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