import os
import tempfile
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
from faster_whisper import WhisperModel
import yt_dlp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Transcription API",
    description="Download audio from social media URLs and transcribe with faster-whisper",
    version="1.0.0"
)

# Model configuration
MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")  # tiny, base, small, medium, large-v3
DEVICE = os.getenv("WHISPER_DEVICE", "auto")      # auto, cpu, cuda
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")  # int8, int16, float16, float32

# Global model instance (loaded on startup)
whisper_model: Optional[WhisperModel] = None


class TranscribeRequest(BaseModel):
    url: HttpUrl
    model: Optional[str] = None
    language: Optional[str] = None


class TranscribeResponse(BaseModel):
    transcript: str
    language: str
    duration: float
    platform: str
    title: str
    url: str


@app.on_event("startup")
async def load_model():
    """Load Whisper model on startup."""
    global whisper_model
    model_name = MODEL_SIZE
    logger.info(f"Loading Whisper model: {model_name} on {DEVICE} with {COMPUTE_TYPE}")
    try:
        whisper_model = WhisperModel(
            model_name,
            device=DEVICE,
            compute_type=COMPUTE_TYPE
        )
        logger.info("Whisper model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}")
        raise


def download_audio(url: str) -> tuple[str, dict]:
    """Download audio from URL using yt-dlp. Returns (audio_path, info_dict)."""
    temp_dir = tempfile.mkdtemp()
    audio_path = os.path.join(temp_dir, "audio.%(ext)s")

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': audio_path,
        'quiet': True,
        'no_warnings': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        # Find the downloaded file
        downloaded_files = list(Path(temp_dir).glob("audio.*"))
        if not downloaded_files:
            raise HTTPException(status_code=500, detail="Audio download failed")
        final_audio_path = str(downloaded_files[0])

    return final_audio_path, info


def transcribe_audio(audio_path: str, language: Optional[str] = None) -> tuple[str, str]:
    """Transcribe audio file using faster-whisper. Returns (transcript, detected_language)."""
    if whisper_model is None:
        raise HTTPException(status_code=503, detail="Whisper model not loaded")

    segments, info = whisper_model.transcribe(
        audio_path,
        language=language,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )

    transcript = " ".join([segment.text for segment in segments])
    return transcript.strip(), info.language


def cleanup_temp_files(audio_path: str):
    """Clean up temporary files."""
    try:
        Path(audio_path).unlink(missing_ok=True)
        Path(audio_path).parent.rmdir()
    except Exception as e:
        logger.warning(f"Cleanup failed: {e}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model": MODEL_SIZE, "device": DEVICE}


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(request: TranscribeRequest, background_tasks: BackgroundTasks):
    """
    Transcribe audio from a social media URL.
    Supports: YouTube, Instagram Reels, TikTok, Twitter/X, and more (via yt-dlp).
    """
    url = str(request.url)
    model_override = request.model or MODEL_SIZE
    language = request.language

    logger.info(f"Transcription request: {url}")

    # Download audio
    try:
        audio_path, info = download_audio(url)
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to download audio: {str(e)}")

    # Transcribe
    try:
        transcript, detected_lang = transcribe_audio(audio_path, language)
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        cleanup_temp_files(audio_path)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

    # Schedule cleanup
    background_tasks.add_task(cleanup_temp_files, audio_path)

    # Extract metadata
    platform = info.get('extractor_key', 'unknown')
    title = info.get('title', 'Untitled')
    duration = info.get('duration', 0)

    return TranscribeResponse(
        transcript=transcript,
        language=detected_lang,
        duration=duration,
        platform=platform,
        title=title,
        url=url
    )


@app.get("/")
async def root():
    return {
        "service": "Transcription API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "transcribe": "/transcribe (POST)"
        },
        "supported_platforms": "YouTube, Instagram Reels, TikTok, Twitter/X, and 1000+ sites via yt-dlp"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)