import os
import tempfile
import uuid
import asyncio
from pathlib import Path
from typing import Optional

import yt_dlp
from faster_whisper import WhisperModel
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, HttpUrl
import uvicorn

# Configuration from environment
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
DEVICE = os.getenv("DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")

# Global model instance (loaded on startup)
whisper_model: Optional[WhisperModel] = None

app = FastAPI(title="Transcription API", version="1.0.0")


class TranscribeRequest(BaseModel):
    url: HttpUrl
    model: Optional[str] = None
    language: Optional[str] = None


class TranscribeResponse(BaseModel):
    transcript: str
    duration: float
    language: str
    model_used: str
    source_url: str
    title: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    device: str


@app.on_event("startup")
async def load_model():
    global whisper_model
    model_name = WHISPER_MODEL
    try:
        whisper_model = WhisperModel(
            model_name,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            download_root="/root/.cache/huggingface"
        )
        print(f"Loaded Whisper model: {model_name} on {DEVICE}")
    except Exception as e:
        print(f"Failed to load Whisper model: {e}")
        whisper_model = None


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if whisper_model else "degraded",
        model_loaded=whisper_model is not None,
        model_name=WHISPER_MODEL,
        device=DEVICE
    )


def download_audio(url: str) -> tuple[str, dict]:
    """Download audio from URL using yt-dlp, return (audio_path, video_info)"""
    temp_dir = Path(tempfile.gettempdir())
    audio_filename = f"{uuid.uuid4().hex}.mp3"
    audio_path = temp_dir / audio_filename

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': str(audio_path.with_suffix('')),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        # Find the actual downloaded file
        downloaded_files = list(temp_dir.glob(f"{audio_filename}*"))
        if not downloaded_files:
            # Try with the video ID
            video_id = info.get('id', uuid.uuid4().hex)
            downloaded_files = list(temp_dir.glob(f"{video_id}*"))
        
        if downloaded_files:
            actual_path = downloaded_files[0]
        else:
            actual_path = audio_path

    return str(actual_path), info


def transcribe_audio(audio_path: str, model_name: Optional[str] = None, language: Optional[str] = None) -> tuple[str, str, float]:
    """Transcribe audio file using faster-whisper"""
    global whisper_model
    
    if whisper_model is None:
        raise HTTPException(status_code=503, detail="Whisper model not loaded")
    
    # Use provided model or default
    model = whisper_model
    if model_name and model_name != WHISPER_MODEL:
        # Load different model on-demand (not ideal for production, but works)
        model = WhisperModel(
            model_name,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            download_root="/root/.cache/huggingface"
        )
    
    segments, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )
    
    transcript = " ".join([segment.text for segment in segments])
    return transcript, info.language, info.duration


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_url(request: TranscribeRequest, background_tasks: BackgroundTasks):
    """Transcribe a video/audio URL"""
    if whisper_model is None:
        raise HTTPException(status_code=503, detail="Whisper model not loaded")
    
    url = str(request.url)
    audio_path = None
    
    try:
        # Download audio
        audio_path, info = download_audio(url)
        
        # Transcribe
        transcript, language, duration = transcribe_audio(
            audio_path,
            model_name=request.model,
            language=request.language
        )
        
        # Cleanup audio file in background
        background_tasks.add_task(cleanup_file, audio_path)
        
        return TranscribeResponse(
            transcript=transcript.strip(),
            duration=duration,
            language=language,
            model_used=request.model or WHISPER_MODEL,
            source_url=url,
            title=info.get('title')
        )
        
    except yt_dlp.utils.DownloadError as e:
        if audio_path:
            cleanup_file(audio_path)
        raise HTTPException(status_code=400, detail=f"Failed to download: {str(e)}")
    except Exception as e:
        if audio_path:
            cleanup_file(audio_path)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


def cleanup_file(filepath: str):
    """Clean up temporary file"""
    try:
        Path(filepath).unlink(missing_ok=True)
    except Exception:
        pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)