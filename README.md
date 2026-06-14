# Transcription API

Local transcription service for social media content. Downloads audio from URLs (YouTube, Instagram Reels, TikTok, Twitter/X, 1000+ sites via yt-dlp) and transcribes using faster-whisper.

## Features

- **Universal URL support** — yt-dlp handles 1000+ sites
- **Local transcription** — faster-whisper runs on CPU/GPU, no API keys needed
- **Multiple models** — tiny, base, small, medium, large-v3
- **FastAPI** — REST API with automatic docs at `/docs`

## Quick Start

### Docker (Recommended)

```bash
docker build -t transcription-api .
docker run -p 8000:8000 \
  -e WHISPER_MODEL=base \
  -e WHISPER_DEVICE=auto \
  -e WHISPER_COMPUTE_TYPE=int8 \
  transcription-api
```

### Local Development

```bash
pip install -r requirements.txt
python main.py
```

## Configuration

| Environment Variable | Default | Options |
|---------------------|---------|---------|
| `WHISPER_MODEL` | `base` | `tiny`, `base`, `small`, `medium`, `large-v3` |
| `WHISPER_DEVICE` | `auto` | `auto`, `cpu`, `cuda` |
| `WHISPER_COMPUTE_TYPE` | `int8` | `int8`, `int16`, `float16`, `float32` |

## API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### Transcribe a URL
```bash
curl -X POST http://localhost:8000/transcribe \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.instagram.com/reel/ABC123/"}'
```

Response:
```json
{
  "transcript": "This is the transcribed text...",
  "language": "en",
  "duration": 15.2,
  "platform": "instagram",
  "title": "Reel Title",
  "url": "https://www.instagram.com/reel/ABC123/"
}
```

### Optional Parameters
```json
{
  "url": "https://youtube.com/shorts/XYZ",
  "model": "small",
  "language": "en"
}
```

## Model Selection Guide

| Model | Size | Speed (CPU) | Accuracy | Best For |
|-------|------|-------------|----------|----------|
| tiny | 39 MB | ~0.1x realtime | Good | Quick tests, shorts |
| **base** | 74 MB | ~0.2x realtime | **Balanced** | **General use** |
| small | 244 MB | ~0.5x realtime | Better | Longer content |
| medium | 769 MB | ~1x realtime | Great | High accuracy needs |
| large-v3 | 1550 MB | ~2x realtime | Best | Production, multilingual |

For 15-60s shorts/reels: **`tiny` or `base`** on CPU is instant.

## Deploy to Coolify

1. Push to GitHub
2. In Coolify: New Application → GitHub → Select this repo
3. Build Pack: `Dockerfile`
4. Port: `8000`
5. Set environment variables as needed
6. Deploy

## License

MIT