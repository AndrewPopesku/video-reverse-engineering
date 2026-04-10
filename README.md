# reverse-engine

Reverse-engineer any video into structured insights. Feed it a video file or YouTube URL and get back scene detection, transcription, visual analysis, named entities, topics, embeddings, and semantic search — all from a single pipeline call.

## What it does

```
video file / URL
      │
      ▼
┌─────────────┐
│  Pipeline    │
│              │
│  1. Download (yt-dlp)
│  2. Scene detection (PySceneDetect)
│  3. Keyframe extraction
│  4. Audio extraction + stem separation (Demucs)
│  5. Transcription (Whisper)
│  6. Visual analysis (Gemini)
│  7. Transcript analysis (Gemini)
│  8. Audio event classification (Gemini)
│  9. Timeline export (OTIO → FCP XML)
└─────┬───────┘
      │
      ▼
  VideoInsights
  ├── scenes, keyframes
  ├── transcript + subtitles
  ├── scene labels, OCR, brands, faces
  ├── entities, keywords, topics, emotions
  ├── audio events
  └── summary
```

## Optional: Embedding search

Run the embedding pipeline to add semantic scene search, near-duplicate detection, and clustering using Gemini Embedding 2.

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
# Clone
git clone https://github.com/AndrewPopesku/reverse-engine.git
cd reverse-engine

# Install dependencies
uv sync

# Configure API key
cp .env.example .env
# Edit .env and add your Gemini API key
```

### System dependencies

- **ffmpeg** — used for audio extraction and clip cutting
- **demucs** — installed via pip, but may need a working PyTorch/CUDA setup for GPU acceleration

## Usage

### Basic pipeline

```bash
python examples/basic_pipeline.py ./my-video.mp4
```

### Embedding search + clustering

```bash
python examples/embedding_search.py ./my-video.mp4
```

### Evaluate a description against extracted insights

```bash
# First run the pipeline to generate output/insights.json, then:
python examples/evaluate_video.py --description "A promo trailer about car restoration"
python examples/evaluate_video.py --description-file my-description.txt
```

### As a library

```python
from reverse_engine import reverse_engineer

result = reverse_engineer(
    "./video.mp4",
    output_dir="./output",
    insights={"transcript", "visual", "text", "audio"},
)

print(result.summary)
print(f"{len(result.scenes)} scenes, {len(result.entities)} entities")
```

## Output

The pipeline writes to `./output/`:

| File | Description |
|------|-------------|
| `insights.json` | All extracted insights as structured JSON |
| `subtitles.srt` | Transcript as SRT subtitles |
| `timeline.xml` | FCP XML timeline with scenes and stems |
| `keyframes/` | One keyframe image per scene |
| `audio.wav` | Extracted audio track |

## Project structure

```
reverse_engine/
├── __init__.py          # Public API
├── pipeline.py          # Main orchestration
├── models.py            # Pydantic data models
├── scenes.py            # Scene detection
├── keyframes.py         # Keyframe extraction
├── transcript.py        # Whisper transcription
├── stems.py             # Audio extraction + Demucs stem separation
├── clips.py             # Per-scene clip extraction
├── embeddings_store.py  # Embedding storage, search, clustering
├── timeline.py          # OTIO timeline + FCP XML export
├── report.py            # JSON/SRT export
├── evaluate.py          # Description-vs-insights evaluation
├── download.py          # yt-dlp downloader
└── gemini/
    ├── client.py        # Gemini API client
    ├── visual.py        # Keyframe analysis
    ├── text.py          # Transcript analysis
    ├── audio.py         # Audio event classification
    └── embeddings.py    # Gemini Embedding 2

examples/
├── basic_pipeline.py    # Run the full pipeline
├── embedding_search.py  # Embeddings + semantic search
└── evaluate_video.py    # Evaluate a description

tests/                   # pytest test suite
```

## Tests

```bash
uv run pytest
```

## License

MIT
