"""Scene embedding using Gemini Embedding 2 (gemini-embedding-2-preview)."""

import time
from pathlib import Path

from google import genai

from ..models import Scene, Transcript

EMBEDDING_MODEL = "gemini-embedding-2-preview"

MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0


def embed_scenes(
    client: genai.Client,
    scenes: list[Scene],
    clips: list[dict[str, Path]],
    transcript: Transcript | None = None,
    dimensions: int = 768,
    task: str = "search result",
) -> list[dict]:
    """Embed each scene as an aggregated multimodal vector.

    Combines video clip + audio segment + transcript text into one
    embedding per scene using gemini-embedding-2-preview.

    Args:
        client: Authenticated genai.Client instance.
        scenes: List of Scene objects with timing info.
        clips: List of dicts with 'video' and 'audio' Path keys per scene.
        transcript: Optional transcript for text overlay.
        dimensions: Output embedding dimensions (128-3072, recommend 768).
        task: Task type for retrieval optimization.

    Returns:
        List of dicts with 'scene_index', 'vector', 'dimensions', 'modalities'.
    """
    clip_lookup = {c["scene_index"]: c for c in clips}
    embeddings = []

    for scene in scenes:
        clip = clip_lookup.get(scene.index)
        if clip is None:
            print(f"    Warning: no clip for scene {scene.index}, skipping")
            continue

        parts = _build_scene_parts(scene, clip, transcript, task)
        modalities = _detect_modalities(clip, scene, transcript)

        vector = _embed_with_retry(client, parts, dimensions)
        if vector is None:
            print(f"    Warning: embedding failed for scene {scene.index}, skipping")
            continue

        embeddings.append({
            "scene_index": scene.index,
            "vector": vector,
            "dimensions": dimensions,
            "modalities": modalities,
        })

    return embeddings


def embed_query(
    client: genai.Client,
    query: str,
    dimensions: int = 768,
) -> list[float]:
    """Embed a text query for search against scene embeddings."""
    content = f"task: search query | query: {query}"

    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=content,
        config={"output_dimensionality": dimensions},
    )
    return list(result.embeddings[0].values)


def _build_scene_parts(
    scene: Scene,
    clip: dict[str, Path],
    transcript: Transcript | None,
    task: str,
) -> list:
    """Build the multimodal Content parts for one scene."""
    parts = []

    # Task description as text
    parts.append(genai.types.Part(text=f"task: {task} | query: video scene"))

    # Video clip
    video_path = clip["video"]
    if video_path.exists() and video_path.stat().st_size > 0:
        parts.append(
            genai.types.Part.from_bytes(
                data=video_path.read_bytes(),
                mime_type="video/mp4",
            )
        )

    # Audio clip
    audio_path = clip["audio"]
    if audio_path.exists() and audio_path.stat().st_size > 0:
        parts.append(
            genai.types.Part.from_bytes(
                data=audio_path.read_bytes(),
                mime_type="audio/wav",
            )
        )

    # Transcript text for this scene's time range
    if transcript:
        scene_text = _get_scene_transcript(transcript, scene.start, scene.end)
        if scene_text:
            parts.append(genai.types.Part(text=scene_text))

    return parts


def _get_scene_transcript(
    transcript: Transcript, start: float, end: float
) -> str:
    """Extract transcript text overlapping a scene's time range."""
    words = []
    for seg in transcript.segments:
        if seg.end < start:
            continue
        if seg.start > end:
            break
        # Segment overlaps with scene
        words.append(seg.text)
    return " ".join(words).strip()


def _detect_modalities(
    clip: dict[str, Path],
    scene: Scene,
    transcript: Transcript | None,
) -> list[str]:
    modalities = []
    if clip["video"].exists() and clip["video"].stat().st_size > 0:
        modalities.append("video")
    if clip["audio"].exists() and clip["audio"].stat().st_size > 0:
        modalities.append("audio")
    if transcript and _get_scene_transcript(transcript, scene.start, scene.end):
        modalities.append("text")
    return modalities


def _embed_with_retry(
    client: genai.Client,
    parts: list,
    dimensions: int,
) -> list[float] | None:
    """Call the embedding API with exponential backoff retry."""
    content = genai.types.Content(parts=parts)

    for attempt in range(MAX_RETRIES):
        try:
            result = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=content,
                config={"output_dimensionality": dimensions},
            )
            return list(result.embeddings[0].values)
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"    Embedding failed after {MAX_RETRIES} attempts: {e}")
                return None
            delay = RETRY_BASE_DELAY * (2 ** attempt)
            print(f"    Retry {attempt + 1}/{MAX_RETRIES} after {delay}s: {e}")
            time.sleep(delay)

    return None
