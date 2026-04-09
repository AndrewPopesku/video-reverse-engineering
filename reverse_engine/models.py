"""Pydantic data models for video insights."""

from pathlib import Path

from pydantic import BaseModel, field_validator


class TimeRange(BaseModel):
    """A time range in seconds."""

    start: float
    end: float

    @field_validator("end")
    @classmethod
    def end_after_start(cls, v: float, info) -> float:
        if "start" in info.data and v < info.data["start"]:
            raise ValueError("end must be >= start")
        return v

    @property
    def duration(self) -> float:
        return self.end - self.start


class VideoMeta(BaseModel):
    """Metadata about the source video."""

    path: Path
    duration: float
    fps: float
    width: int
    height: int


class Scene(BaseModel):
    """A detected scene with optional keyframe."""

    index: int
    start: float
    end: float
    keyframe_path: Path | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


class TranscriptWord(BaseModel):
    """A single word with timing."""

    start: float
    end: float
    word: str
    confidence: float = 0.0


class TranscriptSegment(BaseModel):
    """A segment of transcribed speech."""

    start: float
    end: float
    text: str
    words: list[TranscriptWord] = []
    confidence: float = 0.0


class Transcript(BaseModel):
    """Full transcript with segments."""

    language: str = "en"
    segments: list[TranscriptSegment] = []

    @property
    def full_text(self) -> str:
        return " ".join(seg.text for seg in self.segments)


class NamedEntity(BaseModel):
    """A named entity extracted from transcript or OCR."""

    text: str
    category: str  # PERSON, BRAND, LOCATION, PRODUCT, EVENT
    instances: list[TimeRange] = []


class SceneLabel(BaseModel):
    """Visual labels and description for a scene."""

    scene_index: int
    caption: str
    objects: list[str] = []
    shot_type: str = ""  # close-up, medium, wide, extreme-wide
    mood: str = ""


class OcrResult(BaseModel):
    """On-screen text extracted from a keyframe."""

    text: str
    scene_index: int
    confidence: float = 0.0


class BrandMention(BaseModel):
    """A brand detected visually or in audio/text."""

    name: str
    source: str  # "visual", "audio", "text"
    instances: list[TimeRange] = []


class FaceGroup(BaseModel):
    """A group of face appearances (same person)."""

    id: str
    appearances: list[TimeRange] = []
    description: str = ""
    thumbnail_path: Path | None = None


class EmotionSegment(BaseModel):
    """Emotion detected in a transcript segment."""

    start: float
    end: float
    emotion: str  # joy, anger, sadness, fear, surprise, neutral
    confidence: float = 0.0


class Topic(BaseModel):
    """An inferred topic."""

    name: str
    description: str = ""
    confidence: float = 0.0
    related_keywords: list[str] = []


class Keyword(BaseModel):
    """An extracted keyword."""

    text: str
    relevance: float = 0.0
    count: int = 0


class AudioEvent(BaseModel):
    """A classified audio event."""

    label: str  # music, silence, laughter, applause, siren, etc.
    start: float
    end: float
    confidence: float = 0.0


class ModerationResult(BaseModel):
    """Content moderation flags."""

    is_safe: bool = True
    flags: list[str] = []


class VideoInsights(BaseModel):
    """Complete video insights — the top-level output model."""

    video: VideoMeta
    scenes: list[Scene] = []
    transcript: Transcript | None = None
    labels: list[SceneLabel] = []
    ocr: list[OcrResult] = []
    entities: list[NamedEntity] = []
    keywords: list[Keyword] = []
    topics: list[Topic] = []
    brands: list[BrandMention] = []
    faces: list[FaceGroup] = []
    emotions: list[EmotionSegment] = []
    audio_events: list[AudioEvent] = []
    moderation: ModerationResult | None = None
    summary: str = ""
