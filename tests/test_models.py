"""Tests for Pydantic data models (Phase 0)."""

import json
from pathlib import Path

from reverse_engine.models import (
    AudioEvent,
    BrandMention,
    EmotionSegment,
    FaceGroup,
    Keyword,
    ModerationResult,
    NamedEntity,
    OcrResult,
    Scene,
    SceneLabel,
    TimeRange,
    Topic,
    Transcript,
    TranscriptSegment,
    TranscriptWord,
    VideoInsights,
    VideoMeta,
)


class TestTimeRange:
    def test_create_time_range(self):
        tr = TimeRange(start=1.5, end=3.0)
        assert tr.start == 1.5
        assert tr.end == 3.0

    def test_duration_property(self):
        tr = TimeRange(start=1.0, end=4.5)
        assert tr.duration == 3.5

    def test_end_must_be_after_start(self):
        import pytest

        with pytest.raises(ValueError):
            TimeRange(start=5.0, end=2.0)


class TestVideoMeta:
    def test_create_video_meta(self):
        meta = VideoMeta(
            path=Path("/video.mp4"),
            duration=70.0,
            fps=25.0,
            width=1920,
            height=1080,
        )
        assert meta.duration == 70.0
        assert meta.fps == 25.0
        assert meta.width == 1920
        assert meta.height == 1080


class TestScene:
    def test_create_scene(self):
        scene = Scene(
            index=0,
            start=0.0,
            end=2.5,
            keyframe_path=Path("/keyframes/scene_001.jpg"),
        )
        assert scene.index == 0
        assert scene.duration == 2.5

    def test_scene_without_keyframe(self):
        scene = Scene(index=1, start=2.5, end=5.0)
        assert scene.keyframe_path is None


class TestTranscript:
    def test_transcript_segment(self):
        seg = TranscriptSegment(
            start=0.0,
            end=3.5,
            text="Hello world",
            words=[
                TranscriptWord(start=0.0, end=0.5, word="Hello", confidence=0.95),
                TranscriptWord(start=0.6, end=1.0, word="world", confidence=0.92),
            ],
            confidence=0.93,
        )
        assert seg.text == "Hello world"
        assert len(seg.words) == 2

    def test_transcript_full_text(self):
        transcript = Transcript(
            language="en",
            segments=[
                TranscriptSegment(start=0.0, end=2.0, text="First segment."),
                TranscriptSegment(start=2.0, end=4.0, text="Second segment."),
            ],
        )
        assert transcript.full_text == "First segment. Second segment."


class TestNamedEntity:
    def test_create_entity(self):
        entity = NamedEntity(
            text="BMW",
            category="BRAND",
            instances=[TimeRange(start=5.0, end=7.0)],
        )
        assert entity.text == "BMW"
        assert entity.category == "BRAND"


class TestSceneLabel:
    def test_create_label(self):
        label = SceneLabel(
            scene_index=0,
            caption="A mechanic working on a vintage car engine",
            objects=["car", "engine", "wrench"],
            shot_type="medium",
            mood="focused",
        )
        assert len(label.objects) == 3
        assert label.shot_type == "medium"


class TestOcrResult:
    def test_create_ocr(self):
        ocr = OcrResult(
            text="CAR SOS",
            scene_index=0,
            confidence=0.98,
        )
        assert ocr.text == "CAR SOS"


class TestBrandMention:
    def test_create_brand(self):
        brand = BrandMention(
            name="BMW",
            source="visual",
            instances=[TimeRange(start=10.0, end=15.0)],
        )
        assert brand.source == "visual"


class TestFaceGroup:
    def test_create_face_group(self):
        face = FaceGroup(
            id="person_1",
            appearances=[TimeRange(start=0.0, end=5.0), TimeRange(start=20.0, end=30.0)],
            description="Man with beard wearing blue overalls",
        )
        assert face.id == "person_1"
        assert len(face.appearances) == 2


class TestEmotionSegment:
    def test_create_emotion(self):
        em = EmotionSegment(
            start=0.0,
            end=5.0,
            emotion="joy",
            confidence=0.85,
        )
        assert em.emotion == "joy"


class TestTopic:
    def test_create_topic(self):
        topic = Topic(
            name="Automotive Restoration",
            description="Car restoration and mechanical repair",
            confidence=0.9,
            related_keywords=["car", "engine", "restoration"],
        )
        assert len(topic.related_keywords) == 3


class TestKeyword:
    def test_create_keyword(self):
        kw = Keyword(text="restoration", relevance=0.95, count=12)
        assert kw.text == "restoration"


class TestAudioEvent:
    def test_create_audio_event(self):
        event = AudioEvent(
            label="music",
            start=0.0,
            end=30.0,
            confidence=0.88,
        )
        assert event.label == "music"


class TestModerationResult:
    def test_create_moderation(self):
        mod = ModerationResult(is_safe=True, flags=[])
        assert mod.is_safe is True

    def test_unsafe_moderation(self):
        mod = ModerationResult(is_safe=False, flags=["violence"])
        assert mod.is_safe is False
        assert "violence" in mod.flags


class TestVideoInsights:
    def test_create_full_insights(self):
        insights = VideoInsights(
            video=VideoMeta(
                path=Path("/video.mp4"),
                duration=70.0,
                fps=25.0,
                width=1920,
                height=1080,
            ),
            scenes=[Scene(index=0, start=0.0, end=2.5)],
        )
        assert len(insights.scenes) == 1
        assert insights.transcript is None
        assert insights.entities == []
        assert insights.brands == []

    def test_insights_serialization(self):
        insights = VideoInsights(
            video=VideoMeta(
                path=Path("/video.mp4"),
                duration=70.0,
                fps=25.0,
                width=1920,
                height=1080,
            ),
            scenes=[Scene(index=0, start=0.0, end=2.5)],
            keywords=[Keyword(text="car", relevance=0.9, count=5)],
        )
        data = json.loads(insights.model_dump_json())
        assert data["video"]["duration"] == 70.0
        assert data["scenes"][0]["index"] == 0
        assert data["keywords"][0]["text"] == "car"
