"""Tests for report generation (Phase 5)."""

import json
from pathlib import Path

import pytest

from reverse_engine.models import (
    Keyword,
    Scene,
    Transcript,
    TranscriptSegment,
    VideoInsights,
    VideoMeta,
)
from reverse_engine.report import export_json, export_srt


@pytest.fixture
def sample_insights():
    return VideoInsights(
        video=VideoMeta(
            path=Path("/video.mp4"),
            duration=70.0,
            fps=25.0,
            width=1920,
            height=1080,
        ),
        scenes=[
            Scene(index=0, start=0.0, end=2.5),
            Scene(index=1, start=2.5, end=5.0),
        ],
        transcript=Transcript(
            language="en",
            segments=[
                TranscriptSegment(start=0.0, end=2.5, text="Hello world."),
                TranscriptSegment(start=2.5, end=5.0, text="This is a test."),
            ],
        ),
        keywords=[Keyword(text="world", relevance=0.9, count=1)],
        summary="A test video.",
    )


class TestExportJson:
    def test_writes_valid_json(self, sample_insights, tmp_path):
        output = tmp_path / "insights.json"
        export_json(sample_insights, output)

        assert output.exists()
        data = json.loads(output.read_text())
        assert data["video"]["duration"] == 70.0
        assert len(data["scenes"]) == 2

    def test_includes_summary(self, sample_insights, tmp_path):
        output = tmp_path / "insights.json"
        export_json(sample_insights, output)

        data = json.loads(output.read_text())
        assert data["summary"] == "A test video."


class TestExportSrt:
    def test_writes_valid_srt(self, sample_insights, tmp_path):
        output = tmp_path / "subtitles.srt"
        export_srt(sample_insights.transcript, output)

        assert output.exists()
        content = output.read_text()
        assert "1\n" in content
        assert "Hello world." in content
        assert "00:00:00,000 --> 00:00:02,500" in content

    def test_srt_numbering(self, sample_insights, tmp_path):
        output = tmp_path / "subtitles.srt"
        export_srt(sample_insights.transcript, output)

        content = output.read_text()
        assert "2\n" in content
        assert "This is a test." in content

    def test_empty_transcript(self, tmp_path):
        output = tmp_path / "subtitles.srt"
        empty = Transcript(language="en", segments=[])
        export_srt(empty, output)

        assert output.exists()
        assert output.read_text() == ""
