"""Tests for Gemini audio analysis (Phase 4)."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from reverse_engine.gemini.audio import analyze_audio
from reverse_engine.models import AudioEvent


@pytest.fixture
def fake_audio(tmp_path):
    audio = tmp_path / "no_vocals.wav"
    audio.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfake-audio")
    return audio


@pytest.fixture
def sample_gemini_response():
    return json.dumps({
        "events": [
            {"label": "music", "start": 0.0, "end": 15.0, "confidence": 0.92},
            {"label": "silence", "start": 15.0, "end": 17.0, "confidence": 0.88},
            {"label": "engine_sound", "start": 17.0, "end": 30.0, "confidence": 0.85},
            {"label": "music", "start": 30.0, "end": 70.0, "confidence": 0.90},
        ]
    })


class TestAnalyzeAudio:
    def test_returns_list_of_events(self, fake_audio, sample_gemini_response):
        mock_client = MagicMock()
        mock_client.analyze_audio.return_value = sample_gemini_response

        result = analyze_audio(mock_client, fake_audio)

        assert len(result) == 4
        assert all(isinstance(e, AudioEvent) for e in result)

    def test_event_properties(self, fake_audio, sample_gemini_response):
        mock_client = MagicMock()
        mock_client.analyze_audio.return_value = sample_gemini_response

        result = analyze_audio(mock_client, fake_audio)

        assert result[0].label == "music"
        assert result[0].start == 0.0
        assert result[0].end == 15.0
        assert result[2].label == "engine_sound"

    def test_empty_response(self, fake_audio):
        mock_client = MagicMock()
        mock_client.analyze_audio.return_value = json.dumps({"events": []})

        result = analyze_audio(mock_client, fake_audio)
        assert result == []
