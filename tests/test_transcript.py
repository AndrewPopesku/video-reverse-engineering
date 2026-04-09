"""Tests for transcription module (Phase 1)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from reverse_engine.transcript import transcribe


@pytest.fixture
def fake_audio(tmp_path):
    audio = tmp_path / "vocals.wav"
    audio.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfake-audio-data")
    return audio


class TestTranscribe:
    def test_returns_transcript_model(self, fake_audio):
        mock_segment = MagicMock()
        mock_segment.start = 0.0
        mock_segment.end = 2.5
        mock_segment.text = " Hello world"
        mock_segment.avg_logprob = -0.3
        mock_segment.words = [
            MagicMock(start=0.0, end=0.5, word=" Hello", probability=0.95),
            MagicMock(start=0.6, end=1.0, word=" world", probability=0.92),
        ]

        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.98

        with patch("reverse_engine.transcript.WhisperModel") as MockModel:
            mock_model = MagicMock()
            MockModel.return_value = mock_model
            mock_model.transcribe.return_value = ([mock_segment], mock_info)

            result = transcribe(fake_audio)

        assert result.language == "en"
        assert len(result.segments) == 1
        assert result.segments[0].text == "Hello world"
        assert len(result.segments[0].words) == 2

    def test_full_text_assembly(self, fake_audio):
        seg1 = MagicMock(
            start=0.0, end=2.0, text=" First segment.", avg_logprob=-0.2, words=[]
        )
        seg2 = MagicMock(
            start=2.0, end=4.0, text=" Second segment.", avg_logprob=-0.3, words=[]
        )
        mock_info = MagicMock(language="en", language_probability=0.99)

        with patch("reverse_engine.transcript.WhisperModel") as MockModel:
            mock_model = MagicMock()
            MockModel.return_value = mock_model
            mock_model.transcribe.return_value = ([seg1, seg2], mock_info)

            result = transcribe(fake_audio)

        assert result.full_text == "First segment. Second segment."

    def test_uses_word_timestamps(self, fake_audio):
        """Verify we request word-level timestamps from whisper."""
        mock_info = MagicMock(language="en", language_probability=0.99)

        with patch("reverse_engine.transcript.WhisperModel") as MockModel:
            mock_model = MagicMock()
            MockModel.return_value = mock_model
            mock_model.transcribe.return_value = ([], mock_info)

            transcribe(fake_audio)

            call_kwargs = mock_model.transcribe.call_args
            assert call_kwargs[1].get("word_timestamps") is True

    def test_custom_model_size(self, fake_audio):
        mock_info = MagicMock(language="en", language_probability=0.99)

        with patch("reverse_engine.transcript.WhisperModel") as MockModel:
            mock_model = MagicMock()
            MockModel.return_value = mock_model
            mock_model.transcribe.return_value = ([], mock_info)

            transcribe(fake_audio, model_size="medium")

            MockModel.assert_called_once_with("medium")
