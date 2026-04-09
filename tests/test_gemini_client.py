"""Tests for Gemini API client wrapper (Phase 0)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from reverse_engine.gemini.client import GeminiClient


class TestGeminiClient:
    def test_init_with_api_key(self):
        with patch("reverse_engine.gemini.client.genai") as mock_genai:
            client = GeminiClient(api_key="test-key")
            mock_genai.Client.assert_called_once_with(api_key="test-key")

    def test_init_from_env(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "env-key")
        with patch("reverse_engine.gemini.client.genai") as mock_genai:
            client = GeminiClient()
            mock_genai.Client.assert_called_once_with(api_key="env-key")

    def test_init_no_key_raises(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            with patch("reverse_engine.gemini.client.genai"):
                GeminiClient()

    def test_default_model(self):
        with patch("reverse_engine.gemini.client.genai"):
            client = GeminiClient(api_key="test-key")
            assert client.model == "gemini-2.5-flash"

    def test_custom_model(self):
        with patch("reverse_engine.gemini.client.genai"):
            client = GeminiClient(api_key="test-key", model="gemini-2.5-pro")
            assert client.model == "gemini-2.5-pro"

    def test_analyze_images_calls_api(self, tmp_path):
        img1 = tmp_path / "img1.jpg"
        img2 = tmp_path / "img2.jpg"
        img1.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg")
        img2.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg")

        with patch("reverse_engine.gemini.client.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            mock_response = MagicMock()
            mock_response.text = '{"results": []}'
            mock_client.models.generate_content.return_value = mock_response

            client = GeminiClient(api_key="test-key")
            result = client.analyze_images(
                images=[img1, img2],
                prompt="Describe these images",
            )

            mock_client.models.generate_content.assert_called_once()
            assert result == '{"results": []}'

    def test_analyze_text_calls_api(self):
        with patch("reverse_engine.gemini.client.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            mock_response = MagicMock()
            mock_response.text = '{"entities": []}'
            mock_client.models.generate_content.return_value = mock_response

            client = GeminiClient(api_key="test-key")
            result = client.analyze_text(
                text="Some transcript text",
                prompt="Extract entities",
            )

            mock_client.models.generate_content.assert_called_once()
            assert result == '{"entities": []}'

    def test_analyze_audio_calls_api(self, tmp_path):
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfake")

        with patch("reverse_engine.gemini.client.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            mock_response = MagicMock()
            mock_response.text = '{"events": []}'
            mock_client.models.generate_content.return_value = mock_response

            client = GeminiClient(api_key="test-key")
            result = client.analyze_audio(
                audio_path=audio,
                prompt="Classify audio events",
            )

            mock_client.models.generate_content.assert_called_once()
            assert result == '{"events": []}'
