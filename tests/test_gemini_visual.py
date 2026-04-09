"""Tests for Gemini visual analysis (Phase 2)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from reverse_engine.gemini.visual import analyze_keyframes
from reverse_engine.models import SceneLabel, OcrResult, BrandMention, FaceGroup


@pytest.fixture
def fake_keyframes(tmp_path):
    paths = []
    for i in range(3):
        p = tmp_path / f"keyframe_{i + 1:03d}.jpg"
        p.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg")
        paths.append(p)
    return paths


@pytest.fixture
def sample_gemini_response():
    return json.dumps({
        "scenes": [
            {
                "scene_index": 0,
                "caption": "A mechanic inspecting a vintage car engine",
                "objects": ["car", "engine", "wrench", "mechanic"],
                "ocr_text": "CAR SOS",
                "brands": ["BMW"],
                "people": [
                    {
                        "description": "Man with beard in blue overalls",
                        "appears_in_scenes": [0, 2],
                    }
                ],
                "shot_type": "medium",
                "mood": "focused",
            },
            {
                "scene_index": 1,
                "caption": "Close-up of a rusty exhaust manifold",
                "objects": ["exhaust", "rust", "metal"],
                "ocr_text": "",
                "brands": [],
                "people": [],
                "shot_type": "close-up",
                "mood": "dramatic",
            },
            {
                "scene_index": 2,
                "caption": "Wide shot of the workshop interior",
                "objects": ["workshop", "tools", "cars"],
                "ocr_text": "DANGER",
                "brands": [],
                "people": [
                    {
                        "description": "Man with beard in blue overalls",
                        "appears_in_scenes": [0, 2],
                    }
                ],
                "shot_type": "wide",
                "mood": "industrial",
            },
        ]
    })


class TestAnalyzeKeyframes:
    def test_returns_structured_insights(self, fake_keyframes, sample_gemini_response):
        mock_client = MagicMock()
        mock_client.analyze_images.return_value = sample_gemini_response

        scenes = [(0.0, 2.5), (2.5, 5.0), (5.0, 8.0)]
        result = analyze_keyframes(mock_client, fake_keyframes, scenes)

        assert "labels" in result
        assert "ocr" in result
        assert "brands" in result
        assert "faces" in result

    def test_extracts_scene_labels(self, fake_keyframes, sample_gemini_response):
        mock_client = MagicMock()
        mock_client.analyze_images.return_value = sample_gemini_response

        scenes = [(0.0, 2.5), (2.5, 5.0), (5.0, 8.0)]
        result = analyze_keyframes(mock_client, fake_keyframes, scenes)

        labels = result["labels"]
        assert len(labels) == 3
        assert isinstance(labels[0], SceneLabel)
        assert labels[0].caption == "A mechanic inspecting a vintage car engine"
        assert "car" in labels[0].objects

    def test_extracts_ocr(self, fake_keyframes, sample_gemini_response):
        mock_client = MagicMock()
        mock_client.analyze_images.return_value = sample_gemini_response

        scenes = [(0.0, 2.5), (2.5, 5.0), (5.0, 8.0)]
        result = analyze_keyframes(mock_client, fake_keyframes, scenes)

        ocr = result["ocr"]
        # Only scenes with non-empty ocr_text
        assert len(ocr) == 2
        assert ocr[0].text == "CAR SOS"
        assert ocr[1].text == "DANGER"

    def test_extracts_brands(self, fake_keyframes, sample_gemini_response):
        mock_client = MagicMock()
        mock_client.analyze_images.return_value = sample_gemini_response

        scenes = [(0.0, 2.5), (2.5, 5.0), (5.0, 8.0)]
        result = analyze_keyframes(mock_client, fake_keyframes, scenes)

        brands = result["brands"]
        assert len(brands) == 1
        assert brands[0].name == "BMW"
        assert brands[0].source == "visual"

    def test_extracts_faces(self, fake_keyframes, sample_gemini_response):
        mock_client = MagicMock()
        mock_client.analyze_images.return_value = sample_gemini_response

        scenes = [(0.0, 2.5), (2.5, 5.0), (5.0, 8.0)]
        result = analyze_keyframes(mock_client, fake_keyframes, scenes)

        faces = result["faces"]
        assert len(faces) == 1
        assert faces[0].description == "Man with beard in blue overalls"
        assert len(faces[0].appearances) == 2

    def test_empty_keyframes(self):
        mock_client = MagicMock()
        result = analyze_keyframes(mock_client, [], [])
        assert result["labels"] == []
        assert result["ocr"] == []
        assert result["brands"] == []
        assert result["faces"] == []
