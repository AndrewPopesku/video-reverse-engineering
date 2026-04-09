"""Tests for keyframe extraction (Phase 0)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from reverse_engine.keyframes import extract_keyframes


@pytest.fixture
def fake_video(tmp_path):
    """Create a fake video file path."""
    video = tmp_path / "test.mp4"
    video.touch()
    return video


@pytest.fixture
def sample_scenes():
    return [
        (0.0, 2.5),
        (2.5, 5.0),
        (5.0, 8.0),
    ]


class TestExtractKeyframes:
    def test_returns_list_of_paths(self, fake_video, sample_scenes, tmp_path):
        output_dir = tmp_path / "keyframes"
        # Mock cv2.VideoCapture
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch("reverse_engine.keyframes.cv2") as mock_cv2:
            mock_cap = MagicMock()
            mock_cv2.VideoCapture.return_value = mock_cap
            mock_cap.get.return_value = 25.0  # fps
            mock_cap.set.return_value = True
            mock_cap.read.return_value = (True, mock_frame)
            mock_cv2.imwrite.return_value = True

            result = extract_keyframes(fake_video, sample_scenes, output_dir)

        assert len(result) == 3
        assert all(isinstance(p, Path) for p in result)

    def test_keyframe_naming(self, fake_video, sample_scenes, tmp_path):
        output_dir = tmp_path / "keyframes"
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch("reverse_engine.keyframes.cv2") as mock_cv2:
            mock_cap = MagicMock()
            mock_cv2.VideoCapture.return_value = mock_cap
            mock_cap.get.return_value = 25.0
            mock_cap.set.return_value = True
            mock_cap.read.return_value = (True, mock_frame)
            mock_cv2.imwrite.return_value = True

            result = extract_keyframes(fake_video, sample_scenes, output_dir)

        assert result[0].name == "keyframe_001.jpg"
        assert result[1].name == "keyframe_002.jpg"
        assert result[2].name == "keyframe_003.jpg"

    def test_empty_scenes_returns_empty(self, fake_video, tmp_path):
        output_dir = tmp_path / "keyframes"
        result = extract_keyframes(fake_video, [], output_dir)
        assert result == []

    def test_extracts_midpoint_of_scene(self, fake_video, tmp_path):
        """Keyframe should be taken from the midpoint of each scene."""
        output_dir = tmp_path / "keyframes"
        scenes = [(0.0, 4.0)]  # midpoint = 2.0s
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch("reverse_engine.keyframes.cv2") as mock_cv2:
            mock_cap = MagicMock()
            mock_cv2.VideoCapture.return_value = mock_cap
            mock_cap.get.return_value = 25.0
            mock_cap.set.return_value = True
            mock_cap.read.return_value = (True, mock_frame)
            mock_cv2.imwrite.return_value = True
            mock_cv2.CAP_PROP_FPS = 5
            mock_cv2.CAP_PROP_POS_FRAMES = 1

            extract_keyframes(fake_video, scenes, output_dir)

            # Should seek to midpoint: 2.0s * 25fps = frame 50
            mock_cap.set.assert_called_with(mock_cv2.CAP_PROP_POS_FRAMES, 50)
