from pathlib import Path

from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector


def detect_scenes(
    video_path: Path, threshold: float = 27.0
) -> tuple[list[tuple[float, float]], float]:
    """Detect scene boundaries in a video.

    Returns:
        A tuple of (scenes, fps) where scenes is a list of
        (start_seconds, end_seconds) tuples and fps is the video framerate.
    """
    video = open_video(str(video_path))
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video)

    scene_list = scene_manager.get_scene_list()
    fps = video.frame_rate

    scenes = [
        (start.get_seconds(), end.get_seconds())
        for start, end in scene_list
    ]

    return scenes, fps
