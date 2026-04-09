"""Extract keyframes from video at scene midpoints."""

from pathlib import Path

import cv2


def extract_keyframes(
    video_path: Path,
    scenes: list[tuple[float, float]],
    output_dir: Path,
) -> list[Path]:
    """Extract a keyframe image from the midpoint of each scene.

    Args:
        video_path: Path to the source video file.
        scenes: List of (start_seconds, end_seconds) tuples.
        output_dir: Directory to save keyframe JPEGs.

    Returns:
        List of paths to extracted keyframe images.
    """
    if not scenes:
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)

    keyframe_paths: list[Path] = []

    for i, (start, end) in enumerate(scenes):
        midpoint = (start + end) / 2
        frame_number = int(midpoint * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()

        path = output_dir / f"keyframe_{i + 1:03d}.jpg"

        if ret:
            cv2.imwrite(str(path), frame)

        keyframe_paths.append(path)

    cap.release()
    return keyframe_paths
