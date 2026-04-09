"""Extract per-scene video clips and audio segments via ffmpeg."""

import subprocess
from pathlib import Path


def extract_scene_clips(
    video_path: Path,
    scenes: list[tuple[float, float]],
    output_dir: Path,
    max_video_duration: float = 120.0,
    max_audio_duration: float = 80.0,
    video_scale: int = 480,
) -> list[dict[str, Path]]:
    """Cut per-scene video clips and audio segments from the source video.

    Args:
        video_path: Path to the source video file.
        scenes: List of (start_seconds, end_seconds) tuples.
        output_dir: Directory to write clips into.
        max_video_duration: Gemini Embedding 2 video limit (120s).
        max_audio_duration: Gemini Embedding 2 audio limit (80s).
        video_scale: Scale video height to this value (keeps aspect ratio).

    Returns:
        List of dicts with 'video' and 'audio' Path entries per scene.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clips = []
    for i, (start, end) in enumerate(scenes):
        video_dur = min(end - start, max_video_duration)
        audio_dur = min(end - start, max_audio_duration)

        if end - start > max_video_duration:
            print(f"    Warning: scene {i} is {end - start:.1f}s, truncating to {max_video_duration}s for video")

        clip_video = output_dir / f"clip_{i:03d}.mp4"
        clip_audio = output_dir / f"clip_{i:03d}.wav"

        _extract_video_clip(video_path, start, video_dur, clip_video, video_scale)
        _extract_audio_clip(video_path, start, audio_dur, clip_audio)

        clips.append({"video": clip_video, "audio": clip_audio, "scene_index": i})

    return clips


def _extract_video_clip(
    source: Path, start: float, duration: float, output: Path, scale: int
) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-ss", str(start),
            "-i", str(source),
            "-t", str(duration),
            "-vf", f"scale=-2:{scale}",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-an",
            "-y", str(output),
        ],
        check=True,
        capture_output=True,
    )


def _extract_audio_clip(
    source: Path, start: float, duration: float, output: Path
) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-ss", str(start),
            "-i", str(source),
            "-t", str(duration),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "44100",
            "-ac", "1",
            "-y", str(output),
        ],
        check=True,
        capture_output=True,
    )
