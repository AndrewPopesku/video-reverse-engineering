from pathlib import Path

from .download import download_video
from .scenes import detect_scenes
from .stems import extract_audio, separate_stems
from .timeline import build_timeline, export_fcp_xml


def reverse_engineer(
    source: str,
    output_dir: str = "./output",
    scene_threshold: float = 27.0,
) -> dict:
    """Run the full reverse-engineering pipeline.

    Args:
        source: A YouTube URL or local file path.
        output_dir: Directory for all outputs.
        scene_threshold: Sensitivity for scene cut detection (lower = more cuts).

    Returns:
        Dict with keys: video, scenes, stems, timeline_xml.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Get the video file
    source_path = Path(source)
    if source_path.exists():
        video_path = source_path
    else:
        print(f"[1/5] Downloading video...")
        video_path = download_video(source, output_dir)

    # Step 2: Detect scenes
    print(f"[2/5] Detecting scenes...")
    scenes, fps = detect_scenes(video_path, threshold=scene_threshold)
    print(f"       Found {len(scenes)} scenes")

    # Step 3: Extract audio
    print(f"[3/5] Extracting audio...")
    audio_path = extract_audio(video_path, output_dir)

    # Step 4: Separate stems
    print(f"[4/5] Separating audio stems (this may take a while)...")
    stems = separate_stems(audio_path, output_dir)

    # Step 5: Build and export timeline
    print(f"[5/5] Building timeline...")
    total_duration = scenes[-1][1] if scenes else None
    timeline = build_timeline(video_path, scenes, stems, fps, total_duration)
    xml_path = export_fcp_xml(timeline, output_dir / "timeline.xml")

    print(f"Done! Timeline exported to {xml_path}")

    return {
        "video": video_path,
        "scenes": scenes,
        "stems": stems,
        "timeline_xml": xml_path,
    }
